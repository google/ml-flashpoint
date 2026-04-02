# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import threading
from typing import ClassVar, Dict, Optional

from ml_flashpoint.checkpoint_object_manager.buffer_io import BufferIO
from ml_flashpoint.checkpoint_object_manager.buffer_metadata import METADATA_SIZE
from ml_flashpoint.checkpoint_object_manager.buffer_object.buffer_object_ext import BufferObject, BufferPool
from ml_flashpoint.core.buffer_pool import BufferPoolConfig
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId, CheckpointObjectId
from ml_flashpoint.core.mlf_logging import get_logger

_LOGGER = get_logger(__name__)


class CheckpointObjectManager:
    """Checkpoint object manager is the main API for buffer and file-related operations.

    An instance should be reused across a process (within a rank), for caching and "global"
    awareness of buffers (within a rank).
    """

    # Class-level registry for BufferPools in the worker process.
    # Maps pool type (string) to BufferPool instance.
    _worker_pools: ClassVar[Dict[str, BufferPool]] = {}
    _worker_pool_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, local_pool_config: Optional[BufferPoolConfig] = None, repl_pool_config: Optional[BufferPoolConfig] = None):
        """Initializes the CheckpointObjectManager.

        Args:
            local_pool_config: Optional configuration for the local BufferPool.
            repl_pool_config: Optional configuration for the replication BufferPool.
        """
        self._local_pool_config = local_pool_config
        self._repl_pool_config = repl_pool_config

    def _get_or_create_buffer_pool(self, pool_type: str = "local") -> Optional[BufferPool]:
        """Lazily initializes and returns the specified BufferPool instance.

        Check the class-level registry first to reuse existing pools in this process.
        """
        # 1. Fast path: check class var directly
        if pool_type in CheckpointObjectManager._worker_pools:
            return CheckpointObjectManager._worker_pools[pool_type]

        config = self._local_pool_config if pool_type == "local" else self._repl_pool_config
        if not config:
            return None

        # 2. Registry lookup / Creation
        with CheckpointObjectManager._worker_pool_lock:
            if pool_type in CheckpointObjectManager._worker_pools:
                _LOGGER.debug("Reusing existing %s BufferPool from worker registry.", pool_type)
                return CheckpointObjectManager._worker_pools[pool_type]

            # Create new
            try:
                _LOGGER.info("Initializing C++ BufferPool for %s with config: %s", pool_type, config)
                shm_suffix = "local" if pool_type == "local" else "repl"
                pool = BufferPool(
                    shm_name=f"/mlf_buffer_pool_rank_{config.rank}_{shm_suffix}",
                    pool_dir=config.pool_dir_path,
                    rank=config.rank,
                    num_buffers=config.num_buffers,
                    buffer_size=config.buffer_size,
                )
                CheckpointObjectManager._worker_pools[pool_type] = pool
            except Exception:
                _LOGGER.exception("Failed to initialize BufferPool for %s", pool_type)
                pass

        return CheckpointObjectManager._worker_pools.get(pool_type)

    def teardown_pool(self):
        """Teardown the BufferPools if they exist and remove from registry."""
        with CheckpointObjectManager._worker_pool_lock:
            if CheckpointObjectManager._worker_pools:
                _LOGGER.debug("Clearing BufferPools from registry.")
                CheckpointObjectManager._worker_pools.clear()

    @property
    def replication_pool_shm_name(self) -> str:
        """Returns the shared memory name for the replication pool."""
        if not self._repl_pool_config:
            return ""
        return f"/mlf_buffer_pool_rank_{self._repl_pool_config.rank}_repl"

    def acquire_buffer(self, object_id: CheckpointObjectId, buffer_size: int, overwrite: bool = True, use_replication_pool: bool = False) -> "BufferIO":
        """Acquires a buffer, preferring the BufferPool if available.

        This method attempts to acquire a buffer from the rank level BufferPool. If the pool
        is not initialized or is exhausted, it falls back to creating a standalone
        BufferObject directly at the `object_id` path.

        Note:
            If a buffer is acquired from the pool, its actual capacity may be smaller than
            the requested `buffer_size` if a smaller buffer is reused. But resize will happen
            when writing to the buffer.

        Args:
            object_id: A unique identifier (logical path) for the new buffer object.
            buffer_size: The desired size of the buffer in bytes.
            overwrite: If True, allows overwriting an existing object. Defaults to True.
            use_replication_pool: If True, uses the replication pool instead of the local pool.

        Returns:
            A BufferIO instance.

        Raises:
            ValueError: If the provided buffer_size is not a positive integer.
            RuntimeError: If the buffer cannot be created or acquired.
        """
        # Validate the buffer size argument.
        if buffer_size <= 0:
            _LOGGER.error("Failed to acquire buffer: buffer_size must be a positive integer, but got %d.", buffer_size)
            raise ValueError("Buffer size must be a positive integer.")

        buffer_size += METADATA_SIZE
        _LOGGER.debug(
            "Acquiring buffer for object_id='%s' with size=%d bytes (includes overhead), overwrite=%s.",
            object_id,
            buffer_size,
            overwrite,
        )

        # 1. Try to acquire from BufferPool
        try:
            # Ensure parent dir exists for the link
            os.makedirs(os.path.dirname(str(object_id)), exist_ok=True)

            # Remove existing link/file if overwrite is True
            if os.path.exists(str(object_id)):
                if overwrite:
                    self.delete_object(object_id)
                else:
                    raise FileExistsError(f"File {object_id} already exists and overwrite=False")

            pool_type = "repl" if use_replication_pool else "local"
            pool = self._get_or_create_buffer_pool(pool_type)
            if pool:
                # Pool manages the physical creation/resizing AND the logical link (symlink) creation.
                buffer_path = pool.acquire(associated_symlink=str(object_id))
                _LOGGER.debug("Acquired pool buffer '%s' for '%s'", buffer_path, object_id)
                # Create a writable BufferObject pointing to the physical path in the pool
                buffer_obj = BufferObject(buffer_path, buffer_size, overwrite=True)
                return BufferIO(buffer_obj)
            else:
                _LOGGER.debug(
                    "BufferPool not configured or validation failed. Falling back to standalone buffer creation."
                )
        except RuntimeError:
            _LOGGER.debug("BufferPool exhausted. Falling back to standalone buffer creation.")

        # 2. Fallback: Create a standalone BufferObject
        try:
            _LOGGER.debug("Instantiating standalone C++ BufferObject for '%s'.", object_id)
            buffer_obj = BufferObject(str(object_id), buffer_size, overwrite)
            return BufferIO(buffer_obj)

        except Exception as e:
            _LOGGER.exception("Failed to create buffer for object_id='%s'", object_id)
            raise RuntimeError(f"Failed to create buffer for '{object_id}'") from e

    def get_buffer(
        self,
        object_id: CheckpointObjectId,
    ) -> Optional["BufferIO"]:
        """Opens a buffer for a pre-existing object, strictly in read-only mode.

        Args:
            object_id: The unique identifier (e.g., file path) of the buffer.

        Returns:
            A read-only BufferIO instance on success.
            Returns None if the object does not exist, is a directory, or fails to open
            due to other I/O errors (e.g., permissions).
        """

        _LOGGER.info("Opening buffer for '%s' in read-only mode.", object_id)

        try:
            buffer_obj = BufferObject(str(object_id))
            buffer_io = BufferIO(buffer_obj)
            return buffer_io

        except (OSError, ValueError, RuntimeError):
            _LOGGER.exception("Could not open buffer '%s'. Returning None.", object_id)
            return None

    def close_buffer(self, buffer_io: BufferIO, skip_close_if_symlink: bool = False, truncate: bool = True) -> None:
        """Closes the provided BufferIO object.

        This is a helper method that directly calls the .close() method on the
        given buffer object, providing consistent logging.

        Args:
            buffer_io: The BufferIO instance to close.
            skip_close_if_symlink: If True, the buffer will not be closed if it is a symlink.
            truncate: Passed to the buffer's close() method.

        Raises:
            The original exception from the underlying buffer's close method if the operation fails.
        """
        if not isinstance(buffer_io, BufferIO) or buffer_io.closed:
            _LOGGER.warning("Attempted to close an invalid or already-closed buffer. Ignoring.")
            return
        # TODO: Use a separate BufferIO type to distinguish between pooled and standalone buffers.
        # This is a temporary workaround to prevent closing pooled buffers.
        if skip_close_if_symlink and os.path.islink(buffer_io.buffer_obj.get_id()):
            _LOGGER.debug("Buffer for object_id='%s' is a symlink. Ignoring.", buffer_io.buffer_obj.get_id())
            return

        try:
            object_id = buffer_io.buffer_obj.get_id()
            _LOGGER.info("Manager closing buffer for object_id='%s' with truncate=%s.", object_id, truncate)
            buffer_io.close(truncate=truncate)
            _LOGGER.debug("Buffer for object_id='%s' has been successfully closed.", object_id)
        except Exception:
            _LOGGER.exception("An error occurred while closing buffer for object_id='%s'.", object_id)
            raise

    def delete_container(self, container_id: CheckpointContainerId) -> None:
        """Recursively deletes a container directory from the filesystem.

        Args:
            container_id: The path of the container directory to be deleted.

        Raises:
            OSError: If the directory deletion fails due to filesystem errors
                    (e.g., permissions).
        """
        container_id = str(container_id)
        _LOGGER.info("Starting deletion process for container: '%s'", container_id)
        try:
            if os.path.isdir(container_id):
                # Use shutil.rmtree for recursive deletion.
                shutil.rmtree(container_id)
                _LOGGER.info("Successfully deleted container directory: '%s'", container_id)
            else:
                # This is not an error; the directory might have already been deleted.
                _LOGGER.warning(
                    "Container '%s' is not a directory or does not exist. Nothing to delete from disk.", container_id
                )
        except OSError:
            # This catches filesystem errors (e.g., permissions) during deletion.
            _LOGGER.exception("Error deleting container directory '%s'", container_id)
            # Re-raise to notify the caller that the deletion failed.
            raise

    def delete_object(self, object_id: CheckpointObjectId) -> None:
        """Deletes a checkpoint object (file or symlink) from the filesystem.

        Args:
            object_id: The unique identifier (path) of the object to be deleted.

        Raises:
            OSError: If deletion fails due to filesystem errors.
        """
        object_path = str(object_id)
        _LOGGER.debug("Attempting to delete object: '%s'", object_path)
        try:
            if os.path.lexists(object_path):
                os.remove(object_path)
                _LOGGER.debug("Successfully deleted object: '%s'", object_path)
            else:
                _LOGGER.warning("Object '%s' does not exist. Nothing to delete.", object_path)
        except OSError:
            _LOGGER.exception("Error deleting object '%s'", object_path)
            raise
