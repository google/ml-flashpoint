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
from typing import Any, ClassVar, Dict, Optional

from ml_flashpoint.checkpoint_object_manager.buffer_io import BufferIO
from ml_flashpoint.checkpoint_object_manager.buffer_metadata import METADATA_SIZE
from ml_flashpoint.checkpoint_object_manager.buffer_object.buffer_object_ext import BufferObject
from ml_flashpoint.core.buffer_pool import BufferPool
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId, CheckpointObjectId
from ml_flashpoint.core.mlf_logging import get_logger

_LOGGER = get_logger(__name__)


class CheckpointObjectManager:
    """Checkpoint object manager is the main API for buffer and file-related operations.

    An instance should be reused across a process (within a rank), for caching and "global"
    awareness of buffers (within a rank).
    """

    # Class-level registry for BufferPools in the worker process.
    # Key: pool_dir_path (str), Value: BufferPool instance
    _worker_pools: ClassVar[Dict[str, BufferPool]] = {}
    _worker_pools_lock = threading.Lock()

    def __init__(self, pool_config: Optional[Dict[str, Any]] = None):
        """Initializes the CheckpointObjectManager.

        Args:
            pool_config: Optional configuration for the BufferPool.
        """
        self.pool_config = pool_config
        # Buffer pool is used for worker process only.
        self._worker_process_buffer_pool: Optional[BufferPool] = None
        self._worker_process_pool_lock = threading.Lock()

    def set_pool_config(self, pool_config: Dict[str, Any]):
        """Sets the configuration for the BufferPool.

        Args:
            pool_config: Configuration dictionary for the BufferPool.
        """
        self.pool_config = pool_config

    def __getstate__(self):
        state = self.__dict__.copy()
        # Exclude lock and buffer pool from pickled state
        del state["_worker_process_pool_lock"]
        # buffer_pool is not picklable (contains lock), and we want to re-init it in worker anyway
        state["_worker_process_buffer_pool"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-initialize lock and ensure buffer pool is None
        self._worker_process_pool_lock = threading.Lock()
        self._worker_process_buffer_pool = None

    def _get_or_create_buffer_pool(self) -> Optional[BufferPool]:
        """Lazily initializes and returns the BufferPool instance.

        Check the class-level registry first to reuse existing pools in this process.
        """
        # 1. Fast path: instance already has it
        if self._worker_process_buffer_pool:
            return self._worker_process_buffer_pool

        if not self.pool_config:
            return None

        pool_dir = self.pool_config.get("pool_dir_path")
        if not pool_dir:
            _LOGGER.warning("pool_config missing 'pool_dir_path', cannot use worker registry.")
            return None

        # 2. Registry lookup / Creation
        with self._worker_pools_lock:
            # Re-check instance variable just in case
            if self._worker_process_buffer_pool:
                return self._worker_process_buffer_pool

            # Check registry
            if pool_dir in self._worker_pools:
                _LOGGER.debug("Reusing existing BufferPool for %s from worker registry.", pool_dir)
                self._worker_process_buffer_pool = self._worker_pools[pool_dir]
                return self._worker_process_buffer_pool

            # Create new
            try:
                _LOGGER.info("Initializing BufferPool with config: %s", self.pool_config)
                pool = BufferPool(**self.pool_config)
                self._worker_pools[pool_dir] = pool
                self._worker_process_buffer_pool = pool
            except Exception as e:
                _LOGGER.error("Failed to initialize BufferPool: %s", e)
                # Keep _buffer_pool as None
                pass

        return self._worker_process_buffer_pool

    def teardown_pool(self):
        """Teardown the BufferPool if it exists and remove from registry."""
        # If we don't have a local reference, try to find it via config (e.g. after unpickling)
        pool_to_teardown = self._worker_process_buffer_pool
        pool_dir = None

        if self.pool_config:
            pool_dir = self.pool_config.get("pool_dir_path")

        with self._worker_pools_lock:
            if pool_dir and pool_dir in self._worker_pools:
                pool_to_teardown = self._worker_pools.pop(pool_dir)

            # Also clear local reference
            self._worker_process_buffer_pool = None

        if pool_to_teardown:
            try:
                pool_to_teardown.teardown()
            except Exception as e:
                _LOGGER.debug("Failed to teardown BufferPool: %s", e)

    def acquire_buffer(self, object_id: CheckpointObjectId, buffer_size: int, overwrite: bool = True) -> "BufferIO":
        """Acquires a buffer, preferring the BufferPool if available.

        This method attempts to acquire a buffer from the rank level BufferPool. If the pool
        is not initialized or is exhausted, it falls back to creating a standalone
        BufferObject directly at the `object_id` path.

        Args:
            object_id: A unique identifier (logical path) for the new buffer object.
            buffer_size: The desired size of the buffer in bytes.
            overwrite: If True, allows overwriting an existing object. Defaults to True.

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
                    os.remove(str(object_id))
                else:
                    raise FileExistsError(f"File {object_id} already exists and overwrite=False")

            pool = self._get_or_create_buffer_pool()
            if pool:
                # Pool manages the physical creation/resizing AND the logical link (symlink) creation.
                buffer_io = pool.acquire(associated_symlink=str(object_id))

                _LOGGER.debug("Acquired buffer for '%s'", object_id)

                return buffer_io
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

    def close_buffer(self, buffer_io: BufferIO, truncate: bool = True) -> None:
        """Closes the provided BufferIO object.

        This is a helper method that directly calls the .close() method on the
        given buffer object, providing consistent logging.

        Args:
            buffer_io: The BufferIO instance to close.
            truncate: Passed to the buffer's close() method.

        Raises:
            The original exception from the underlying buffer's close method if the operation fails.
        """
        if not isinstance(buffer_io, BufferIO) or buffer_io.closed:
            _LOGGER.warning("Attempted to close an invalid or already-closed buffer. Ignoring.")
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
