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
from typing import Optional

import cupy

from ml_flashpoint.checkpoint_object_manager.buffer_io import BufferIO
from ml_flashpoint.checkpoint_object_manager.buffer_metadata import METADATA_SIZE
from ml_flashpoint.checkpoint_object_manager.buffer_object.buffer_object_ext import BufferObject
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId, CheckpointObjectId
from ml_flashpoint.core.mlf_logging import get_logger

_LOGGER = get_logger(__name__)


class CheckpointObjectManager:
    """Checkpoint object manager is the main API for buffer and file-related operations.

    An instance should be reused across a process (within a rank), for caching and "global"
    awareness of buffers (within a rank).
    """

    def create_buffer(self, object_id: CheckpointObjectId, buffer_size: int, overwrite: bool = False) -> "BufferIO":
        """Creates a new underlying C++ BufferObject and wraps it in a BufferIO stream.

        This method handles the instantiation of the low-level buffer and its
        registration within the manager's tracking map.

        Args:
            object_id: A unique identifier for the new buffer object.
            buffer_size: The desired size of the buffer in bytes.
            overwrite: If True, allows overwriting an existing object. Defaults to False.

        Returns:
            A new BufferIO instance wrapping the created buffer on success.

        Raises:
            ValueError: If the provided buffer_size is not a positive integer.
            RuntimeError: If the underlying C++ BufferObject or the BufferIO wrapper
                        fails to initialize for any reason (e.g., file system errors).
        """
        # Validate the buffer size argument.
        if buffer_size <= 0:
            _LOGGER.error("Failed to create buffer: buffer_size must be a positive integer, but got %d.", buffer_size)
            raise ValueError("Buffer size must be a positive integer.")

        buffer_size += METADATA_SIZE
        _LOGGER.info(
            "Creating a new buffer for object_id='%s' with size=%d bytes (includes additional overhead), "
            + "overwrite=%s.",
            object_id,
            buffer_size,
            overwrite,
        )

        try:
            # Instantiate the underlying C++ BufferObject.
            _LOGGER.debug("Instantiating C++ BufferObject for '%s'.", object_id)
            buffer_obj = BufferObject(str(object_id), buffer_size, overwrite)

            # Wrap the C++ object in our Python BufferIO file-like object.
            _LOGGER.debug("Wrapping C++ object in BufferIO for '%s'.", object_id)
            buffer_io = BufferIO(buffer_obj)

        except Exception as e:
            # This block catches any error during the C++ object creation or the
            # BufferIO wrapper initialization (e.g., from pybind11 or BufferIO's __init__).
            _LOGGER.exception("An unexpected error occurred during buffer creation for object_id='%s'", object_id)
            # Re-raise it as a more generic RuntimeError to signal a fundamental
            # failure in the creation process.
            raise RuntimeError(f"Failed to create and wrap buffer for '{object_id}'") from e

        _LOGGER.debug("Successfully created and wrapped buffer for '%s'.", object_id)

        return buffer_io

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
