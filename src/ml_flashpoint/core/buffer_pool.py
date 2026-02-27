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


import logging
import os
import threading
from typing import Dict, List, Optional

from ml_flashpoint.checkpoint_object_manager.buffer_io import BufferIO
from ml_flashpoint.checkpoint_object_manager.buffer_metadata import METADATA_SIZE
from ml_flashpoint.checkpoint_object_manager.buffer_object.buffer_object_ext import BufferObject
from ml_flashpoint.core.mlf_logging import get_logger
from ml_flashpoint.core.utils import log_execution_time

_LOGGER = get_logger(__name__)

# Constants for buffer resizing
PADDING_SIZE = 1024 * 1024
RESIZE_FACTOR = 1.1


class BufferIOProxy:
    """Proxies a BufferIO object to prevent it from being closed by the client.

    This allows the BufferPool to reuse the underlying BufferIO object even if
    the client (e.g. CheckpointSaver) calls close() on it.
    """

    def __init__(self, buffer_io: BufferIO):
        self._buffer_io = buffer_io
        self._closed = False

    def __getattr__(self, name):
        # Delegate attribute access to the underlying BufferIO object
        if self._closed:
            _LOGGER.warning("BufferIOProxy: Accessing closed buffer")
        return getattr(self._buffer_io, name)

    @log_execution_time(logger=_LOGGER, name="close", level=logging.INFO)
    def close(self, truncate: bool = True):
        """Marks the proxy as closed, releasing the buffer back to the pool.

        This method does NOT close the underlying BufferIO object, allowing it to be
        reused by the BufferPool. Ideally, it truncates the buffer to the written size
        to save space.

        Args:
            truncate: If True, truncates the underlying buffer to the size of the
                written data (plus metadata) before releasing it.
        """
        if self._closed:
            return

        self._closed = True

        _LOGGER.info("Closing BufferIOProxy object...")
        if truncate and not self._buffer_io.is_readonly:
            try:
                final_data_len = self._buffer_io._metadata.len_written_data
                truncate_size = METADATA_SIZE + final_data_len

                current_size = len(self._buffer_io._mv)
                if truncate_size != current_size:
                    _LOGGER.info(
                        "BufferIOProxy: Truncating reusable buffer from %d to %d bytes", current_size, truncate_size
                    )
                    self._buffer_io.resize(truncate_size)
            except Exception:
                _LOGGER.warning("BufferIOProxy: Failed to truncate buffer during close.", exc_info=True)

    @property
    def closed(self) -> bool:
        return self._closed or self._buffer_io.closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _auto_resize(self, required_bytes: int):
        """Resizes the buffer to accommodate required_bytes from current position."""
        current_size = self._buffer_io.buffer_obj.get_capacity()
        current_pos = self._buffer_io.tell()
        required_size = METADATA_SIZE + current_pos + required_bytes

        # Use 1MB padding or 10% growth (whichever is larger) to amortize resize costs
        new_size = int(max(current_size * RESIZE_FACTOR, required_size + PADDING_SIZE))

        _LOGGER.info(
            "BufferIOProxy: Auto-resizing from %d to %d bytes (required: %d)",
            current_size,
            new_size,
            required_size,
        )
        self._buffer_io.resize(new_size)

    def write(self, data: bytes) -> int:
        data_len = len(data)
        current_pos = self._buffer_io.tell()
        current_capacity = self._buffer_io.buffer_obj.get_capacity()
        required_capacity = METADATA_SIZE + current_pos + data_len

        if required_capacity > current_capacity:
            self._auto_resize(data_len)

        return self._buffer_io.write(data)

    def next_buffer_slice(self, size: int) -> memoryview:
        current_pos = self._buffer_io.tell()
        current_capacity = self._buffer_io.buffer_obj.get_capacity()
        required_capacity = METADATA_SIZE + current_pos + size

        if required_capacity > current_capacity:
            self._auto_resize(size)

        return self._buffer_io.next_buffer_slice(size)


class BufferPool:
    """Singleton class to manage a pool of persistent BufferIO objects.

    This class maintains a pool of buffer file paths in a dedicated directory.
    When a buffer is requested via `acquire`, it returns a BufferIO object pointing
    to a free buffer file (reusing it if available) or creates a new one.

    Buffers are strictly managed by file paths. The underlying file descriptors
    are closed when BufferIO.close() is called by the user, but the file remains
    on disk for reuse. Garbage collection reclaims these files when their
    associated symlinks (checkpoints) are deleted.
    """

    def __init__(
        self,
        pool_dir_path: str,
        rank: int = 0,
        num_buffers: int = 3,
        buffer_size: int = 0,
    ):
        """Initializes the BufferPool.

        Args:
            pool_dir_path: The directory path where buffer files will be stored.
            rank: The rank of the process using this pool (used for naming).
            num_buffers: The fixed number of buffers to allocate in the pool.
            buffer_size: The initial size of each buffer in bytes.
        """
        self.pool_dir = pool_dir_path
        self.rank = rank
        self.num_buffers = num_buffers
        self.buffer_size = buffer_size
        self.free_buffers: List[BufferIO] = []
        # active_buffers maps buffer_path (str) -> (BufferIO, associated_symlink_path (str))
        self.active_buffers: Dict[str, tuple[BufferIO, str]] = {}
        self._lock = threading.Lock()

        if self.pool_dir and self.buffer_size > 0:
            try:
                os.makedirs(self.pool_dir, exist_ok=True)
                _LOGGER.info("BufferPool initialized with directory: %s", self.pool_dir)
                self._preallocate_buffers()
            except OSError as e:
                _LOGGER.error("Failed to create/populate BufferPool: %s", e)
                raise

    @log_execution_time(logger=_LOGGER, name="acquire", level=logging.INFO)
    def acquire(self, associated_symlink: Optional[str] = None) -> BufferIOProxy:
        """Acquires a buffer from the pool.

        Args:
            associated_symlink: The path to the symlink that will point to this buffer.
                If provided, GC will check this symlink. If None, the buffer is protected
                from GC until the pool is torn down.

        Returns:
            A BufferIO object (wrapped in a BufferIOProxy).

        Raises:
            RuntimeError: If the pool is exhausted and no buffers can be reclaimed.
        """
        with self._lock:
            # 0. Opportunistic GC
            self._gc()

            # 1. Try to find a free buffer
            if self.free_buffers:
                _LOGGER.info("Number of free buffers: %d", len(self.free_buffers))
                free_buffer = self.free_buffers.pop()
                _LOGGER.info(
                    "Acquired buffer from pool for %s, the buffer name is %s",
                    associated_symlink,
                    free_buffer.buffer_obj.get_id(),
                )
                try:
                    buf_io = self._reuse_buffer(free_buffer, associated_symlink)
                    return BufferIOProxy(buf_io)
                except Exception:
                    # If reuse fails (e.g. symlink creation), we must put the buffer back!
                    _LOGGER.exception("Failed to reuse buffer for '%s'. Releasing buffer.", associated_symlink)
                    # _reuse_buffer might have added it to active_buffers, so ensure it's removed.
                    buffer_path = free_buffer.buffer_obj.get_id()
                    if buffer_path in self.active_buffers:
                        del self.active_buffers[buffer_path]
                    self.free_buffers.append(free_buffer)
                    raise

            # 2. If no free buffer, raise error (Fixed Size Pool)
            raise RuntimeError(f"BufferPool exhausted. All {self.num_buffers} buffers are in use.")

    def _gc(self) -> None:
        """Releases buffers whose associated symlinks no longer exist."""
        # Check all active buffers
        to_release = []
        for buffer_path, (buf_io, symlink) in self.active_buffers.items():
            if symlink and not os.path.exists(symlink):
                to_release.append(buffer_path)

        if to_release:
            _LOGGER.info("Garbage collecting %d buffers whose symlinks are gone.", len(to_release))
            for buffer_path in to_release:
                _LOGGER.info("Garbage collecting buffer %s", buffer_path)
                buf_io, _ = self.active_buffers.pop(buffer_path)
                self.free_buffers.append(buf_io)

    def teardown(self) -> None:
        """Closes all buffers and clears the pool.

        The files remain on disk (persistent).
        If we wanted to adhere to strict cleanup of tests, we might delete them?
        But persistent pool implies persistence.
        Use cases using temp dir will cleanup the dir.
        """
        with self._lock:
            _LOGGER.info(
                "Tearing down BufferPool. Closing %d free and %d active buffers.",
                len(self.free_buffers),
                len(self.active_buffers),
            )

            for buf in self.free_buffers:
                try:
                    buf.close(truncate=True)
                except Exception:
                    _LOGGER.warning("BufferIOProxy: Failed to close buffer during teardown.", exc_info=True)
            self.free_buffers.clear()

            for buffer_path, (buf, _) in self.active_buffers.items():
                try:
                    buf.close(truncate=True)
                except Exception:
                    _LOGGER.warning("BufferIOProxy: Failed to close buffer during teardown.", exc_info=True)
            self.active_buffers.clear()

    @log_execution_time(logger=_LOGGER, name="_reuse_buffer", level=logging.INFO)
    def _reuse_buffer(self, buffer_io: BufferIO, symlink: Optional[str]) -> BufferIO:
        """Reuses an existing buffer object, resizing if necessary."""
        try:
            current_capacity = buffer_io.buffer_obj.get_capacity()
            _LOGGER.info("Reusing pool buffer (capacity %d).", current_capacity)

            # Reset the buffer for reuse: start at the beginning and clear written length.
            buffer_io.seek(0)
            buffer_io._metadata.len_written_data = 0

            buffer_path = buffer_io.buffer_obj.get_id()
            self.active_buffers[buffer_path] = (buffer_io, symlink)

            if symlink:
                try:
                    # Create symlink pointing to the physical buffer path
                    os.symlink(buffer_path, symlink)
                    _LOGGER.debug("Created symlink '%s' -> '%s'", symlink, buffer_path)
                except OSError:
                    # If symlink creation fails, propagate the error.
                    # The caller (acquire) is responsible for cleanup.
                    raise

            return buffer_io
        except Exception:
            raise

    def _preallocate_buffers(self) -> None:
        """Pre-allocates fixed number of buffers."""
        for idx in range(self.num_buffers):
            buffer_name = f"buffer_{self.rank}_{idx}.dist"
            buffer_path = os.path.join(self.pool_dir, buffer_name)

            # Create/Reset buffer file
            try:
                _LOGGER.info("Pre-allocating buffer: %s with size %d", buffer_path, self.buffer_size)

                # Always overwrite to ensure a clean state
                buffer_obj = BufferObject(buffer_path, self.buffer_size, overwrite=True)

                buffer_io = BufferIO(buffer_obj)
                self.free_buffers.append(buffer_io)

            except Exception:
                _LOGGER.exception("Failed to pre-allocate buffer %s", buffer_path)
                raise
