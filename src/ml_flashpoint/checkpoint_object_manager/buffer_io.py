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

import io
from typing import Union

from ml_flashpoint.checkpoint_object_manager.buffer_object.buffer_object_ext import BufferObject
from ml_flashpoint.core.mlf_logging import get_logger

from .buffer_metadata import METADATA_SIZE, BufferMetadataType

_LOGGER = get_logger(__name__)


class BufferIO:
    """Provides I/O operations on a BufferObject."""

    def __init__(
        self,
        buffer_obj: BufferObject,
    ):
        """Initializes the BufferIO stream.

        Args:
            buffer_obj: The underlying BufferObject to wrap.
        """
        if buffer_obj.closed:
            _LOGGER.error("Cannot initialize BufferIO with a closed BufferObject.")
            raise ValueError("Cannot initialize BufferIO with a closed BufferObject.")

        self.buffer_obj = buffer_obj
        # Position relative to the START OF THE DATA SECTION (after metadata)
        self._pos: int = 0

        try:
            self._mv = memoryview(self.buffer_obj)
        except TypeError:
            _LOGGER.error("Underlying BufferObject does not support the buffer protocol, cannot create memoryview.")
            raise ValueError("Underlying BufferObject does not support the buffer protocol.")

        if len(self._mv) < METADATA_SIZE:
            _LOGGER.error("A buffer of at least %d bytes is required, but got %d.", METADATA_SIZE, len(self._mv))
            raise ValueError(f"Buffer size must be at least {METADATA_SIZE} bytes.")

        try:
            if self.buffer_obj.is_readonly:
                # For read-only buffers, we cannot create a writable live view.
                # Instead, we create a copy of the metadata. This is safe because
                # we only need to read it, not modify it.
                _LOGGER.info("Initializing in read-only mode: creating a copy of the metadata.")
                self._metadata = BufferMetadataType.from_buffer_copy(self._mv[:METADATA_SIZE])
            else:
                # For writable buffers, create a "live" view that directly maps to the memory.
                _LOGGER.info("Initializing in writable mode: creating a live view of the metadata section.")
                self._metadata = BufferMetadataType.from_buffer(self._mv[:METADATA_SIZE])
        except Exception as e:
            _LOGGER.exception("Failed to create metadata object from buffer slice.")
            raise IOError(f"Could not initialize metadata from buffer: {e}") from e

    def _check_validity(self, operation: str = None):
        """Checks if the stream is open and optionally if the operation is permitted.

        If an `operation` string is provided, this helper also uses an allow-list
        to enforce that the operation is allowed in the buffer's current mode.
        If no operation is provided, it only checks that the buffer is open.

        Args:
            operation: An optional string identifying the I/O operation being
                attempted (e.g., "read", "write").

        Raises:
            ValueError: If the buffer has already been closed.
            TypeError: If a specific operation is provided and it is not permitted
                in the buffer's current mode.
        """
        # This universal check for the closed state always comes first.
        if self.closed:
            _LOGGER.error("Attempted to perform an I/O operation on a closed BufferIO object.")
            raise ValueError("I/O operation on a closed BufferIO object")

        # If no specific operation was passed, the validation is complete.
        if operation is None:
            return

        # If an operation was specified, check it against the allowed list for the current mode.
        if self.is_readonly:
            allowed_ops = ("read", "readinto", "seek")
            if operation not in allowed_ops:
                _LOGGER.error("Attempted to perform '%s' on a BufferIO stream opened in READ_ONLY mode.", operation)
                raise TypeError(f"Operation '{operation}' is not supported in read-only mode.")
        else:  # This is a writable buffer.
            allowed_ops = ("write", "read", "readinto", "seek")
            if operation not in allowed_ops:
                _LOGGER.error("Attempted to perform '%s' on a BufferIO stream opened in READ_WRITE mode.", operation)
                raise TypeError(f"Operation '{operation}' is not supported in read-write mode.")

    def _update_written_data_length(self, new_length: int):
        """Updates the data length in metadata if new_length is greater.

        This method maintains a high-water mark for the amount of data written
        to the buffer, ensuring the metadata always reflects the total size
        of the content.

        Args:
            new_length: The new potential length of the written data.
        """
        if new_length > self._metadata.len_written_data:
            self._metadata.len_written_data = new_length

    def write(self, data: bytes) -> int:
        """Writes a bytes object to the current position in the data section.

        Args:
            data: The bytes object to be written.

        Returns:
            The number of bytes written.

        Raises:
            TypeError: If the buffer was opened in READ_ONLY mode.
            ValueError: If the buffer has already been closed.
            OSError: If the write operation would exceed the buffer's total
            capacity.
        """
        self._check_validity("write")

        data_len = len(data)
        if data_len == 0:
            return 0

        # Calculate the write position within the underlying memoryview.
        # The actual write position must be offset by the size of the metadata header.
        actual_write_start = METADATA_SIZE + self._pos
        actual_write_end = actual_write_start + data_len

        # Check for capacity. Ensure the write does not go out of bounds.
        if actual_write_end > len(self._mv):
            error_msg = (
                "Write operation exceeds buffer capacity. "
                f"Attempted to write {data_len} bytes at position {self._pos} "
                f"which would exceed the total capacity of {len(self._mv)}."
            )
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

        self._mv[actual_write_start:actual_write_end] = data

        self._pos += data_len

        # Update the metadata's 'high-water mark' for written data.
        self._update_written_data_length(self._pos)

        return data_len

    def read(self, size: int = -1) -> bytes:
        """Reads up to `size` bytes from the current position in the data section.

        If `size` is -1 or omitted, it reads all bytes from the current position
        until the end of the written data.

        Args:
            size: The maximum number of bytes to read.

        Returns:
            A bytes object containing the data read from the buffer. Returns an
            empty bytes object (b"") if the end of the file is reached.

        Raises:
            ValueError: If the buffer has already been closed.
        """
        self._check_validity("read")

        written_data_len = self._metadata.len_written_data

        if self._pos >= written_data_len:
            return b""  # Return empty bytes to signal EOF.

        # Calculate the number of bytes to read.
        if size < 0:
            # A negative size means "read everything until the end".
            bytes_to_read = written_data_len - self._pos
        else:
            # Otherwise, read the smaller of `size` or the remaining available bytes.
            bytes_to_read = min(size, written_data_len - self._pos)

        if bytes_to_read <= 0:
            return b""

        # Calculate the read position within the underlying memoryview.
        # The actual read position must be offset by the size of the metadata header.
        actual_read_start = METADATA_SIZE + self._pos
        actual_read_end = actual_read_start + bytes_to_read

        # Slice the memoryview and convert it to a new bytes object.
        data = self._mv[actual_read_start:actual_read_end].tobytes()

        self._pos += bytes_to_read

        return data

    def readinto(self, b: Union[bytearray, memoryview]) -> int:
        """Reads data directly into a pre-allocated, writable buffer.

        This is a performance-optimized method that avoids the overhead of creating
        a new bytes object, which `read()` does.

        Args:
            b: A writable, bytes-like object (e.g., a bytearray or a writable
              memoryview) where the data will be read into.

        Returns:
            The number of bytes actually read into the buffer `b`.

        Raises:
            ValueError: If the buffer has already been closed.
            TypeError: If the provided object `b` is not a writable buffer.
            OSError: If copying data into the provided buffer fails.
        """
        self._check_validity("readinto")

        # Verify that the provided buffer `b` is actually writable.
        if not isinstance(b, (bytearray, memoryview)) or (isinstance(b, memoryview) and b.readonly):
            _LOGGER.error("The buffer provided to readinto() must be a writable bytes-like object (e.g., a bytearray).")
            raise TypeError(f"readinto() argument must be a writable bytes-like object, not {type(b).__name__}")

        bytes_to_read = len(b)
        if bytes_to_read == 0:
            return 0

        written_data_len = self._metadata.len_written_data

        if self._pos >= written_data_len:
            return 0  # Return 0 bytes read to signal EOF.

        # Calculate the actual number of bytes to read.
        bytes_to_read = min(bytes_to_read, written_data_len - self._pos)
        if bytes_to_read <= 0:
            return 0

        # Calculate the read position within the underlying memoryview.
        actual_read_start = METADATA_SIZE + self._pos
        actual_read_end = actual_read_start + bytes_to_read

        try:
            # Data is copied directly from our internal memoryview
            # into the user-provided buffer's memory slice.
            b[:bytes_to_read] = self._mv[actual_read_start:actual_read_end]
        except Exception as e:
            # This could fail for various reasons (e.g., issues with the
            # destination buffer `b`). Log the details before raising an OSError.
            _LOGGER.exception("Failed to copy data into the destination buffer during readinto()")
            raise OSError(f"Failed to read data into buffer from slice [{actual_read_start}:{actual_read_end}]") from e

        self._pos += bytes_to_read

        return bytes_to_read

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Changes the stream position within the data section.

        Args:
            offset: The byte offset to move the position.
            whence: The reference point for the offset.

              - `0` or `io.SEEK_SET`: from the start of the data section.
              - `1` or `io.SEEK_CUR`: from the current position.
              - `2` or `io.SEEK_END`: from the end of the currently written data.

        Returns:
            The new absolute position within the data section.

        Raises:
            ValueError: If an invalid `whence` value is provided, if the resulting
                        position is negative, or if the stream is closed.
        """
        self._check_validity("seek")

        written_data_len = self._metadata.len_written_data
        new_pos: int

        # Calculate the new position based on the 'whence' mode.
        if whence == io.SEEK_SET:  # Seek from the start of the data area
            new_pos = offset
        elif whence == io.SEEK_CUR:  # Seek from the current position
            new_pos = self._pos + offset
        elif whence == io.SEEK_END:  # Seek from the end of the written data
            new_pos = written_data_len + offset
        else:
            error_msg = f"invalid whence value ({whence})"
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

        # Check if the resulting position is valid. Seeking before the start
        # of the data section is not allowed.
        if new_pos < 0:
            error_msg = f"Negative seek position is not allowed ({new_pos})"
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

        self._pos = new_pos

        return self._pos

    def tell(self) -> int:
        """Returns the current stream position within the data section.

        Returns:
            The current position as an integer offset from the start of the
            data section.
        """

        self._check_validity()
        return self._pos

    def next_buffer_slice(self, size: int) -> memoryview:
        """Returns a writable memoryview slice of the buffer at the current position.

        This allows for zero-copy operations into the buffer (e.g., direct tensor copy).
        The stream position is advanced by `size` bytes.

        Args:
            size: The size of the slice in bytes.

        Returns:
            A writable memoryview slice.
        """
        self._check_validity("write")
        if size < 0:
            raise ValueError(f"Size must be non-negative, got {size}")

        actual_start = METADATA_SIZE + self._pos
        actual_end = actual_start + size

        if actual_end > len(self._mv):
            raise ValueError(
                f"Requested slice (size={size}) exceeds buffer capacity "
                f"(pos={self._pos}, cap={len(self._mv) - METADATA_SIZE})"
            )

        # Create the slice
        slice_mv = self._mv[actual_start:actual_end]

        # Advance position
        self._pos += size
        self._update_written_data_length(self._pos)

        return slice_mv

    def close(self, truncate: bool = True) -> None:
        """Closes the BufferIO stream and the underlying C++ BufferObject.

        This method releases all associated resources, including the memoryview.
        After `close()` is called, any further I/O operations will fail.

        Args:
            truncate: If True (default), the underlying buffer is truncated to the
              size of the data actually written (`len_written_data`). If False, the
              buffer is closed without changing its size.
        """
        if self.closed:
            _LOGGER.debug("close() called on an already-closed BufferIO object.")
            return

        _LOGGER.info("Closing BufferIO object...")
        try:
            # Attempt to close the underlying buffer and handle potential errors,
            # ensuring resources are released in the `finally` block.
            if truncate and not self.is_readonly:
                final_data_len = self._metadata.len_written_data
                truncate_size = METADATA_SIZE + final_data_len
                _LOGGER.info("Closing and truncating buffer to %d bytes.", truncate_size)
                self.buffer_obj.close(truncate_size=truncate_size)
            else:
                _LOGGER.info("Closing buffer without truncation.")
                self.buffer_obj.close()
        except Exception:
            _LOGGER.exception("Error occurred while calling the underlying BufferObject's close() method")
            raise

        finally:
            # This block will ALWAYS execute, whether the try block succeeded or
            # failed, ensuring that Python-level resources are released.
            _LOGGER.info("Releasing BufferIO resources (memoryview, object references).")
            if self._mv:
                self._mv.release()
                self._mv = None

            # Set all references to None to mark the object as closed and allow
            # garbage collection.
            self.buffer_obj = None
            self._metadata = None

            # Set position to -1, a common convention for a closed stream.
            self._pos = -1

    # --- Properties and Context Manager ---

    @property
    def closed(self) -> bool:
        """Returns True if the stream is closed.

        A stream is considered closed if its underlying BufferObject has been
        set to None (during the close() operation) or if the BufferObject itself
        reports that it is closed.

        Returns:
            True if the stream is closed, False otherwise.
        """
        return self.buffer_obj is None or self.buffer_obj.closed

    @property
    def is_readonly(self) -> bool:
        """Returns True if the underlying buffer is read-only.

        Returns:
            True if the buffer is read-only, False otherwise.
        """
        self._check_validity()
        # Delegate the check to the underlying C++ object.
        return self.buffer_obj.is_readonly

    def __enter__(self):
        """Enters the runtime context for use in a `with` statement.

        Returns:
            The BufferIO object itself.

        Raises:
            ValueError: If the object is already closed when entering the context.
        """
        _LOGGER.debug("BufferIO context manager entered (`__enter__`).")
        # Ensure the object is valid before it's used inside the 'with' block.
        self._check_validity()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exits the runtime context, automatically closing the stream.

        This ensures that `self.close()` is called even if exceptions occur
        within the `with` block.
        """
        _LOGGER.debug("BufferIO context manager exited (`__exit__`), automatically calling close().")
        # The call to close() is unconditional to ensure resource release.
        self.close()

    def flush(self):
        """Provided for file-like interface compatibility. This is a no-op.

        For a memory-mapped buffer, flushing is typically a no-op because the
        operating system is responsible for managing the synchronization of
        memory pages to the underlying storage.

        Raises:
            ValueError: If the stream is already closed.
        """
        # Check validity first, as flush() is still an I/O operation.
        self._check_validity()
        pass
