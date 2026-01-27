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
import os
import pathlib
import re
import shutil
import tempfile

import pytest

from ml_flashpoint.checkpoint_object_manager.buffer_io import METADATA_SIZE, BufferIO
from ml_flashpoint.checkpoint_object_manager.buffer_metadata import BufferMetadataType
from ml_flashpoint.checkpoint_object_manager.buffer_object.buffer_object_ext import BufferObject


# --- Mocks and Fixtures ---
class MockBufferObject(bytearray):
    """
    A mock C++ BufferObject.
    It inherits from bytearray to natively support the buffer protocol,
    while retaining custom methods to simulate the behavior of the C++ object.
    """

    def __init__(self, object_id, capacity, overwrite=False, is_readonly=False):
        # Initialize the parent bytearray class, allocating memory for the specified capacity.
        super().__init__(capacity)
        self.id = object_id
        self._is_readonly = is_readonly
        self._closed = False
        self._capacity_prop = capacity

    def close(self, truncate_size=None):
        if self._closed:
            return
        if truncate_size is not None and not self._is_readonly:
            self._capacity_prop = truncate_size
        self._closed = True

    @property
    def closed(self):
        return self._closed

    @property
    def is_readonly(self):
        return self._is_readonly

    def get_capacity(self):
        return self._capacity_prop

    def get_id(self):
        return self.id


@pytest.fixture(params=["real", "mock"])
def buffer_impl(request):
    """Parametrized fixture to provide either the real or mock object implementation."""
    return request.param


@pytest.fixture
def writable_buffer_obj(temp_dir_path, buffer_impl):
    """Provides a writable buffer object, either real or mocked."""
    if buffer_impl == "real":
        file_path = str(temp_dir_path / "writable_test.bin")
        buffer_obj = BufferObject(file_path, METADATA_SIZE + 1024, overwrite=True)
    else:  # mock
        buffer_obj = MockBufferObject("writable_mock.bin", METADATA_SIZE + 1024, is_readonly=False)
    yield buffer_obj
    if not buffer_obj.closed:
        buffer_obj.close()


@pytest.fixture
def readonly_buffer_obj(temp_dir_path, buffer_impl):
    """Provides a read-only buffer object, either real or mocked."""
    file_path = str(temp_dir_path / "readonly_test.bin")
    initial_content = b"some pre-existing data"

    if buffer_impl == "real":
        with BufferIO(BufferObject(file_path, METADATA_SIZE + 512, overwrite=True)) as bio_write:
            bio_write.write(initial_content)
        buffer_obj = BufferObject(file_path)
    else:  # mock
        buffer_obj = MockBufferObject("readonly_mock.bin", METADATA_SIZE + 512, is_readonly=True)
        metadata = BufferMetadataType.from_buffer(buffer_obj)
        metadata.len_written_data = len(initial_content)
        data_start = METADATA_SIZE
        data_end = METADATA_SIZE + len(initial_content)
        buffer_obj[data_start:data_end] = initial_content
    yield buffer_obj
    if not buffer_obj.closed:
        buffer_obj.close()


@pytest.fixture
def small_buffer_obj(temp_dir_path, buffer_impl):
    """Provides a buffer object that is smaller than METADATA_SIZE."""
    if buffer_impl == "real":
        file_path = str(temp_dir_path / "small_test.bin")
        buffer_obj = BufferObject(file_path, METADATA_SIZE - 1, overwrite=True)
        yield buffer_obj
        if not buffer_obj.closed:
            buffer_obj.close()
    else:  # mock
        yield MockBufferObject("small.bin", METADATA_SIZE - 1, is_readonly=False)


@pytest.fixture
def mock_writable_buffer_obj():
    """Provides a writable mock BufferObject instance."""
    return MockBufferObject("writable_mock.bin", METADATA_SIZE + 1024)


@pytest.fixture
def temp_dir_path():
    _temp_dir = tempfile.mkdtemp()
    yield pathlib.Path(_temp_dir)
    shutil.rmtree(_temp_dir)


# --- Test Cases ---
class TestInitialization:
    def test_buffer_io_successful_initialization(self, writable_buffer_obj):
        """Test: successful initialization with a valid BufferObject."""
        bio = BufferIO(writable_buffer_obj)
        assert bio.buffer_obj is writable_buffer_obj
        assert bio.tell() == 0
        assert bio.closed is False
        assert bio.is_readonly is False

    def test_buffer_io_readonly_initialization(self, readonly_buffer_obj):
        """Test: successful initialization of a read-only buffer."""
        bio = BufferIO(readonly_buffer_obj)
        assert bio.buffer_obj is readonly_buffer_obj
        assert bio.tell() == 0
        assert bio.closed is False
        assert bio.is_readonly is True

    def test_buffer_io_init_with_closed_buffer_raises_error(self, writable_buffer_obj):
        """Test: initializing with a closed BufferObject should raise ValueError."""
        writable_buffer_obj.close()
        with pytest.raises(ValueError, match="Cannot initialize BufferIO with a closed BufferObject"):
            BufferIO(writable_buffer_obj)

    def test_buffer_io_init_with_insufficient_size_raises_error(self, small_buffer_obj):
        """Test: a buffer size smaller than METADATA_SIZE should raise ValueError."""
        with pytest.raises(ValueError, match=f"Buffer size must be at least {METADATA_SIZE} bytes"):
            BufferIO(small_buffer_obj)


class TestWriteOperations:
    def test_write_successful_and_updates_state(self, writable_buffer_obj):
        """
        Test: A successful write returns the correct number of bytes and updates
        the internal position and metadata.
        """
        bio = BufferIO(writable_buffer_obj)
        test_data = b"This is a test."
        bytes_written = bio.write(test_data)

        assert bytes_written == len(test_data)
        assert bio.tell() == len(test_data)
        # Verify that the underlying metadata was updated correctly.
        # We access the raw buffer to check the "high-water mark".
        meta = BufferMetadataType.from_buffer(memoryview(writable_buffer_obj))
        assert meta.len_written_data == len(test_data)

        # Verify the actual data was written to the correct location (after metadata).
        end_of_data = METADATA_SIZE + len(test_data)
        written_data_in_buffer = memoryview(writable_buffer_obj)[METADATA_SIZE:end_of_data]
        assert written_data_in_buffer.tobytes() == test_data

    def test_write_to_readonly_buffer_raises_type_error(self, readonly_buffer_obj):
        """
        Test: Attempting to write to a buffer opened in read-only mode must
        raise a TypeError.
        """
        bio = BufferIO(readonly_buffer_obj)

        # This write operation should fail because the buffer is not writable.
        with pytest.raises(TypeError, match="Operation 'write' is not supported in read-only mode."):
            bio.write(b"This should fail.")

    def test_write_exceeding_capacity_raises_value_error(self, writable_buffer_obj):
        """
        Test: Attempting to write data that would exceed the total capacity of
        the buffer must raise a ValueError.
        """
        # The writable buffer has a data capacity of 1024 bytes.
        # Move the position near the end of the buffer.
        bio = BufferIO(writable_buffer_obj)
        bio.seek(1000)

        # This write is too large (29 bytes requested, but only 24 are available).
        with pytest.raises(ValueError, match="Write operation exceeds buffer capacity"):
            bio.write(b"This data is too long to fit.")

    def test_write_zero_bytes_is_a_noop(self, writable_buffer_obj):
        """
        Test: Writing an empty bytes object should do nothing and return 0.
        """
        bio = BufferIO(writable_buffer_obj)
        initial_pos = bio.tell()
        bytes_written = bio.write(b"")
        assert bytes_written == 0
        assert bio.tell() == initial_pos

    def test_write_multiple_chunks_updates_position_correctly(self, writable_buffer_obj):
        """
        Test: A sequence of write calls should correctly advance the internal
        position each time.
        """
        bio = BufferIO(writable_buffer_obj)

        # First chunk
        bytes_written1 = bio.write(b"chunk1,")
        assert bytes_written1 == 7
        assert bio.tell() == 7

        # Second chunk
        bytes_written2 = bio.write(b"chunk2")
        assert bytes_written2 == 6
        assert bio.tell() == 13  # 7 + 6

        # Verify the final written length in metadata
        meta = BufferMetadataType.from_buffer(memoryview(writable_buffer_obj))
        assert meta.len_written_data == 13

    def test_write_to_closed_buffer_raises_value_error(self, writable_buffer_obj):
        """
        Test: Attempting to write to a buffer that has already been closed
        must raise a ValueError.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.close()

        with pytest.raises(ValueError, match="I/O operation on a closed BufferIO object"):
            bio.write(b"data")

    # --- len_written_data high-water mark tests ---
    def test_write_after_seeking_back_does_not_shrink_metadata_length(self, writable_buffer_obj):
        """
        Test: Verifies the 'high-water mark' logic for `len_written_data`.
        """
        bio = BufferIO(writable_buffer_obj)

        # Write initial data to establish a high-water mark.
        bio.write(b"0123456789")  # len_written_data is now 10.
        assert bio._metadata.len_written_data == 10

        # Seek back into the middle of the written data.
        bio.seek(4)

        # Overwrite with shorter data.
        bio.write(b"ABCD")  # This overwrites bytes at positions 4, 5, 6, 7.

        # The position is now 8, but the high-water mark should remain 10.
        assert bio.tell() == 8
        assert bio._metadata.len_written_data == 10, (
            "len_written_data should not decrease after overwriting a smaller portion."
        )


class TestGetBufferSlice:
    def test_get_buffer_slice_returns_correct_view(self, writable_buffer_obj):
        """Test: get_buffer_slice returns a valid memoryview into the buffer."""
        bio = BufferIO(writable_buffer_obj)
        # Write some initial data
        bio.write(b"prefix")
        assert bio.tell() == 6

        # Get a slice of 10 bytes
        mv = bio.next_buffer_slice(10)
        assert isinstance(mv, memoryview)
        assert len(mv) == 10
        assert not mv.readonly

        # Write to the slice
        mv[:] = b"0123456789"

        # Verify position advanced
        assert bio.tell() == 16

        # Verify data in buffer
        bio.seek(6)
        data = bio.read(10)
        assert data == b"0123456789"

    def test_get_buffer_slice_validates_negative_size(self, writable_buffer_obj):
        """Test: get_buffer_slice raises ValueError for negative size."""
        bio = BufferIO(writable_buffer_obj)

        with pytest.raises(ValueError, match="Size must be non-negative"):
            bio.next_buffer_slice(-1)

    def test_get_buffer_slice_validates_capacity_exceeded(self, writable_buffer_obj):
        """Test: get_buffer_slice raises ValueError if size exceeds capacity."""
        bio = BufferIO(writable_buffer_obj)
        capacity = len(memoryview(writable_buffer_obj)) - METADATA_SIZE
        with pytest.raises(ValueError, match="exceeds buffer capacity"):
            bio.next_buffer_slice(capacity + 1)

    def test_get_buffer_slice_updates_metadata(self, writable_buffer_obj):
        """Test: getting a slice updates the len_written_data metadata."""
        bio = BufferIO(writable_buffer_obj)
        bio.next_buffer_slice(5)

        # Check metadata
        meta = BufferMetadataType.from_buffer(memoryview(writable_buffer_obj))
        assert meta.len_written_data == 5

    def test_get_buffer_slice_on_readonly_raises_error(self, readonly_buffer_obj):
        """Test: calling get_buffer_slice on readonly buffer raises error."""
        bio = BufferIO(readonly_buffer_obj)
        with pytest.raises(TypeError, match="Operation 'write' is not supported in read-only mode"):
            bio.next_buffer_slice(10)

    def test_get_buffer_slice_on_closed_raises_error(self, writable_buffer_obj):
        """Test: calling get_buffer_slice on closed buffer raises error."""
        bio = BufferIO(writable_buffer_obj)
        bio.close()
        with pytest.raises(ValueError, match="I/O operation on a closed BufferIO object"):
            bio.next_buffer_slice(10)


class TestReadOperations:
    def test_read_successful_and_updates_position(self, writable_buffer_obj):
        """
        Test: A successful read with a specified size returns the correct data
        and advances the internal position.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.write(b"abcdefghij")
        bio.seek(0)

        # Read the first chunk.
        first_chunk = bio.read(4)
        assert first_chunk == b"abcd"
        assert bio.tell() == 4

        # Read the next chunk from the new position.
        second_chunk = bio.read(3)
        assert second_chunk == b"efg"
        assert bio.tell() == 7

    def test_read_all_remaining_data_with_no_size(self, writable_buffer_obj):
        """
        Test: Calling read() with no arguments (or size=-1) reads all remaining
        data from the current position to the end.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.write(b"read all of this data")
        # Move position to the start of the word "all".
        bio.seek(5)
        # read() should get everything from position 5 onwards.
        remaining_data = bio.read()
        assert remaining_data == b"all of this data"
        # The final position should be at the end of the written data.
        assert bio.tell() == len(b"read all of this data")

    def test_read_requests_more_than_available_returns_whats_left(self, writable_buffer_obj):
        """
        Test: If the requested size is larger than the remaining data, read()
        should return only the available data without error.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.write(b"only 13 bytes")
        bio.seek(5)

        data = bio.read(100)
        assert data == b"13 bytes"
        assert bio.tell() == 13

    def test_read_at_end_of_file_returns_empty_bytes(self, writable_buffer_obj):
        """
        Test: Calling read() when the position is already at the end of the
        written data (EOF) must return an empty bytes object.
        """
        bio = BufferIO(writable_buffer_obj)
        content = b"some content"
        bio.write(content)

        # The position is now at the end of the file.
        assert bio.tell() == len(content)
        # Any subsequent read should return b"".
        assert bio.read(10) == b""
        assert bio.read() == b""
        assert bio.tell() == len(content)

    def test_read_zero_bytes_is_a_noop(self, writable_buffer_obj):
        """
        Test: Calling read(0) should do nothing, return an empty bytes object,
        and not change the position.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.write(b"some data")
        bio.seek(4)
        initial_pos = bio.tell()

        result = bio.read(0)
        assert result == b""
        assert bio.tell() == initial_pos

    def test_read_from_closed_buffer_raises_value_error(self, writable_buffer_obj):
        """
        Test: Attempting to read from a buffer that has already been closed
        must raise a ValueError.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.write(b"some data")
        bio.close()

        with pytest.raises(ValueError, match="I/O operation on a closed BufferIO object"):
            bio.read()


class TestReadintoOperations:
    def test_readinto_successful_fills_buffer_and_updates_state(self, writable_buffer_obj):
        """
        Test: A successful readinto call correctly fills the provided buffer,
        returns the number of bytes read, and advances the internal position.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.write(b"This is the source data for the test.")
        bio.seek(5)
        # Prepare the destination buffer.
        destination_buffer = bytearray(11)

        bytes_read = bio.readinto(destination_buffer)
        assert bytes_read == 11, "Should return the number of bytes read into the buffer."
        assert destination_buffer == b"is the sour"
        assert bio.tell() == 16  # 5 (initial) + 11 (read)

    def test_readinto_with_destination_larger_than_source(self, writable_buffer_obj):
        """
        Test: If the destination buffer is larger than the remaining data,
        readinto should only read what's available and return that count.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.write(b"short data")
        bio.seek(0)
        destination_buffer = bytearray(100)

        bytes_read = bio.readinto(destination_buffer)
        assert bytes_read == 10

        # Only the beginning of the destination buffer should be modified.
        assert destination_buffer.startswith(b"short data")
        # The rest of the buffer should remain untouched (zero-filled).
        assert destination_buffer[10:] == b"\x00" * 90

    def test_readinto_at_end_of_file_returns_zero(self, writable_buffer_obj):
        """
        Test: Calling readinto when the position is at the end of the written
        data (EOF) must return 0 and not modify the destination buffer.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.write(b"some data")  # Position is now at the end.
        destination_buffer = bytearray(b"original content")

        bytes_read = bio.readinto(destination_buffer)
        assert bytes_read == 0
        # The content of the destination buffer must not be changed.
        assert destination_buffer == b"original content"
        assert bio.tell() == len(b"some data")

    def test_readinto_with_zero_sized_buffer_is_a_noop(self, writable_buffer_obj):
        """
        Test: Calling readinto with a zero-sized destination buffer should
        do nothing, return 0, and not change the position.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.write(b"some data")
        bio.seek(4)  # Move to an arbitrary position.
        initial_pos = bio.tell()
        destination_buffer = bytearray(0)

        bytes_read = bio.readinto(destination_buffer)
        assert bytes_read == 0
        assert bio.tell() == initial_pos, "Position should not change."

    def test_readinto_with_non_writable_buffer_raises_type_error(self, writable_buffer_obj):
        """
        Test: Calling readinto with a non-writable buffer (like bytes or a
        read-only memoryview) must raise a TypeError.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.write(b"some data")
        bio.seek(0)
        # It will match the core error message, ignoring the type name at the end.
        match_pattern = r"readinto\(\) argument must be a writable bytes-like object"

        # Test with a read-only memoryview.
        readonly_destination = memoryview(bytearray(10)).toreadonly()
        with pytest.raises(TypeError, match=match_pattern):
            bio.readinto(readonly_destination)

        # Test with a bytes object (which is inherently not writable).
        bytes_destination = b" " * 10
        with pytest.raises(TypeError, match=match_pattern):
            bio.readinto(bytes_destination)

    def test_readinto_on_closed_buffer_raises_value_error(self, writable_buffer_obj):
        """
        Test: Attempting to call readinto on a buffer that has already been closed
        must raise a ValueError.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.close()
        destination_buffer = bytearray(10)
        with pytest.raises(ValueError, match="I/O operation on a closed BufferIO object"):
            bio.readinto(destination_buffer)


class TestSeekOperations:
    def test_seek_with_whence_set_updates_position(self, writable_buffer_obj):
        """
        Test: Seeking with whence=SEEK_SET (0) correctly sets the position
        relative to the start of the data section.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.write(b"some initial data")
        new_pos = bio.seek(5, io.SEEK_SET)
        assert new_pos == 5
        assert bio.tell() == 5

    def test_seek_with_whence_cur_updates_position(self, writable_buffer_obj):
        """
        Test: Seeking with whence=SEEK_CUR (1) correctly sets the position
        relative to the current position.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.write(b"some data for testing seek cur")
        bio.seek(10)
        # Seek forward.
        pos_after_forward = bio.seek(5, io.SEEK_CUR)
        assert pos_after_forward == 15
        assert bio.tell() == 15

        # Seek backward.
        pos_after_backward = bio.seek(-8, io.SEEK_CUR)
        assert pos_after_backward == 7
        assert bio.tell() == 7

    def test_seek_with_whence_end_updates_position(self, writable_buffer_obj):
        """
        Test: Seeking with whence=SEEK_END (2) correctly sets the position
        relative to the end of the *written data*.
        """
        bio = BufferIO(writable_buffer_obj)
        content = b"data with a known length of 30"  # Length is 30
        bio.write(content)
        assert bio._metadata.len_written_data == 30
        new_pos = bio.seek(-10, io.SEEK_END)
        assert new_pos == 20
        assert bio.tell() == 20

    def test_write_after_seeking_past_end_extends_file(self, writable_buffer_obj):
        """
        Test: Writing data after seeking past the current end of written data
        should correctly update the high-water mark.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.write(b"0123456789")
        assert bio.tell() == 10
        assert bio._metadata.len_written_data == 10
        bio.seek(10, io.SEEK_END)
        assert bio.tell() == 20
        bio.write(b"XYZ")
        assert bio.tell() == 23
        assert bio._metadata.len_written_data == 23
        bio.seek(0)
        final_data = bio.read()
        assert final_data == b"0123456789" + (b"\x00" * 10) + b"XYZ"

    def test_seek_to_negative_position_raises_value_error(self, writable_buffer_obj):
        """
        Test: Any seek operation that results in a negative final position
        must raise a ValueError.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.write(b"data")
        with pytest.raises(ValueError, match="Negative seek position is not allowed"):
            bio.seek(-1, io.SEEK_SET)
        bio.seek(5)
        with pytest.raises(ValueError, match="Negative seek position is not allowed"):
            bio.seek(-10, io.SEEK_CUR)

    def test_seek_with_invalid_whence_raises_value_error(self, writable_buffer_obj):
        """
        Test: Calling seek with an invalid 'whence' value (not 0, 1, or 2)
        must raise a ValueError.
        """
        bio = BufferIO(writable_buffer_obj)
        with pytest.raises(ValueError, match="invalid whence value"):
            bio.seek(0, 3)  # 3 is not a valid whence value.
        with pytest.raises(ValueError, match="invalid whence value"):
            bio.seek(0, -1)

    def test_seek_on_closed_buffer_raises_value_error(self, writable_buffer_obj):
        """
        Test: Attempting to call seek on a buffer that has already been closed
        must raise a ValueError.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.close()
        with pytest.raises(ValueError, match="I/O operation on a closed BufferIO object"):
            bio.seek(0)


class TestCloseOperations:
    def test_close_on_writable_buffer_with_truncation(self, writable_buffer_obj, mocker):
        """
        Test: close(truncate=True) correctly truncates the buffer.
        - For Mocks: Verifies behavior (parameters passed to close).
        - For Real Objects: Verifies state (final file size on disk).
        """
        bio = BufferIO(writable_buffer_obj)
        content = b"12345"
        bio.write(content)

        # The way we test depends on the type of the underlying object.
        if isinstance(writable_buffer_obj, MockBufferObject):
            # BEHAVIOR TEST for the mock object.
            mock_close = mocker.spy(bio.buffer_obj, "close")
            bio.close(truncate=True)
            expected_truncate_size = METADATA_SIZE + len(content)
            mock_close.assert_called_once_with(truncate_size=expected_truncate_size)
        else:
            # STATE TEST for the real C++ object.
            file_path = writable_buffer_obj.get_id()
            bio.close(truncate=True)
            expected_file_size = METADATA_SIZE + len(content)
            assert os.path.getsize(file_path) == expected_file_size

        assert bio.closed

    def test_close_on_writable_buffer_without_truncation(self, writable_buffer_obj, mocker):
        """
        Test: close(truncate=False) leaves the buffer at its original capacity.
        """
        initial_capacity = writable_buffer_obj.get_capacity()
        bio = BufferIO(writable_buffer_obj)
        bio.write(b"some data")

        if isinstance(writable_buffer_obj, MockBufferObject):
            # BEHAVIOR TEST for the mock object.
            mock_close = mocker.spy(bio.buffer_obj, "close")
            bio.close(truncate=False)
            mock_close.assert_called_once_with()  # Assert it was called with no args
        else:
            # STATE TEST for the real C++ object.
            file_path = writable_buffer_obj.get_id()
            bio.close(truncate=False)
            assert os.path.getsize(file_path) == initial_capacity

        assert bio.closed

    def test_close_on_readonly_buffer_ignores_truncation(self, readonly_buffer_obj, mocker):
        """
        Test: close(truncate=True) on a read-only buffer is ignored.
        """
        initial_capacity = readonly_buffer_obj.get_capacity()
        bio = BufferIO(readonly_buffer_obj)

        if isinstance(readonly_buffer_obj, MockBufferObject):
            # BEHAVIOR TEST for the mock object.
            mock_close = mocker.spy(bio.buffer_obj, "close")
            bio.close(truncate=True)
            mock_close.assert_called_once_with()  # Should be called with no args
        else:
            # STATE TEST for the real C++ object.
            file_path = readonly_buffer_obj.get_id()
            bio.close(truncate=True)
            # File size should NOT have changed.
            assert os.path.getsize(file_path) == initial_capacity

        assert bio.closed

    def test_multiple_close_calls_are_safe(self, writable_buffer_obj):
        """
        Test: Calling close() multiple times should be safe (a no-op).
        """
        bio = BufferIO(writable_buffer_obj)
        # Close it for the first time.
        bio.close()
        assert bio.closed
        # Try closing it again.
        try:
            bio.close()
            bio.close()
            bio.close()  # Multiple calls should be no-ops.
        except Exception as e:
            pytest.fail(f"Calling close() on a closed buffer raised an unexpected exception: {e}")

    # The following exception tests can ONLY run on mock objects, as we cannot
    # reliably make the real C++ object fail in a specific way.
    def test_close_propagates_underlying_exception_on_mock(self, mock_writable_buffer_obj, mocker):
        """
        Test: If the underlying mock object's close() fails, the exception propagates.
        """
        bio = BufferIO(mock_writable_buffer_obj)
        error_message = "Underlying close failed!"
        mocker.patch.object(bio.buffer_obj, "close", side_effect=RuntimeError(error_message))
        with pytest.raises(RuntimeError, match=re.escape(error_message)):
            bio.close()

    def test_close_releases_resources_even_on_failure_on_mock(self, mock_writable_buffer_obj, mocker):
        """
        Test: Python-level resources are cleaned up even if the underlying mock close() fails.
        """
        bio = BufferIO(mock_writable_buffer_obj)
        mv = bio._mv
        mocker.patch.object(bio.buffer_obj, "close", side_effect=RuntimeError("Failure during close"))
        with pytest.raises(RuntimeError):
            bio.close()
        assert bio.buffer_obj is None
        assert bio._mv is None
        with pytest.raises(ValueError, match="operation forbidden on released memoryview object"):
            _ = mv[0]


class TestFlushOperations:
    def test_flush_on_open_buffer_is_a_noop(self, writable_buffer_obj):
        """
        Test: Calling flush() on an open buffer should do nothing and not raise an error.
        """
        bio = BufferIO(writable_buffer_obj)
        try:
            bio.flush()
        except Exception as e:
            pytest.fail(f"flush() on an open buffer raised an unexpected exception: {e}")

    def test_flush_on_closed_buffer_raises_value_error(self, writable_buffer_obj):
        """
        Test: Calling flush() on a closed buffer must raise a ValueError.
        """
        bio = BufferIO(writable_buffer_obj)
        bio.close()
        with pytest.raises(ValueError, match="I/O operation on a closed BufferIO object"):
            bio.flush()


class TestGeneralInterface:
    # --- Tell Tests ---
    def test_tell_reflects_current_position(self, writable_buffer_obj):
        """
        Test: The tell() method should always return the current internal position.
        """
        bio = BufferIO(writable_buffer_obj)
        assert bio.tell() == 0
        bio.write(b"12345")
        assert bio.tell() == 5
        bio.seek(2)
        assert bio.tell() == 2
        bio.read(2)
        assert bio.tell() == 4

    # --- Context Manager Tests ---
    def test_context_manager_closes_on_success(self, writable_buffer_obj):
        """Test: The 'with' statement correctly closes the buffer."""
        with BufferIO(writable_buffer_obj) as bio:
            bio.write(b"data")
        assert bio.closed

    def test_context_manager_closes_on_exception(self, writable_buffer_obj):
        """Test: The 'with' statement closes the buffer even if an error occurs."""
        try:
            with BufferIO(writable_buffer_obj) as bio:
                raise ValueError("Something went wrong")
        except ValueError:
            pass  # Catch the expected exception
        assert bio.closed


class TestEndToEndIntegration:
    def test_e2e_write_read_truncate_lifecycle(self, temp_dir_path):
        """
        Tests the primary lifecycle: create, write, close (with truncate), re-open, and read.
        """
        file_path = str(temp_dir_path / "lifecycle_test.bin")
        initial_size = METADATA_SIZE + 1024
        test_content = b"Hello, this is a real end-to-end test!"

        # --- Phase 1: Create, Write, and Close (with truncate) ---
        with BufferIO(BufferObject(file_path, initial_size, overwrite=True)) as bio:
            bytes_written = bio.write(test_content)
            assert bytes_written == len(test_content)
            assert bio.tell() == len(test_content)

        # After exiting the 'with' block, close(truncate=True) is called.
        # Verify the file was created and truncated to the correct size.
        expected_file_size = METADATA_SIZE + len(test_content)
        assert os.path.exists(file_path)
        assert os.path.getsize(file_path) == expected_file_size

        # --- Phase 2: Re-open and Read ---
        with BufferIO(BufferObject(file_path)) as bio_ro:
            assert bio_ro.is_readonly is True

            # Read all content and verify.
            read_content = bio_ro.read()
            assert read_content == test_content

            # Verify EOF behavior.
            assert bio_ro.read(10) == b""

    def test_e2e_close_without_truncation(self, temp_dir_path):
        """
        Tests that closing a buffer with truncate=False leaves the file at its
        original, larger size.
        """
        file_path = str(temp_dir_path / "no_truncate_test.bin")
        initial_size = METADATA_SIZE + 8192
        test_content = b"small write"

        # Create, write, and close with truncate=False.
        bio = BufferIO(BufferObject(file_path, initial_size, overwrite=True))
        bio.write(test_content)
        bio.close(truncate=False)

        # The file size on disk should remain the original, large size.
        assert os.path.exists(file_path)
        assert os.path.getsize(file_path) == initial_size

        # Re-open and verify that only the small amount of data is considered "written".
        with BufferIO(BufferObject(file_path)) as bio_ro:
            read_data = bio_ro.read()
            assert read_data == test_content
            assert bio_ro.tell() == len(test_content)

    def test_e2e_readinto_and_seeking(self, temp_dir_path):
        """
        Tests the readinto() and seek() methods against a real file buffer.
        """
        file_path = str(temp_dir_path / "readinto_test.bin")
        test_content = b"0123456789abcdefghijklmnopqrstuvwxyz"

        # Create a file with known content ---
        with BufferIO(BufferObject(file_path, METADATA_SIZE + 100, overwrite=True)) as bio:
            bio.write(test_content)

        # Open read-only and use readinto/seek ---
        with BufferIO(BufferObject(file_path)) as bio_ro:
            # Seek to a specific position.
            bio_ro.seek(10)
            assert bio_ro.tell() == 10

            # Use readinto to read a chunk from that position.
            destination_buffer = bytearray(16)
            bytes_read = bio_ro.readinto(destination_buffer)

            assert bytes_read == 16
            assert destination_buffer == b"abcdefghijklmnop"
            assert bio_ro.tell() == 26  # 10 (initial) + 16 (read)

            # Seek from the end of the file.
            bio_ro.seek(-10, io.SEEK_END)
            assert bio_ro.tell() == len(test_content) - 10

            final_chunk = bio_ro.read()
            assert final_chunk == b"qrstuvwxyz"

    def test_e2e_overwriting_and_high_water_mark(self, temp_dir_path):
        """
        Tests that the 'high-water mark' for written data is correctly maintained
        in a real file scenario.
        """
        file_path = str(temp_dir_path / "high_water_mark.bin")

        with BufferIO(BufferObject(file_path, METADATA_SIZE + 100, overwrite=True)) as bio:
            # Write initial data to establish a high-water mark of 20.
            bio.write(b"01234567890123456789")
            assert bio._metadata.len_written_data == 20

            # Seek back into the middle.
            bio.seek(5)

            # Overwrite with shorter data. The new position will be 5 + 4 = 9.
            bio.write(b"ABCD")
            assert bio.tell() == 9

            # The high-water mark should still be 20.
            assert bio._metadata.len_written_data == 20, "High-water mark should not shrink after an overwrite."

        # Re-open and verify the final content on disk.
        with BufferIO(BufferObject(file_path)) as bio_ro:
            content = bio_ro.read()
            # The content should be a mix of old and new data.
            assert content == b"01234ABCD90123456789"

    def test_e2e_write_fails_on_readonly_buffer(self, temp_dir_path):
        """
        Tests that any write operation on a real, read-only buffer is correctly
        prevented by the BufferIO's internal checks.
        """
        file_path = str(temp_dir_path / "readonly_write_fail.bin")

        # Create a file with some initial content.
        with BufferIO(BufferObject(file_path, METADATA_SIZE + 128, overwrite=True)) as bio:
            bio.write(b"initial content")

        # Re-open the file (which is now read-only) and expect
        # a TypeError when trying to write.
        with BufferIO(BufferObject(file_path)) as bio_ro:
            assert bio_ro.is_readonly is True

            with pytest.raises(TypeError, match="Operation 'write' is not supported in read-only mode."):
                bio_ro.write(b"this should fail")

    def test_e2e_create_fails_if_file_exists_and_overwrite_is_false(self, temp_dir_path):
        """
        Tests that the C++ BufferObject constructor correctly fails if a file
        already exists and overwrite is False.
        """
        file_path = str(temp_dir_path / "overwrite_fail.bin")

        # Create a file on disk.
        with open(file_path, "w") as f:
            f.write("dummy file")

        # Expect a RuntimeError when trying to create a BufferObject
        # at the same path without the overwrite flag.
        with pytest.raises(RuntimeError, match="File already exists"):
            BufferObject(file_path, METADATA_SIZE + 1024, overwrite=False)

    def test_e2e_write_exceeds_capacity_raises_value_error(self, temp_dir_path):
        """
        Tests that attempting to write beyond the buffer's allocated capacity
        raises a ValueError, as expected by the Python layer checks.
        """
        file_path = str(temp_dir_path / "capacity_test.bin")
        # Create a small buffer with a total data capacity of 100 bytes.
        buffer_size = METADATA_SIZE + 100

        with BufferIO(BufferObject(file_path, buffer_size, overwrite=True)) as bio:
            # Write 90 bytes, leaving 10 bytes of space.
            bio.write(b"a" * 90)
            assert bio.tell() == 90

            # This write of 11 bytes should fail because it exceeds the capacity.
            with pytest.raises(ValueError, match="Write operation exceeds buffer capacity"):
                bio.write(b"b" * 11)

    def test_e2e_open_fails_on_empty_file(self, temp_dir_path):
        """
        Tests that the C++ layer correctly prevents opening a zero-byte file,
        and that this failure is propagated as a RuntimeError.
        """
        file_path = str(temp_dir_path / "empty_file.bin")

        # Create an empty file on disk.
        open(file_path, "w").close()
        assert os.path.getsize(file_path) == 0

        # Expect a RuntimeError when BufferObject tries to open it.
        with pytest.raises(RuntimeError, match="File cannot be empty"):
            BufferObject(file_path)

    def test_e2e_open_fails_on_directory(self, temp_dir_path):
        """
        Tests that the C++ layer correctly prevents opening a directory.
        """
        # temp_dir_path itself is a directory.
        directory_path = str(temp_dir_path)

        # Expect a RuntimeError indicating the path is a directory.
        with pytest.raises(RuntimeError, match="Path is a directory"):
            BufferObject(directory_path)
