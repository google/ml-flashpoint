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

import pathlib
import shutil
import tempfile

import pytest

# Import the compiled module. As you pointed out, the module name is buffer_object_ext.
# We'll alias it to buffer_object for easier use in the test code.
from ml_flashpoint.checkpoint_object_manager.buffer_object import buffer_object_ext


@pytest.fixture
def temp_dir_path():
    _temp_dir = tempfile.mkdtemp()
    yield pathlib.Path(_temp_dir)
    shutil.rmtree(_temp_dir)


def test_buffer_object_creation_and_properties(temp_dir_path):
    """
    Tests the creation of a BufferObject and validates its initial properties.
    A new BufferObject should be in an open, read-write state.
    """
    file_path = temp_dir_path / "test_buffer"
    # Create a new BufferObject with a capacity of 1024 bytes.
    bo = buffer_object_ext.BufferObject(str(file_path), capacity=1024, overwrite=False)

    assert bo.get_id() == str(file_path)
    assert bo.get_capacity() == 1024
    assert bo.closed is False
    assert bo.is_readonly is False

    # Get a memoryview and check its properties.
    mv = memoryview(bo)
    assert mv.nbytes == 1024
    assert not mv.readonly

    bo.close()
    assert bo.closed


def test_overwrite_flag(temp_dir_path):
    """
    Tests the behavior of the 'overwrite' flag during creation.
    1. Creating a file that exists with overwrite=False should fail.
    2. Creating it with overwrite=True should succeed.
    """
    file_path = temp_dir_path / "overwrite_test"

    # Create the file for the first time.
    bo1 = buffer_object_ext.BufferObject(str(file_path), capacity=32, overwrite=False)
    bo1.close()

    # Try to create it again (with overwrite=False), which should fail.
    with pytest.raises(RuntimeError, match="File already exists and overwrite is set to false"):
        buffer_object_ext.BufferObject(str(file_path), capacity=32, overwrite=False)

    # Using overwrite=True should succeed.
    bo2 = buffer_object_ext.BufferObject(str(file_path), capacity=64, overwrite=True)
    assert bo2.get_capacity() == 64
    bo2.close()
    assert bo2.closed


def test_buffer_object_read_write(temp_dir_path):
    """
    Tests the full cycle of writing data to a BufferObject, closing it,
    reopening it, and reading the data back to verify its integrity.
    """
    # Create the BufferObject.
    file_path = temp_dir_path / "test_buffer"
    bo = buffer_object_ext.BufferObject(str(file_path), capacity=128, overwrite=True)

    # Write data using a memoryview.
    test_data = b"Hello, BufferObject!"
    mv = memoryview(bo)
    mv[: len(test_data)] = test_data

    # Close the BufferObject.
    bo.close()

    # Reopen in read-only mode.
    bo_read = buffer_object_ext.BufferObject(str(file_path))
    mv_read = memoryview(bo_read)

    assert bo_read.get_capacity() == 128
    assert bo_read.closed is False
    assert bo_read.is_readonly is True
    assert mv_read.readonly
    assert mv_read[: len(test_data)].tobytes() == test_data

    bo_read.close()
    assert bo_read.closed


def test_buffer_object_read_only_mode(temp_dir_path):
    """Tests the read-only mode of the BufferObject."""
    file_path = temp_dir_path / "read_only_test"
    with open(file_path, "wb") as f:
        f.write(b"initial data")

    # Open in read-only mode.
    bo = buffer_object_ext.BufferObject(str(file_path))
    assert bo.is_readonly

    # Check if the memoryview is read-only.
    mv = memoryview(bo)
    assert mv.readonly

    # Attempting to write should raise a TypeError.
    with pytest.raises(TypeError):
        mv[0] = 65  # ASCII for 'A'

    bo.close()
    assert bo.closed


def test_buffer_object_closed_error(temp_dir_path):
    """Tests that operating on a closed BufferObject raises an error."""
    file_path = temp_dir_path / "closed_test"
    bo = buffer_object_ext.BufferObject(str(file_path), capacity=16, overwrite=True)
    bo.close()
    assert bo.closed

    # The C++ code throws a RuntimeError, but Python's memoryview() constructor
    # wraps it in a BufferError. We must catch the top-level BufferError.
    with pytest.raises(BufferError) as excinfo:
        memoryview(bo)

    # Now, we inspect the captured exception's `__cause__` to verify
    # that the underlying reason for the failure is the specific
    # RuntimeError we expect from our C++ binding.
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "Cannot create buffer view on a closed BufferObject" in str(excinfo.value.__cause__)


def test_open_directory_error(temp_dir_path):
    """Tests that attempting to open a directory raises an error."""

    with pytest.raises(RuntimeError, match="Path is a directory, not a file"):
        buffer_object_ext.BufferObject(str(temp_dir_path))


def test_multiple_close_calls_are_safe(temp_dir_path):
    """
    Tests that calling close() multiple times on a BufferObject is safe
    and does not raise an error. The object should simply remain in a closed state.
    This validates that the close() method can be called repeatedly without
    unexpected side effects.
    """
    file_path = temp_dir_path / "multiple_close_test"
    bo = buffer_object_ext.BufferObject(str(file_path), capacity=16, overwrite=True)
    assert not bo.closed

    # Call close() multiple times to ensure it's safe.
    bo.close()
    bo.close()
    bo.close()

    assert bo.closed  # The object should remain closed.

    # Attempting further operations should still fail as expected.
    with pytest.raises(BufferError):
        memoryview(bo)


def test_reopening_on_same_variable_is_safe(temp_dir_path):
    """
    Tests that repeatedly creating a BufferObject for the same path and
    assigning it to the same variable is safe.
    """
    file_path = temp_dir_path / "reopen_test.bin"
    file_path.write_bytes(b"initial data")

    # Assign a new BufferObject to the same variable multiple times.
    # This is analogous to "open, open, open".
    bo = buffer_object_ext.BufferObject(str(file_path))
    bo = buffer_object_ext.BufferObject(str(file_path))
    bo = buffer_object_ext.BufferObject(str(file_path))

    # Assert: The final instance of the object should be valid, open, and
    # contain the correct data.
    assert not bo.closed
    assert memoryview(bo) == b"initial data"


# TODO: Add tests for get_data_ptr
