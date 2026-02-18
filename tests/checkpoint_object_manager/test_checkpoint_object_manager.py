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
import pathlib
import shutil
import tempfile

import pytest

from ml_flashpoint.checkpoint_object_manager.buffer_io import METADATA_SIZE, BufferIO
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId, CheckpointObjectId


# --- Fixtures ---
@pytest.fixture
def temp_dir_path():
    """Creates a temporary directory for tests, cleaning it up afterward."""
    _temp_dir = tempfile.mkdtemp()
    yield pathlib.Path(_temp_dir)
    shutil.rmtree(_temp_dir)


@pytest.fixture(params=["real", "mock"])
def manager_setup(request, mocker, temp_dir_path):
    """
    Parametrized fixture to set up the manager in 'real' or 'mock' mode.

    Yields:
        tuple: (manager, is_mock, mocks_dict, temp_dir_path)
    """
    is_mock = request.param == "mock"
    mocks = {}

    if is_mock:
        # In 'mock' mode, patch the underlying classes that the manager depends on.
        mocks["BufferObject"] = mocker.patch(
            "ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager.BufferObject", autospec=True
        )
        mocks["BufferIO"] = mocker.patch(
            "ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager.BufferIO", autospec=True
        )
        mocks["BufferPool"] = mocker.patch(
            "ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager.BufferPool", autospec=True
        )
        # Default behavior: Pool not initialized, so we fall back to standalone
        mocks["BufferPool"].get_instance.side_effect = RuntimeError("BufferPool not initialized")

    manager = CheckpointObjectManager()
    yield manager, is_mock, mocks, temp_dir_path


@pytest.fixture
def mock_buffer_manager(mocker, temp_dir_path):
    """A dedicated fixture that ONLY provides a mocked environment for unit tests."""
    mocks = {
        "BufferObject": mocker.patch(
            "ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager.BufferObject", autospec=True
        ),
        "BufferIO": mocker.patch(
            "ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager.BufferIO", autospec=True
        ),
        "BufferPool": mocker.patch(
            "ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager.BufferPool", autospec=True
        ),
    }
    # Ensure BufferPool.get_instance() raises RuntimeError by default to simulate "not initialized"
    # or return a mock if needed. For tests expecting fallback, it should raise RuntimeError.
    mocks["BufferPool"].get_instance.side_effect = RuntimeError("BufferPool not initialized")

    manager = CheckpointObjectManager()
    yield manager, mocks, temp_dir_path


@pytest.fixture
def real_buffer_manager(temp_dir_path):
    """A dedicated fixture that ONLY provides a real environment for specific unit tests."""
    manager = CheckpointObjectManager()
    yield manager, temp_dir_path


# --- Test Cases ---
class TestAcquireBuffer:
    def test_acquire_buffer_success(self, manager_setup, mocker):
        """Tests that acquire_buffer successfully returns a BufferIO instance."""
        manager, is_mock, mocks, temp_dir_path = manager_setup
        object_id = CheckpointObjectId(str(temp_dir_path / "new_buffer.bin"))
        buffer_size = 1024

        if is_mock:
            mock_instance = mocker.MagicMock()
            mock_io_instance = mocker.MagicMock()
            mocks["BufferObject"].return_value = mock_instance
            mocks["BufferIO"].return_value = mock_io_instance
            buffer_io = manager.acquire_buffer(object_id, buffer_size=buffer_size, overwrite=False)
            mocks["BufferObject"].assert_called_once_with(object_id, buffer_size + METADATA_SIZE, False)
            mocks["BufferIO"].assert_called_once_with(mock_instance)
            assert buffer_io is mock_io_instance
        else:
            buffer_io = manager.acquire_buffer(object_id, buffer_size=buffer_size, overwrite=False)
            assert isinstance(buffer_io, BufferIO)
            assert os.path.exists(str(object_id))
            assert os.path.getsize(str(object_id)) == buffer_size + METADATA_SIZE
        buffer_io.close()

    def test_acquire_buffer_succeeds_with_overwrite_on_non_existent_buffer(self, manager_setup, mocker):
        """
        Tests that acquire_buffer with overwrite=True succeeds when the buffer does not already exist.
        """
        manager, is_mock, mocks, temp_dir_path = manager_setup
        object_id = CheckpointObjectId(str(temp_dir_path / "new_with_overwrite.bin"))
        buffer_size = 1024

        # Capture the return value of the method call.
        returned_buffer = manager.acquire_buffer(object_id, buffer_size=buffer_size, overwrite=True)

        if is_mock:
            # In mock mode, verify the full chain of calls and the return value.
            mock_instance = mocks["BufferObject"].return_value
            mocks["BufferObject"].assert_called_once_with(object_id, buffer_size + METADATA_SIZE, True)
            mocks["BufferIO"].assert_called_once_with(mock_instance)

            assert returned_buffer is mocks["BufferIO"].return_value
        else:
            # In real mode, verify the file system state, return type, and internal state.
            assert os.path.exists(str(object_id))
            assert os.path.getsize(str(object_id)) == buffer_size + METADATA_SIZE
            assert isinstance(returned_buffer, BufferIO)
        returned_buffer.close()

    def test_acquire_buffer_fails_with_non_positive_size(self, manager_setup):
        """
        Tests that acquire_buffer fails if buffer_size is zero or negative.
        """
        manager, _, _, _ = manager_setup

        with pytest.raises(ValueError, match="Buffer size must be a positive integer"):
            manager.acquire_buffer(CheckpointObjectId("/zero_size.bin"), buffer_size=0)

        with pytest.raises(ValueError, match="Buffer size must be a positive integer"):
            manager.acquire_buffer(CheckpointObjectId("/negative_size.bin"), buffer_size=-100)

    def test_acquire_buffer_propagates_exception_from_buffer_object(self, mock_buffer_manager):
        """
        Unit Test: Verifies that acquire_buffer correctly handles and re-raises an
        exception from the underlying BufferObject C++ extension during creation.
        This test ONLY runs in a mocked environment.
        """

        manager, mocks, temp_dir_path = mock_buffer_manager
        object_id = CheckpointObjectId(str(temp_dir_path / "cpp_fail.bin"))
        buffer_size = 1010

        # Configure the mocked BufferObject class to simulate a failure on instantiation.
        mocks["BufferObject"].side_effect = RuntimeError("C++ layer failed: disk is full")

        with pytest.raises(RuntimeError, match=f"Failed to create buffer for '{object_id}'"):
            manager.acquire_buffer(object_id, buffer_size=buffer_size)

        # Ensure the failing class was called, but the subsequent wrapper was not.
        mocks["BufferObject"].assert_called_once_with(object_id, buffer_size + METADATA_SIZE, True)
        mocks["BufferIO"].assert_not_called()

    def test_acquire_buffer_propagates_exception_from_buffer_io_creation(self, mock_buffer_manager, mocker):
        """
        Tests that acquire_buffer correctly handles an exception from the BufferIO wrapper.
        This test ONLY runs in a mocked environment.
        """
        manager, mocks, temp_dir_path = mock_buffer_manager
        object_id = CheckpointObjectId(str(temp_dir_path / "python_fail.bin"))
        buffer_size = 450

        mock_instance = mocker.MagicMock()
        mocks["BufferObject"].return_value = mock_instance

        # We configure the BufferIO wrapper class to fail upon instantiation.
        error_message = "Python wrapper failed: invalid metadata"
        mocks["BufferIO"].side_effect = Exception(error_message)

        # We expect the manager to catch the internal exception and re-raise it as a RuntimeError.
        with pytest.raises(RuntimeError, match=f"Failed to create buffer for '{object_id}'"):
            manager.acquire_buffer(object_id, buffer_size=buffer_size)

        # Verify that the C++ object was created (the first step).
        mocks["BufferObject"].assert_called_once_with(object_id, buffer_size + METADATA_SIZE, True)

        # Verify that the code ATTEMPTED to create the BufferIO wrapper (the second, failing step).
        mocks["BufferIO"].assert_called_once_with(mock_instance)

    def test_acquire_buffer_size_includes_metadata_overhead(self, real_buffer_manager):
        """
        Unit Test: Verifies that the size passed to the underlying BufferObject
        is the user-requested size PLUS the METADATA_SIZE overhead.
        """
        # Given
        manager, temp_dir_path = real_buffer_manager
        object_id = CheckpointObjectId(str(temp_dir_path / "metadata_size_test.bin"))
        user_requested_size = 1024
        expected_internal_size = user_requested_size + METADATA_SIZE

        # When
        with manager.acquire_buffer(object_id, buffer_size=user_requested_size) as buffer:
            # Then
            assert os.path.exists(str(object_id))
            assert os.path.getsize(str(object_id)) == expected_internal_size
            assert buffer.buffer_obj.get_capacity() == expected_internal_size

    def test_acquire_buffer_fails_on_directory_path(self, manager_setup):
        """
        Tests that acquire_buffer fails if the object_id points to an existing directory.
        """
        manager, is_mock, mocks, temp_dir_path = manager_setup

        # The object_id is the directory path itself.
        # In 'real' mode, this directory actually exists.
        object_id = CheckpointObjectId(str(temp_dir_path))
        buffer_size = 1024

        if is_mock:
            mocks["BufferObject"].side_effect = RuntimeError("OS error: path is a directory")

        # we expect the manager to catch the underlying error and re-raise it as a FileExistsError (since it exists).
        with pytest.raises(FileExistsError, match=f"File {object_id} already exists"):
            manager.acquire_buffer(object_id, buffer_size=buffer_size, overwrite=False)

        if is_mock:
            mocks["BufferObject"].assert_not_called()


class TestGetBuffer:
    def test_get_buffer_opens_existing_file_as_readonly(self, manager_setup, mocker):
        """
        Tests get_buffer correctly opens a new file in read-only mode.
        """

        manager, is_mock, mocks, temp_dir_path = manager_setup
        object_id = CheckpointObjectId(str(temp_dir_path / "get_new.bin"))

        if is_mock:
            mock_instance = mocker.MagicMock()
            mocks["BufferObject"].return_value = mock_instance
            mock_io_instance = mocker.MagicMock(spec=BufferIO)
            mock_io_instance.is_readonly = True
            mocks["BufferIO"].return_value = mock_io_instance
        else:
            # For the real test, we must create a file on disk for get_buffer to open.
            # It must be non-empty for BufferIO's validation to pass.
            with open(str(object_id), "wb") as f:
                f.write(b"\x00" * (METADATA_SIZE + 1))  # Write a minimal non-empty file.

        returned_buffer = manager.get_buffer(object_id)

        if is_mock:
            mocks["BufferObject"].assert_called_once_with(object_id)
            mocks["BufferIO"].assert_called_once_with(mock_instance)
            assert returned_buffer is mock_io_instance
            assert returned_buffer.is_readonly is True
        else:
            assert isinstance(returned_buffer, BufferIO)
            assert returned_buffer.is_readonly is True
        returned_buffer.close()

    def test_get_buffer_returns_none_for_non_existent_file(self, manager_setup):
        """
        Tests that get_buffer returns None if the target file does not exist.
        """
        manager, is_mock, mocks, temp_dir_path = manager_setup
        object_id = CheckpointObjectId(str(temp_dir_path / "non_existent.bin"))

        if is_mock:
            # Simulate the underlying BufferObject raising an OSError when the file is not found.
            mocks["BufferObject"].side_effect = OSError("File not found")

        returned_buffer = manager.get_buffer(object_id)

        assert returned_buffer is None

        if is_mock:
            mocks["BufferObject"].assert_called_once_with(object_id)
            mocks["BufferIO"].assert_not_called()

    def test_get_buffer_for_directory_returns_none(self, manager_setup):
        """
        Tests that get_buffer returns None gracefully if the object_id is a directory.
        """
        manager, is_mock, mocks, temp_dir_path = manager_setup

        # The object_id is the directory path itself.
        object_id = CheckpointObjectId(str(temp_dir_path))

        if is_mock:
            mocks["BufferObject"].side_effect = OSError(f"Is a directory: '{object_id}'")

        # In real mode, calling get_buffer with a directory path will cause the
        # underlying BufferObject to raise an OSError, which is caught and
        # handled by the manager, returning None.
        returned_buffer = manager.get_buffer(object_id)

        assert returned_buffer is None

        if is_mock:
            mocks["BufferObject"].assert_called_once_with(object_id)
            mocks["BufferIO"].assert_not_called()

    def test_get_buffer_returns_none_on_value_error(self, mock_buffer_manager):
        """
        Unit Test: Tests that get_buffer returns None if the underlying BufferObject
        raises a ValueError (e.g., due to corrupted metadata).
        """
        # Given
        manager, mocks, temp_dir_path = mock_buffer_manager
        object_id = CheckpointObjectId(str(temp_dir_path / "value_error.bin"))
        mocks["BufferObject"].side_effect = ValueError("Invalid metadata in buffer")

        # When
        returned_buffer = manager.get_buffer(object_id)

        # Then
        assert returned_buffer is None
        mocks["BufferObject"].assert_called_once_with(str(object_id))
        mocks["BufferIO"].assert_not_called()

    def test_get_buffer_returns_none_on_runtime_error(self, mock_buffer_manager):
        """
        Unit Test: Tests that get_buffer returns None if the underlying BufferObject
        raises a RuntimeError.
        """
        # Given
        manager, mocks, temp_dir_path = mock_buffer_manager
        object_id = CheckpointObjectId(str(temp_dir_path / "runtime_error.bin"))
        mocks["BufferObject"].side_effect = RuntimeError("A C++ level runtime error")

        # When
        returned_buffer = manager.get_buffer(object_id)

        # Then
        assert returned_buffer is None
        mocks["BufferObject"].assert_called_once_with(str(object_id))
        mocks["BufferIO"].assert_not_called()

    def test_get_buffer_returns_none_for_malformed_file(self, real_buffer_manager):
        """
        Tests that get_buffer returns None if the file is smaller than the metadata size.
        """
        manager, temp_dir_path = real_buffer_manager
        object_id = CheckpointObjectId(str(temp_dir_path / "malformed.bin"))

        # Create a file that is too small
        with open(str(object_id), "wb") as f:
            f.write(b"bad_data")

        assert manager.get_buffer(object_id) is None


class TestCloseBuffer:
    def test_close_buffer_success(self, real_buffer_manager):
        """
        Tests that close_buffer successfully closes a real buffer.
        """
        manager, temp_dir_path = real_buffer_manager
        object_id = CheckpointObjectId(str(temp_dir_path / "real_buffer.bin"))

        # Create a real buffer
        buffer_to_close = manager.acquire_buffer(object_id, METADATA_SIZE + 1)
        assert buffer_to_close.closed is False

        # Close it via the manager
        manager.close_buffer(buffer_to_close)

        # Verify it is now closed
        assert buffer_to_close.closed is True

    def test_close_buffer_on_already_closed_is_safe(self, real_buffer_manager):
        """
        Tests that calling close_buffer on an already-closed real buffer is safe.
        """
        manager, temp_dir_path = real_buffer_manager
        object_id = CheckpointObjectId(str(temp_dir_path / "real_buffer_to_pre_close.bin"))

        # Create and immediately close a real buffer
        buffer_to_close = manager.acquire_buffer(object_id, METADATA_SIZE + 1)
        manager.close_buffer(buffer_to_close)
        assert buffer_to_close.closed is True, "Buffer should be closed after the first close"

        # Calling close again should not raise any exceptions
        try:
            manager.close_buffer(buffer_to_close)
        except Exception as e:
            pytest.fail(f"Calling close on an already-closed buffer raised an unexpected exception: {e}")

    def test_close_buffer_propagates_exception(self, mocker):
        """
        Unit Test: Verifies that if the underlying buffer's close() method fails,
        the exception is propagated up through the manager.
        """
        # Given
        manager = CheckpointObjectManager()
        mock_buffer_io = mocker.MagicMock(spec=BufferIO)
        mock_buffer_io.closed = False
        mock_buffer_io.buffer_obj = mocker.MagicMock()
        mock_buffer_io.buffer_obj.get_id.return_value = "mock_buffer_id"
        mock_buffer_io.close.side_effect = OSError("Disk full")

        # When/Then
        with pytest.raises(OSError, match="Disk full"):
            manager.close_buffer(mock_buffer_io)

        # Verify that the close method was indeed called.
        mock_buffer_io.close.assert_called_once_with(truncate=True)


class TestDeleteContainer:
    def test_delete_container_success(self, manager_setup, mocker):
        """
        Tests the happy path: deleting a container with a non-empty file inside.
        """
        manager, is_mock, _, temp_dir_path = manager_setup

        container_path = temp_dir_path / "container_to_delete"
        container_path.mkdir()

        # Create a non-empty file inside the container.
        file_path = container_path / "file1.bin"
        file_content = "This is some test data."
        file_path.write_text(file_content)

        if is_mock:
            # In mock mode, we must mock the filesystem operations.
            mocker.patch("os.path.isdir", return_value=True)
            mock_rmtree = mocker.patch("shutil.rmtree")

        # In 'real' mode, we can add an assertion to verify the file was created correctly.
        if not is_mock:
            assert container_path.exists()
            assert file_path.read_text() == file_content

        # Call the method under test.
        manager.delete_container(CheckpointContainerId(str(container_path)))

        if is_mock:
            # In mock mode, verify that shutil.rmtree was called.
            mock_rmtree.assert_called_once_with(str(container_path))
        else:
            # In real mode, verify the directory and its content are gone.
            assert not container_path.exists()

    def test_delete_container_with_no_content(self, manager_setup, mocker):
        """
        Tests that deleting an empty container directory works correctly.
        """
        manager, is_mock, _, temp_dir_path = manager_setup
        container_path = CheckpointContainerId(str(temp_dir_path / "empty_container"))

        if is_mock:
            mocker.patch("os.path.isdir", return_value=True)
            mock_rmtree = mocker.patch("shutil.rmtree")
        else:
            os.mkdir(str(container_path))
            assert os.path.exists(str(container_path))

        manager.delete_container(container_path)

        if is_mock:
            mock_rmtree.assert_called_once_with(container_path)
        else:
            assert not os.path.exists(str(container_path))

    def test_delete_container_when_directory_does_not_exist(self, manager_setup, mocker):
        """
        Tests that it does not raise an error if the container directory does not exist on disk.
        """
        manager, is_mock, _, temp_dir_path = manager_setup
        container_path = CheckpointContainerId(str(temp_dir_path / "non_existent_dir"))

        if is_mock:
            mocker.patch("os.path.isdir", return_value=False)
            mock_rmtree = mocker.patch("shutil.rmtree")
        # For the real test, we simply don't create the directory.
        try:
            manager.delete_container(container_path)
        except Exception as e:
            pytest.fail(f"Deleting a non-existent container raised an unexpected exception: {e}")

        if is_mock:
            mock_rmtree.assert_not_called()

    def test_delete_container_propagates_os_error_on_rmtree_failure(self, mocker):
        """
        Unit Test: If shutil.rmtree fails, the OSError should be propagated.
        This test is now fully self-contained and does not rely on filesystem fixtures.
        """
        manager = CheckpointObjectManager()
        fake_container_path = CheckpointContainerId("/a/fake/path/that/does/not/exist")

        mocker.patch("os.path.isdir", return_value=True)
        mocker.patch("shutil.rmtree", side_effect=OSError("Permission denied"))

        with pytest.raises(OSError, match="Permission denied"):
            manager.delete_container(fake_container_path)

    def test_delete_container_on_file_path_does_nothing(self, real_buffer_manager):
        """
        Tests that calling delete_container on a file path does not delete the
        file and does not raise an error.
        """
        manager, temp_dir_path = real_buffer_manager

        # Create a file
        file_path = temp_dir_path / "a_file.txt"
        file_path.write_text("I am a file, not a container.")

        assert file_path.exists()

        # Call delete_container with the path to the file
        try:
            manager.delete_container(CheckpointContainerId(str(file_path)))
        except Exception as e:
            pytest.fail(f"delete_container raised an unexpected exception on a file path: {e}")

        # Assert that the file still exists
        assert file_path.exists()


class TestManagerIntegration:
    def test_manager_full_lifecycle(self, temp_dir_path):
        """
        Tests the complete lifecycle using the manager:
        create -> write -> close -> get -> read -> close.
        This is the primary "happy path" integration test.
        """
        manager = CheckpointObjectManager()
        object_id = CheckpointObjectId(str(temp_dir_path / "lifecycle_test.bin"))
        initial_size = METADATA_SIZE + 1024
        test_content = b"This is a full lifecycle test."

        # --- Phase 1: Create, Write, and Close ---
        # Create a new writable buffer via the manager.
        buffer_io = manager.acquire_buffer(object_id, buffer_size=initial_size)
        assert buffer_io is not None
        assert buffer_io.is_readonly is False

        # Write content to it.
        bytes_written = buffer_io.write(test_content)
        assert bytes_written == len(test_content)

        manager.close_buffer(buffer_io)
        assert buffer_io.closed

        # Verify the file was created on disk and truncated to the correct size.
        expected_file_size = METADATA_SIZE + len(test_content)
        assert os.path.exists(str(object_id))
        assert os.path.getsize(str(object_id)) == expected_file_size

        # --- Phase 2: Get, Read, and Close ---
        # Get the buffer again, which should open it in read-only mode.
        buffer_io_ro = manager.get_buffer(object_id)
        assert buffer_io_ro is not None
        assert buffer_io_ro.is_readonly is True

        # Read the content back and verify it's correct.
        read_content = buffer_io_ro.read()
        assert read_content == test_content

        manager.close_buffer(buffer_io_ro)
        assert buffer_io_ro.closed

    def test_manager_overwrite_succeeds_on_untracked_disk_file(self, temp_dir_path):
        """
        Tests that acquire_buffer with overwrite=True correctly replaces a file
        that exists on disk.
        """
        manager = CheckpointObjectManager()
        object_id = CheckpointObjectId(str(temp_dir_path / "overwrite_test.bin"))

        # Directly create a file on disk to represent a pre-existing state.
        with open(str(object_id), "wb") as f:
            f.write(b"This is the original, old data that will be overwritten.")

        new_size = METADATA_SIZE + 512
        new_content = b"This is the new content after overwriting."

        with manager.acquire_buffer(object_id, buffer_size=new_size, overwrite=True) as buffer_io:
            buffer_io.write(new_content)

        # Verify the file size on disk has been updated to reflect the new content.
        expected_new_size = METADATA_SIZE + len(new_content)
        assert os.path.getsize(str(object_id)) == expected_new_size

        # Verify the content has been correctly overwritten.
        with manager.get_buffer(object_id) as read_only_buffer:
            content = read_only_buffer.read()
            assert content == new_content

    def test_manager_delete_container(self, temp_dir_path):
        """
        Tests that delete_container correctly removes a directory
        and all its contents from the filesystem.
        """
        manager = CheckpointObjectManager()
        container_path = temp_dir_path / "container_to_delete"
        container_path.mkdir()

        # Use the manager to create some files inside the container.
        with manager.acquire_buffer(CheckpointObjectId(str(container_path / "file1.bin")), METADATA_SIZE + 128) as f:
            f.write(b"data1")

        with manager.acquire_buffer(CheckpointObjectId(str(container_path / "file2.bin")), METADATA_SIZE + 128) as f:
            f.write(b"data2")

        # Verify that the files exist before deletion.
        assert (container_path / "file1.bin").exists()
        assert (container_path / "file2.bin").exists()

        # Delete the entire container.
        manager.delete_container(CheckpointContainerId(str(container_path)))

        # The directory and all its contents should be gone.
        assert not container_path.exists()

    def test_acquire_buffer_fails_if_already_exists_on_disk(self, temp_dir_path):
        """
        Tests an edge case: acquire_buffer with overwrite=False should fail
        if a file already exists at the target path.
        """
        manager = CheckpointObjectManager()
        object_id = CheckpointObjectId(str(temp_dir_path / "disk_file.bin"))

        # Create a file on disk manually.
        with open(str(object_id), "wb") as f:
            f.write(b"pre-existing data")

        # Attempting to create should fail because of the explicit check in acquire_buffer.
        with pytest.raises(FileExistsError, match=f"File {object_id} already exists and overwrite=False"):
            manager.acquire_buffer(object_id, buffer_size=METADATA_SIZE + 1024, overwrite=False)

    def test_close_buffer_truncation_behavior(self, temp_dir_path):
        manager = CheckpointObjectManager()
        object_id_truncate = CheckpointObjectId(str(temp_dir_path / "truncate.bin"))
        object_id_no_truncate = CheckpointObjectId(str(temp_dir_path / "no_truncate.bin"))
        user_requested_size = 1024
        content = b"short content"

        # Case 1: Truncate (default)
        buf1 = manager.acquire_buffer(object_id_truncate, user_requested_size)
        buf1.write(content)
        manager.close_buffer(buf1, truncate=True)
        expected_size = METADATA_SIZE + len(content)
        assert os.path.getsize(str(object_id_truncate)) == expected_size

        # Case 2: No Truncate
        buf2 = manager.acquire_buffer(object_id_no_truncate, user_requested_size)
        buf2.write(content)
        manager.close_buffer(buf2, truncate=False)
        assert os.path.getsize(str(object_id_no_truncate)) == user_requested_size + METADATA_SIZE
