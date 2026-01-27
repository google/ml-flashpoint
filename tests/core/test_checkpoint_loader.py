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

import dataclasses
import io
import logging
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Tuple
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture
from torch.distributed.checkpoint import Metadata
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    ReadItem,
)

from ml_flashpoint.checkpoint_object_manager.buffer_metadata import TensorHeader
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId, CheckpointObjectId
from ml_flashpoint.core.checkpoint_loader import DefaultMLFlashpointCheckpointLoader, MLFlashpointCheckpointLoader
from ml_flashpoint.replication.replication_manager import ReplicationManager

_LOGGER = logging.getLogger(__name__)


def _tensor_1d_bytes() -> io.BytesIO:
    """Creates a fixture for a serialized 1D PyTorch tensor.

    Returns:
        The byte representation of a sample 1D tensor.
    """
    tensor = torch.arange(20, dtype=torch.float32)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    return buffer


def _tensor_2d_bytes() -> io.BytesIO:
    """Creates a fixture for a serialized 2D PyTorch tensor.

    Returns:
        The byte representation of a sample 2D tensor.
    """
    tensor = torch.arange(20, dtype=torch.int64).reshape(4, 5)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    return buffer


def _tensor_3d_bytes() -> io.BytesIO:
    """Creates a fixture for a serialized 3D PyTorch tensor.
    Returns:
        The byte representation of a sample 3D tensor.
    """
    tensor = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    return buffer


def _tensor_with_extra_data_bytes() -> io.BytesIO:
    """Creates a fixture for a serialized 1D PyTorch tensor with extra non-tensor data."""
    tensor = torch.arange(20, dtype=torch.float32)
    extra_data = {"some_key": "some_value", "number": 123}
    buffer = io.BytesIO()
    torch.save((tensor, extra_data), buffer)  # Save as a tuple
    buffer.seek(0)
    return buffer


@pytest.fixture
def checkpoint_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@dataclasses.dataclass(frozen=True)
class TestStorageMetadata:
    """Metadata describing a chunk of data in storage for testing."""

    offset: int
    length: int


@pytest.fixture
def checkpoint_file() -> Tuple[io.BytesIO, Dict[int, TestStorageMetadata], int]:
    """Creates a fake checkpoint file in memory.

    The file contains two items:
    - Index 0: A serialized PyTorch tensor.
    - Index 1: A serialized dictionary (raw bytes).

    Returns:
        A tuple containing the BytesIO stream and the storage metadata.
    """
    # Item 0: A 2D tensor
    tensor_data = torch.arange(20, dtype=torch.int32).reshape(4, 5)
    tensor_buffer = io.BytesIO()
    torch.save(tensor_data, tensor_buffer)
    tensor_bytes = tensor_buffer.getvalue()

    # Item 1: Some arbitrary pickled bytes
    byte_data = {"key": "value", "count": 100}
    byte_bytes = pickle.dumps(byte_data)

    # Combine into a single file stream
    stream = io.BytesIO()
    metadata = {}

    # Write tensor at index 0
    metadata[0] = TestStorageMetadata(offset=stream.tell(), length=len(tensor_bytes))
    stream.write(tensor_bytes)

    # Write bytes at index 1
    metadata[1] = TestStorageMetadata(offset=stream.tell(), length=len(byte_bytes))
    stream.write(byte_bytes)

    total_stream_size = stream.tell()
    stream.seek(0)
    return stream, metadata, total_stream_size


class TestReadMetadata:
    @pytest.fixture
    def chkpt_object_manager(self):
        return CheckpointObjectManager()

    @pytest.fixture(autouse=True)
    def _setup(self, chkpt_object_manager, mocker) -> None:
        mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
        self.loader = DefaultMLFlashpointCheckpointLoader(
            checkpoint_object_manager=chkpt_object_manager,
            replication_manager=mock_replication_manager,
        )

    def test_read_metadata_success(self, checkpoint_directory):
        metadata_object_name = ".metadata"
        metadata = Metadata(state_dict_metadata={})
        metadata_path = Path(checkpoint_directory) / metadata_object_name
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        loaded_metadata = self.loader.read_metadata(
            CheckpointContainerId(checkpoint_directory),
            object_name=metadata_object_name,
        )
        assert loaded_metadata == metadata

    def test_read_metadata_diff_object_name_success(self, checkpoint_directory):
        metadata_object_name = "metadata.pt"
        metadata = Metadata(state_dict_metadata={})
        metadata_path = Path(checkpoint_directory) / metadata_object_name
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        loaded_metadata = self.loader.read_metadata(
            CheckpointContainerId(checkpoint_directory),
            object_name=metadata_object_name,
        )
        assert loaded_metadata == metadata

    def test_read_metadata_file_not_found(self, checkpoint_directory):
        with pytest.raises(FileNotFoundError):
            self.loader.read_metadata(
                CheckpointContainerId(checkpoint_directory),
                object_name="non_existent.pt",
            )

    def test_read_metadata_invalid_format(self, checkpoint_directory):
        invalid_metadata_path = Path(checkpoint_directory) / "invalid_metadata.pt"
        with open(invalid_metadata_path, "w") as f:
            f.write("this is not a valid metadata file")

        with pytest.raises(Exception):
            self.loader.read_metadata(
                CheckpointContainerId(checkpoint_directory),
                object_name="invalid_metadata.pt",
            )


class TestReadTensor:
    @pytest.fixture
    def chkpt_object_manager(self):
        return CheckpointObjectManager()

    @pytest.fixture(autouse=True)
    def _setup(self, chkpt_object_manager, mocker) -> None:
        mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
        self.loader = DefaultMLFlashpointCheckpointLoader(
            checkpoint_object_manager=chkpt_object_manager,
            replication_manager=mock_replication_manager,
        )

    def test_read_tensor_1d_slice(self):
        """Tests reading a slice from a 1D tensor."""
        # Arrange
        buffer_slice = _tensor_1d_bytes()
        # Request a slice from index 5 of length 10
        req = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=0,
            storage_offsets=(5,),
            lengths=(10,),
            dest_index=(0,),
            dest_offsets=(0,),
        )
        expected_tensor = torch.arange(5, 15, dtype=torch.float32)

        # Act
        result_tensor = self.loader.read_tensor(buffer_slice, req)

        # Assert
        _LOGGER.info("result_tensor: %s", result_tensor)
        _LOGGER.info("expected_tensor: %s", expected_tensor)
        assert result_tensor.dtype == expected_tensor.dtype
        assert torch.equal(result_tensor, expected_tensor)

    def test_read_tensor_2d_slice(self):
        """Tests reading a sub-matrix from a 2D tensor."""
        # Arrange
        buffer_slice = _tensor_2d_bytes()
        # Original tensor:
        # [[ 0,  1,  2,  3,  4],
        #  [ 5,  6,  7,  8,  9],
        #  [10, 11, 12, 13, 14],
        #  [15, 16, 17, 18, 19]]
        # Request a 2x3 slice starting at offset (row=1, col=2)
        req = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=0,
            storage_offsets=(1, 2),
            lengths=(2, 3),
            dest_index=(0,),
            dest_offsets=(0,),
        )
        expected_tensor = torch.tensor([[7, 8, 9], [12, 13, 14]], dtype=torch.int64)

        # Act
        result_tensor = self.loader.read_tensor(buffer_slice, req)
        _LOGGER.info("result_tensor: %s", result_tensor)
        _LOGGER.info("expected_tensor: %s", expected_tensor)

        # Assert
        assert result_tensor.dtype == expected_tensor.dtype
        assert torch.equal(result_tensor, expected_tensor)

    def test_read_tensor_3d_slice(self):
        """Tests reading a sub-volume from a 3D tensor."""
        # Arrange
        buffer_slice = _tensor_3d_bytes()
        # Original tensor is 3x4x5
        # Request a 2x2x2 slice starting at offset (depth=1, row=0, col=3)
        req = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=0,
            storage_offsets=(1, 0, 3),
            lengths=(2, 2, 2),
            dest_index=(0,),
            dest_offsets=(0,),
        )
        expected_tensor = torch.tensor(
            [[[23.0, 24.0], [28.0, 29.0]], [[43.0, 44.0], [48.0, 49.0]]],
            dtype=torch.float32,
        )

        # Act
        result_tensor = self.loader.read_tensor(buffer_slice, req)
        _LOGGER.info("result_tensor: %s", result_tensor)
        _LOGGER.info("expected_tensor: %s", expected_tensor)

        # Assert
        assert result_tensor.dtype == expected_tensor.dtype
        assert torch.equal(result_tensor, expected_tensor)

    def test_read_tensor_with_corrupted_data_raises_error(self):
        """Tests that a UnpicklingError is raised for invalid data."""
        # Arrange
        corrupted_bytes = b"this is not a valid tensor"
        buffer_slice = io.BytesIO(corrupted_bytes)
        req = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=0,
            storage_offsets=(0,),
            lengths=(1,),
            dest_index=(0,),
            dest_offsets=(0,),
        )

        # Act & Assert
        with pytest.raises(IndexError):
            self.loader.read_tensor(buffer_slice, req)

    def test_read_tensor_optimized_format(self):
        """test_read_tensor detects magic bytes and uses zero-copy path."""
        # Arrange
        tensor = torch.tensor([1, 2, 3], dtype=torch.int32)

        # Manually create optimized format buffer
        tensor_header = TensorHeader(
            shape=tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        buffer_data = tensor.numpy().tobytes()
        buffer_slice = io.BytesIO(buffer_data)

        req = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=0,
            storage_offsets=(0,),
            lengths=(len(buffer_data),),
            dest_index=(0,),
            dest_offsets=(0,),
        )

        # Act
        # Pass header explicitly simulating manifest lookup
        result_tensor = self.loader.read_tensor(buffer_slice, req, header=tensor_header)

        # Assert
        assert torch.equal(result_tensor, tensor)

    def test_read_tensor_fallback_legacy(self):
        """test_read_tensor falls back to torch.load if magic bytes missing."""
        # Arrange
        tensor = torch.tensor([4, 5, 6], dtype=torch.float32)
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)

        req = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=0,
            storage_offsets=(0,),
            lengths=(buffer.getbuffer().nbytes,),
            dest_index=(0,),
            dest_offsets=(0,),
        )

        # Act
        result_tensor = self.loader.read_tensor(buffer, req)

        # Assert
        assert torch.equal(result_tensor, tensor)

    @pytest.mark.skip(reason="Test skipped until it is setup properly.")
    def test_read_tensor_with_weights_only_behavior(self):
        """Tests that read_tensor correctly loads only tensor data when weights_only=True is used internally."""
        # Arrange
        buffer_slice = _tensor_with_extra_data_bytes()
        # Request a slice from index 5 of length 10
        req = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=0,
            storage_offsets=(5,),
            lengths=(10,),
            dest_index=(0,),
            dest_offsets=(0,),
        )
        expected_tensor_slice = torch.arange(5, 15, dtype=torch.float32)

        # Act
        result_tensor = self.loader.read_tensor(buffer_slice, req)

        # Assert
        assert isinstance(result_tensor, torch.Tensor)
        assert torch.equal(result_tensor, expected_tensor_slice)
        # The key assertion here is that no error was raised due to the extra_data,
        # implying weights_only=True successfully filtered it out.


class TestReadData:
    @pytest.fixture
    def chkpt_object_manager(self):
        return CheckpointObjectManager()

    @pytest.fixture(autouse=True)
    def _setup(self, chkpt_object_manager, mocker) -> None:
        mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
        self.loader = DefaultMLFlashpointCheckpointLoader(
            checkpoint_object_manager=chkpt_object_manager,
            replication_manager=mock_replication_manager,
        )

    def test_read_data_file_not_found(self, mocker: MockerFixture) -> None:
        """Tests that an error is logged and FileNotFoundError is raised for a non-existent checkpoint."""
        # Arrange
        mocker.patch("os.path.exists", return_value=False)
        checkpoint_obj_id = CheckpointObjectId("/non_existent/path")
        expected_error_msg = f"Checkpoint object '{checkpoint_obj_id.data}' does not exist"

        # Act & Assert
        with pytest.raises(
            FileNotFoundError,
            match=expected_error_msg,
        ):
            self.loader.read_data(checkpoint_obj_id, [], mocker.MagicMock(), {})

    def test_read_data_retrieval_success(self, mocker: MockerFixture, checkpoint_directory):
        """Tests successful retrieval of missing file."""
        # Arrange
        mock_repl_manager = mocker.MagicMock()
        self.loader._replication_manager = mock_repl_manager

        checkpoint_obj_id = CheckpointObjectId(str(Path(checkpoint_directory) / "checkpoint.data"))
        container_id = CheckpointContainerId(checkpoint_directory)

        # Populate cache
        self.loader._available_objects_cache[container_id] = {
            checkpoint_obj_id.data: [1]  # Available on rank 1
        }

        # Mock os.path.exists: False (initial), False (double check)
        mocker.patch("os.path.exists", side_effect=[False, False])

        mock_repl_manager.sync_bulk_retrieve.return_value = True

        # Mock fcntl
        mocker.patch("fcntl.flock")

        # Mock get_buffer
        mock_buffer_ctx = mocker.MagicMock()
        mocker.patch.object(self.loader._checkpoint_object_manager, "get_buffer", return_value=mock_buffer_ctx)
        mock_buffer_ctx.__enter__.return_value = io.BytesIO(b"")

        # Act
        self.loader.read_data(checkpoint_obj_id, [], mocker.MagicMock(), {})

        # Assert
        mock_repl_manager.sync_bulk_retrieve.assert_called_once()
        call_kwargs = mock_repl_manager.sync_bulk_retrieve.call_args[1]
        assert call_kwargs["source_global_rank"] == 1
        assert call_kwargs["object_ids_to_retrieve"] == [checkpoint_obj_id]

    def test_read_data_retrieval_failure(self, mocker: MockerFixture, checkpoint_directory):
        """Tests failure to retrieve missing file raises FileNotFoundError."""
        # Arrange
        mock_repl_manager = mocker.MagicMock()
        self.loader._replication_manager = mock_repl_manager

        checkpoint_obj_id = CheckpointObjectId(str(Path(checkpoint_directory) / "checkpoint.data"))
        container_id = CheckpointContainerId(checkpoint_directory)

        self.loader._available_objects_cache[container_id] = {checkpoint_obj_id.data: [1]}

        mocker.patch("os.path.exists", side_effect=[False, False])
        mock_repl_manager.sync_bulk_retrieve.return_value = False  # Retrieval failed
        mocker.patch("fcntl.flock")

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="does not exist"):
            self.loader.read_data(checkpoint_obj_id, [], mocker.MagicMock(), {})

    def test_read_data_retrieval_exception(self, mocker: MockerFixture, checkpoint_directory):
        """Tests that exceptions during retrieval are caught and FileNotFoundError is raised."""
        # Arrange
        mock_repl_manager = mocker.MagicMock()
        self.loader._replication_manager = mock_repl_manager

        checkpoint_obj_id = CheckpointObjectId(str(Path(checkpoint_directory) / "checkpoint.data"))
        container_id = CheckpointContainerId(checkpoint_directory)
        self.loader._available_objects_cache[container_id] = {checkpoint_obj_id.data: [1]}

        mocker.patch("os.path.exists", return_value=False)
        mocker.patch("fcntl.flock", side_effect=Exception("Lock failed"))

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="does not exist"):
            self.loader.read_data(checkpoint_obj_id, [], mocker.MagicMock(), {})

    def test_read_data_retrieval_concurrent_success(self, mocker: MockerFixture, checkpoint_directory):
        """Tests that if file appears after lock, retrieval is skipped."""
        # Arrange
        mock_repl_manager = mocker.MagicMock()
        self.loader._replication_manager = mock_repl_manager

        checkpoint_obj_id = CheckpointObjectId(str(Path(checkpoint_directory) / "checkpoint.data"))
        container_id = CheckpointContainerId(checkpoint_directory)

        self.loader._available_objects_cache[container_id] = {checkpoint_obj_id.data: [1]}

        # Mock os.path.exists: False (initial), True (double check after lock)
        mocker.patch("os.path.exists", side_effect=[False, True])

        mocker.patch("fcntl.flock")

        mock_buffer_ctx = mocker.MagicMock()
        mocker.patch.object(self.loader._checkpoint_object_manager, "get_buffer", return_value=mock_buffer_ctx)
        mock_buffer_ctx.__enter__.return_value = io.BytesIO(b"")

        # Act
        self.loader.read_data(checkpoint_obj_id, [], mocker.MagicMock(), {})

        # Assert
        mock_repl_manager.sync_bulk_retrieve.assert_not_called()

    def test_read_data_for_tensor_slice(
        self,
        mocker: MockerFixture,
        checkpoint_file: Tuple[io.BytesIO, Dict[int, TestStorageMetadata], int],
        checkpoint_directory: str,
        chkpt_object_manager: CheckpointObjectManager,
    ) -> None:
        stream, storage_data, stream_size = checkpoint_file
        checkpoint_obj_id = CheckpointObjectId(str(Path(checkpoint_directory) / "checkpoint.data"))
        with chkpt_object_manager.create_buffer(checkpoint_obj_id, buffer_size=stream_size) as f:
            f.write(stream.read())
        stream.seek(0)

        destination_tensor = torch.empty(2, 3, dtype=torch.int32)
        req = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=0,
            storage_offsets=(1, 1),
            lengths=(2, 3),
            dest_index=(0,),
            dest_offsets=(0,),
        )

        # 1. Create a mock planner.
        mock_planner = mocker.MagicMock()

        # 2. Configure the mock: when `resolve_tensor` is called with our
        # specific request, it should return our destination tensor.
        mock_planner.resolve_tensor.return_value = destination_tensor

        # Act
        self.loader.read_data(checkpoint_obj_id, [req], mock_planner, storage_data)

        # Assert
        mock_planner.resolve_tensor.assert_called_once_with(req)

        mock_planner.commit_tensor.assert_called_once()

        actual_args, _ = mock_planner.commit_tensor.call_args

        assert actual_args[0] == req
        actual_tensor = actual_args[1]
        expected_slice = torch.tensor([[6, 7, 8], [11, 12, 13]], dtype=torch.int32)
        assert torch.equal(actual_tensor, expected_slice)

        assert torch.equal(destination_tensor, expected_slice)

    def test_read_data_for_bytes(
        self,
        mocker: MockerFixture,
        checkpoint_file: Tuple[io.BytesIO, Dict[int, TestStorageMetadata], int],
        checkpoint_directory: str,
        chkpt_object_manager: CheckpointObjectManager,
    ) -> None:
        """Tests that read_data correctly handles a BYTE_IO request."""
        # Arrange
        stream, storage_data, stream_size = checkpoint_file
        checkpoint_obj_id = CheckpointObjectId(str(Path(checkpoint_directory) / "checkpoint.data"))
        with chkpt_object_manager.create_buffer(checkpoint_obj_id, buffer_size=stream_size) as f:
            f.write(stream.read())
        stream.seek(0)

        req = ReadItem(
            type=LoadItemType.BYTE_IO,
            storage_index=1,
            storage_offsets=(0, 0),
            lengths=(0, 0),
            dest_index=(0,),
            dest_offsets=(0,),
        )
        mock_planner = mocker.MagicMock()

        # Act
        self.loader.read_data(checkpoint_obj_id, [req], mock_planner, storage_data)

        # Assert
        # 1. Verify that only the `load_bytes` method was called.
        mock_planner.load_bytes.assert_called_once()
        mock_planner.resolve_tensor.assert_not_called()
        mock_planner.commit_tensor.assert_not_called()

        # 2. Extract and verify the arguments passed to `load_bytes`.
        actual_args, _ = mock_planner.load_bytes.call_args
        assert actual_args[0] == req

        # 3. Verify the content of the passed BytesIO stream.
        actual_bytes_stream = actual_args[1]
        result_data = pickle.loads(actual_bytes_stream.getvalue())
        assert result_data == {"key": "value", "count": 100}

    def test_read_data_for_mixed_requests(
        self,
        mocker: MockerFixture,
        checkpoint_file: Tuple[io.BytesIO, Dict[int, TestStorageMetadata], int],
        checkpoint_directory: str,
        chkpt_object_manager: CheckpointObjectManager,
    ) -> None:
        """Tests processing a list with both tensor and byte requests."""
        # Arrange
        stream, storage_data, stream_size = checkpoint_file
        checkpoint_obj_id = CheckpointObjectId(str(Path(checkpoint_directory) / "checkpoint.data"))
        with chkpt_object_manager.create_buffer(checkpoint_obj_id, buffer_size=stream_size) as f:
            f.write(stream.read())

        # Setup for the tensor request
        req_tensor = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=0,
            storage_offsets=(0, 0),
            lengths=(4, 5),
            dest_index=(0,),
            dest_offsets=(0,),
        )
        destination_tensor = torch.empty(4, 5, dtype=torch.int32)

        # Setup for the bytes request
        req_bytes = ReadItem(
            type=LoadItemType.BYTE_IO,
            storage_index=1,
            storage_offsets=(0, 0),
            lengths=(0, 0),
            dest_index=(0,),
            dest_offsets=(0,),
        )

        # Configure the mock planner
        mock_planner = mocker.MagicMock()
        mock_planner.resolve_tensor.return_value = destination_tensor

        # Act
        self.loader.read_data(
            checkpoint_obj_id,
            [req_tensor, req_bytes],
            mock_planner,
            storage_data,
        )

        # Assert
        # Assert Tensor Path
        mock_planner.resolve_tensor.assert_called_once_with(req_tensor)
        mock_planner.commit_tensor.assert_called_once()
        commit_args, _ = mock_planner.commit_tensor.call_args
        assert commit_args[0] == req_tensor
        expected_tensor = torch.arange(20, dtype=torch.int32).reshape(4, 5)
        assert torch.equal(commit_args[1], expected_tensor)

        # Assert Bytes Path
        mock_planner.load_bytes.assert_called_once()
        load_bytes_args, _ = mock_planner.load_bytes.call_args
        assert load_bytes_args[0] == req_bytes
        result_data = pickle.loads(load_bytes_args[1].getvalue())
        assert result_data == {"key": "value", "count": 100}

    def test_read_data_raises_on_mismatched_tensor_size(
        self,
        mocker: MockerFixture,
        checkpoint_file: Tuple[io.BytesIO, Dict[int, TestStorageMetadata], int],
        checkpoint_directory: str,
        chkpt_object_manager: CheckpointObjectManager,
    ) -> None:
        """Tests that an AssertionError is raised for incorrect tensor shapes."""
        # Arrange
        stream, storage_data, stream_size = checkpoint_file
        checkpoint_obj_id = CheckpointObjectId(str(Path(checkpoint_directory) / "checkpoint.data"))
        with chkpt_object_manager.create_buffer(checkpoint_obj_id, buffer_size=stream_size) as f:
            f.write(stream.read())

        req = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=0,
            storage_offsets=(1, 1),
            lengths=(2, 3),  # Expected slice size is 2x3
            dest_index=(0,),
            dest_offsets=(0,),
        )
        # Planner provides a destination tensor with the WRONG size (5x5).
        destination_tensor = torch.empty(5, 5, dtype=torch.int32)

        mock_planner = mocker.MagicMock()
        mock_planner.resolve_tensor.return_value = destination_tensor

        # Act & Assert
        with pytest.raises(AssertionError, match="mismatch sizes"):
            self.loader.read_data(checkpoint_obj_id, [req], mock_planner, storage_data)

        # Also assert that the commit method was never reached.
        mock_planner.commit_tensor.assert_not_called()


class TestGetLatestCompleteCheckpoint:
    """Tests for the get_latest_complete_checkpoint method."""

    @pytest.fixture
    def chkpt_object_manager(self):
        return CheckpointObjectManager()

    @pytest.fixture(autouse=True)
    def loader(self, chkpt_object_manager, mocker) -> MLFlashpointCheckpointLoader:
        mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
        return DefaultMLFlashpointCheckpointLoader(
            checkpoint_object_manager=chkpt_object_manager,
            replication_manager=mock_replication_manager,
        )

    @pytest.fixture(autouse=True)
    def _setup_mocks(self, mocker: MockerFixture) -> None:
        """Sets up common mocks for all tests in this class."""
        self.mock_get_num_nodes = mocker.patch("ml_flashpoint.core.checkpoint_loader.get_num_of_nodes", return_value=1)
        # Mocking specific dist functions that are used
        mocker.patch("torch.distributed.get_rank", return_value=0)
        mocker.patch("torch.distributed.broadcast_object_list")
        mocker.patch("torch.distributed.all_gather_object")
        mocker.patch("torch.distributed.all_reduce")
        self.mock_dist_get_rank = mocker.patch("torch.distributed.get_rank", return_value=0)
        self.mock_dist_get_world_size = mocker.patch("torch.distributed.get_world_size", return_value=2)
        self.mock_dist_all_gather_object = mocker.patch("torch.distributed.all_gather_object")
        self.mock_dist_broadcast_object_list = mocker.patch("torch.distributed.broadcast_object_list")
        self.mock_dist_all_reduce = mocker.patch("torch.distributed.all_reduce")
        self.mock_dist_get_node_local_rank = mocker.patch("torch.distributed.get_node_local_rank", return_value=0)
        self.mock_dist_is_initialized = mocker.patch("torch.distributed.is_initialized", return_value=True)
        self.mock_open = mocker.patch("builtins.open", mocker.mock_open())
        self.mock_path_exists = mocker.patch("os.path.exists", return_value=False)

    def test_no_candidate_checkpoints(self, loader, mocker):
        """Tests that it returns None if no candidate checkpoints are found."""
        base_container = CheckpointContainerId("/tmp/checkpoints")
        mocker.patch.object(loader, "get_candidate_checkpoints", return_value=[])

        assert loader.get_latest_complete_checkpoint(base_container) is None

    def test_rank0_success(self, loader, mocker):
        """Tests successful retrieval flow on Rank 0."""
        # Given
        self.mock_dist_get_rank.return_value = 0
        self.mock_get_num_nodes.return_value = 2

        base_container = CheckpointContainerId("/tmp/checkpoints")
        ckpt_id = CheckpointContainerId.create_child(base_container, "step-100_ckpt")
        mocker.patch.object(loader, "get_candidate_checkpoints", return_value=[ckpt_id])

        # Mock metadata
        mock_metadata = mocker.MagicMock()
        mock_metadata.storage_data = {
            0: _StorageInfo(relative_path="/src0/obj", offset=0, length=100),
            1: _StorageInfo(relative_path="/src1/obj", offset=0, length=100),
        }
        mocker.patch.object(loader, "read_metadata", return_value=mock_metadata)

        # Mock available objects (Rank 0 has src1/obj, Rank 1 has src0/obj) - SWAPPED to force retrieval
        # Both have common.pt to avoid retrieval of it
        mocker.patch.object(
            loader,
            "get_checkpoint_objects_by_rank",
            return_value={
                0: [
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/src1/obj"),
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/common.pt"),
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/.metadata"),
                ],
                1: [
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/src0/obj"),
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/common.pt"),
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/.metadata"),
                ],
            },
        )

        # Mock retrieve_checkpoint success
        mock_retrieve = mocker.patch.object(loader, "retrieve_checkpoint", return_value=True)

        # When
        result = loader.get_latest_complete_checkpoint(base_container)

        # Then
        assert result == ckpt_id

        args, _ = self.mock_dist_broadcast_object_list.call_args
        plan_container = args[0]
        assert len(plan_container) == 1
        plan = plan_container[0]

        # Verify plan content
        assert 0 in plan
        assert (1, "/tmp/checkpoints/step-100_ckpt/src0/obj") in plan[0]

        assert 1 in plan
        assert (0, "/tmp/checkpoints/step-100_ckpt/src1/obj") in plan[1]

        # Verify retrieve called with plan
        mock_retrieve.assert_called_once_with(plan)

    def test_rank0_metadata_failure(self, loader, mocker):
        """Tests Rank 0 handling metadata read failure."""
        self.mock_dist_get_rank.return_value = 0
        # Set num_nodes to 2 to trigger broadcast logic
        self.mock_get_num_nodes.return_value = 2

        base_container = CheckpointContainerId("/tmp/checkpoints")
        ckpt_id = CheckpointContainerId.create_child(base_container, "step-100_ckpt")
        mocker.patch.object(loader, "get_candidate_checkpoints", return_value=[ckpt_id])

        # Metadata read fails
        mocker.patch.object(loader, "read_metadata", side_effect=Exception("Read failed"))

        # Mock available objects to ensure we try to plan
        mocker.patch.object(
            loader,
            "get_checkpoint_objects_by_rank",
            return_value={
                0: [CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/.metadata")],
                1: [],
            },
        )

        assert loader.get_latest_complete_checkpoint(base_container) is None

        # Should still broadcast None to unblock other ranks
        self.mock_dist_broadcast_object_list.assert_called()
        args, _ = self.mock_dist_broadcast_object_list.call_args
        assert args[0] == [None]

    def test_rank0_missing_objects(self, loader, mocker):
        """Tests Rank 0 handling missing objects (cannot satisfy requirements)."""
        self.mock_dist_get_rank.return_value = 0
        self.mock_get_num_nodes.return_value = 2

        base_container = CheckpointContainerId("/tmp/checkpoints")
        ckpt_id = CheckpointContainerId.create_child(base_container, "step-100_ckpt")
        mocker.patch.object(loader, "get_candidate_checkpoints", return_value=[ckpt_id])

        mock_metadata = mocker.MagicMock()
        mock_metadata.storage_data = {0: _StorageInfo(relative_path="/src0/obj", offset=0, length=100)}
        mocker.patch.object(loader, "read_metadata", return_value=mock_metadata)

        # No objects available anywhere (except metadata to trigger planning)
        mocker.patch.object(
            loader,
            "get_checkpoint_objects_by_rank",
            return_value={0: [CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/.metadata")], 1: []},
        )

        assert loader.get_latest_complete_checkpoint(base_container) is None

    def test_rank0_metadata_retrieval(self, loader, mocker):
        """Tests that .metadata is added to retrieval plan for rank 0 if missing."""
        self.mock_dist_get_rank.return_value = 0
        self.mock_get_num_nodes.return_value = 2

        base_container = CheckpointContainerId("/tmp/checkpoints")
        ckpt_id = CheckpointContainerId.create_child(base_container, "step-100_ckpt")
        mocker.patch.object(loader, "get_candidate_checkpoints", return_value=[ckpt_id])

        # Mock metadata read success (we need it to succeed to compute plan)
        mock_metadata = mocker.MagicMock()
        mock_metadata.storage_data = {
            0: _StorageInfo(relative_path="/src0/obj", offset=0, length=100),
        }
        mocker.patch.object(loader, "read_metadata", return_value=mock_metadata)

        # Rank 0 is missing .metadata locally
        mocker.patch.object(
            loader,
            "get_checkpoint_objects_by_rank",
            return_value={
                0: [CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/src0_obj")],  # Missing metadata
                1: [CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/.metadata")],  # Rank 1 has it
            },
        )

        # Mock retrieve_checkpoint success
        mocker.patch.object(loader, "retrieve_checkpoint", return_value=True)

        # Mock broadcast to return a valid plan
        # Rank 0 is NOT planner (Rank 1 is), so Rank 0 receives plan via broadcast.
        expected_plan = {0: [(1, "/tmp/checkpoints/step-100_ckpt/.metadata")]}

        def broadcast_side_effect(obj_list, src):
            if src == 1:
                obj_list[0] = expected_plan

        self.mock_dist_broadcast_object_list.side_effect = broadcast_side_effect

        # When
        result = loader.get_latest_complete_checkpoint(base_container)

        # Then
        assert result == ckpt_id
        # Verify that retrieve_checkpoint was called with the plan received from broadcast
        loader.retrieve_checkpoint.assert_called_once_with(expected_plan)

        base_container = CheckpointContainerId("/tmp/checkpoints")
        ckpt_id = CheckpointContainerId.create_child(base_container, "step-100_ckpt")
        mocker.patch.object(loader, "get_candidate_checkpoints", return_value=[ckpt_id])

        # Metadata read fails initially (simulating missing file)
        mocker.patch.object(loader, "read_metadata", side_effect=Exception("Read failed"))

        # Rank 0 is missing .metadata locally
        mocker.patch.object(
            loader,
            "get_checkpoint_objects_by_rank",
            return_value={
                0: [],  # Missing metadata
                1: [CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/.metadata")],  # Rank 1 has it
            },
        )

        mock_plan = {0: [(1, "/tmp/checkpoints/step-100_ckpt/.metadata")]}
        mocker.patch.object(loader, "_compute_retrieval_plan", return_value=mock_plan)

        mock_retrieve = mocker.patch.object(loader, "retrieve_checkpoint", return_value=True)

        # When
        result = loader.get_latest_complete_checkpoint(base_container)

        # Then
        assert result == ckpt_id
        mock_retrieve.assert_called_once_with(mock_plan)

    def test_no_retrieval_needed(self, loader, mocker):
        """Tests that no retrieval is attempted if the plan is empty."""
        # Given
        self.mock_dist_get_rank.return_value = 0
        self.mock_get_num_nodes.return_value = 1

        base_container = CheckpointContainerId("/tmp/checkpoints")
        ckpt_id = CheckpointContainerId.create_child(base_container, "step-100_ckpt")
        mocker.patch.object(loader, "get_candidate_checkpoints", return_value=[ckpt_id])

        # Mock metadata
        mock_metadata = mocker.MagicMock()
        mock_metadata.storage_data = {
            0: _StorageInfo(relative_path="/src0/obj", offset=0, length=100),
        }
        mocker.patch.object(loader, "read_metadata", return_value=mock_metadata)

        # Mock available objects (Rank 0 has everything needed)
        mocker.patch.object(
            loader,
            "get_checkpoint_objects_by_rank",
            return_value={
                0: [
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/src0/obj"),
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/common.pt"),
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/.metadata"),
                ],
            },
        )

        # Mock _compute_retrieval_plan to return empty dict (no retrieval needed)
        mocker.patch.object(loader, "_compute_retrieval_plan", return_value={})

        mock_retrieve = mocker.patch.object(loader, "retrieve_checkpoint")

        # When
        result = loader.get_latest_complete_checkpoint(base_container)

        # Then
        assert result == ckpt_id
        mock_retrieve.assert_not_called()

    def test_non_rank0_success(self, loader, mocker):
        """Tests successful retrieval flow on non-Rank 0."""
        # Given
        self.mock_dist_get_rank.return_value = 1
        self.mock_get_num_nodes.return_value = 2

        base_container = CheckpointContainerId("/tmp/checkpoints")
        ckpt_id = CheckpointContainerId.create_child(base_container, "step-100_ckpt")
        mocker.patch.object(loader, "get_candidate_checkpoints", return_value=[ckpt_id])

        # Mock sending objects (get_checkpoint_objects_by_rank)
        mock_get_objs = mocker.patch.object(
            loader,
            "get_checkpoint_objects_by_rank",
            return_value={
                0: [
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/src1/obj"),
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/common.pt"),
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/.metadata"),
                ],
                1: [
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/src0/obj"),
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/common.pt"),
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/.metadata"),
                ],
            },
        )

        # Mock receiving plan
        plan = {1: [(0, "/tmp/checkpoints/step-100_ckpt/src0/obj")]}

        def side_effect_broadcast(obj_list, src):
            obj_list[0] = plan

        self.mock_dist_broadcast_object_list.side_effect = side_effect_broadcast

        mock_retrieve = mocker.patch.object(loader, "retrieve_checkpoint", return_value=True)

        # When
        result = loader.get_latest_complete_checkpoint(base_container)

        # Then
        assert result == ckpt_id
        mock_get_objs.assert_called_once()
        mock_retrieve.assert_called_once_with(plan)

    def test_non_rank0_failure(self, loader, mocker):
        """Tests non-Rank 0 handling failure (receiving None plan)."""
        self.mock_dist_get_rank.return_value = 1
        self.mock_get_num_nodes.return_value = 2

        base_container = CheckpointContainerId("/tmp/checkpoints")
        ckpt_id = CheckpointContainerId.create_child(base_container, "step-100_ckpt")
        mocker.patch.object(loader, "get_candidate_checkpoints", return_value=[ckpt_id])

        mocker.patch.object(loader, "get_checkpoint_objects_by_rank")

        # Receive None plan
        def side_effect_broadcast(obj_list, src):
            obj_list[0] = None

        self.mock_dist_broadcast_object_list.side_effect = side_effect_broadcast

        mock_retrieve = mocker.patch.object(loader, "retrieve_checkpoint")

        result = loader.get_latest_complete_checkpoint(base_container)

        assert result is None
        mock_retrieve.assert_not_called()

    def test_rank0_retrieval_failure(self, loader, mocker):
        """Tests Rank 0 handling retrieval failure."""
        self.mock_dist_get_rank.return_value = 0
        self.mock_get_num_nodes.return_value = 2

        base_container = CheckpointContainerId("/tmp/checkpoints")
        ckpt_id = CheckpointContainerId.create_child(base_container, "step-100_ckpt")
        mocker.patch.object(loader, "get_candidate_checkpoints", return_value=[ckpt_id])

        mock_metadata = mocker.MagicMock()
        mock_metadata.storage_data = {
            0: _StorageInfo(relative_path="/src0/obj", offset=0, length=100),
            1: _StorageInfo(relative_path="/src1/obj", offset=0, length=100),
        }
        mocker.patch.object(loader, "read_metadata", return_value=mock_metadata)
        mocker.patch.object(
            loader,
            "get_checkpoint_objects_by_rank",
            return_value={
                0: [
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/src1/obj"),
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/common.pt"),
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/.metadata"),
                ],
                1: [
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/src0/obj"),
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/common.pt"),
                    CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/.metadata"),
                ],
            },
        )

        # Retrieve fails
        mocker.patch.object(loader, "retrieve_checkpoint", return_value=False)

        assert loader.get_latest_complete_checkpoint(base_container) is None

    def test_compute_retrieval_plan_success(self, loader, mocker):
        """Tests _compute_retrieval_plan with valid data."""
        checkpoint = CheckpointContainerId("/tmp/ckpt")
        mock_metadata = mocker.MagicMock()
        mock_metadata.storage_data = {
            0: _StorageInfo(relative_path="src0/obj", offset=0, length=100),
            1: _StorageInfo(relative_path="src1/obj", offset=0, length=100),
        }
        mocker.patch.object(loader, "read_metadata", return_value=mock_metadata)

        available_objects = {
            0: [
                CheckpointObjectId("/tmp/ckpt/src1/obj"),
                CheckpointObjectId("/tmp/ckpt/common.pt"),
                CheckpointObjectId("/tmp/ckpt/.metadata"),
            ],  # Rank 0 has rank 1's object + common.pt + .metadata
            1: [CheckpointObjectId("/tmp/ckpt/src0/obj")],  # Rank 1 has rank 0's object
        }

        plan = loader._compute_retrieval_plan(checkpoint, available_objects)

        assert plan[0] == [(1, "/tmp/ckpt/src0/obj")]
        assert plan[1] == [(0, "/tmp/ckpt/src1/obj")]

    def test_compute_retrieval_plan_missing_object(self, loader, mocker):
        """Tests _compute_retrieval_plan when an object is missing globally."""
        checkpoint = CheckpointContainerId("/tmp/ckpt")
        mock_metadata = mocker.MagicMock()
        mock_metadata.storage_data = {
            0: _StorageInfo(relative_path="src0/obj", offset=0, length=100),
        }
        mocker.patch.object(loader, "read_metadata", return_value=mock_metadata)

        available_objects = {
            0: [],
            1: [],
        }

        plan = loader._compute_retrieval_plan(checkpoint, available_objects)
        assert plan is None

    def test_compute_retrieval_plan_no_storage_data(self, loader, mocker):
        """Tests _compute_retrieval_plan when storage_data is None."""
        checkpoint = CheckpointContainerId("/tmp/ckpt")
        mock_metadata = mocker.MagicMock()
        mock_metadata.storage_data = None
        mocker.patch.object(loader, "read_metadata", return_value=mock_metadata)

        available_objects = {0: []}
        plan = loader._compute_retrieval_plan(checkpoint, available_objects)
        assert plan is None

    def test_planner_enforces_metadata_for_remote_node(self, loader, mocker):
        """Tests that planner enforces metadata requirement for remote local_rank 0."""
        checkpoint = CheckpointContainerId("/tmp/ckpt")
        mock_metadata = mocker.MagicMock()
        mock_metadata.storage_data = {
            0: _StorageInfo(relative_path="src0/obj", offset=0, length=100),
            1: _StorageInfo(relative_path="src1/obj", offset=0, length=100),
        }
        mocker.patch.object(loader, "read_metadata", return_value=mock_metadata)

        # Mock 2 nodes, 1 rank per node.
        # Rank 0 is local_rank 0. Rank 1 is local_rank 0.
        mocker.patch("ml_flashpoint.core.checkpoint_loader.get_num_of_nodes", return_value=2)
        mocker.patch("torch.distributed.get_world_size", return_value=2)

        available_objects = {
            0: [
                CheckpointObjectId("/tmp/ckpt/src0/obj"),
                CheckpointObjectId("/tmp/ckpt/common.pt"),
                CheckpointObjectId("/tmp/ckpt/.metadata"),
            ],
            1: [
                CheckpointObjectId("/tmp/ckpt/src1/obj"),
                CheckpointObjectId("/tmp/ckpt/common.pt"),
                # Rank 1 MISSING .metadata
            ],
        }

        plan = loader._compute_retrieval_plan(checkpoint, available_objects)

        # Rank 1 should retrieve .metadata from Rank 0
        assert 1 in plan
        metadata_retrieval = (0, "/tmp/ckpt/.metadata")
        assert metadata_retrieval in plan[1]

    def test_planner_selection_first_rank(self, loader, mocker):
        """Tests that the planner is selected as the first rank (lowest ID) from candidates with metadata."""
        self.mock_dist_get_rank.return_value = 0
        self.mock_get_num_nodes.return_value = 2

        base_container = CheckpointContainerId("/tmp/checkpoints")
        ckpt_id = CheckpointContainerId.create_child(base_container, "step-100_ckpt")
        mocker.patch.object(loader, "get_candidate_checkpoints", return_value=[ckpt_id])

        # Mock available objects: Rank 1 and Rank 2 have metadata
        mocker.patch.object(
            loader,
            "get_checkpoint_objects_by_rank",
            return_value={
                0: [CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/obj1")],
                1: [CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/.metadata")],
                2: [CheckpointObjectId("/tmp/checkpoints/step-100_ckpt/.metadata")],
            },
        )

        # Mock compute_retrieval_plan (should NOT be called on Rank 0)
        mock_compute = mocker.patch.object(loader, "_compute_retrieval_plan")

        # Mock broadcast (should be called with src=1, since 1 < 2)
        mock_broadcast = self.mock_dist_broadcast_object_list

        # When
        loader.get_latest_complete_checkpoint(base_container)

        # Then
        mock_compute.assert_not_called()  # We are rank 0, planner is 1

        mock_broadcast.assert_called_once()
        assert mock_broadcast.call_args[1]["src"] == 1


class TestGetCandidateCheckpoints:
    """Tests for the get_candidate_checkpoints method."""

    @pytest.fixture
    def chkpt_object_manager(self):
        return CheckpointObjectManager()

    @pytest.fixture(autouse=True)
    def loader(self, chkpt_object_manager, mocker) -> MLFlashpointCheckpointLoader:
        mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
        return DefaultMLFlashpointCheckpointLoader(
            checkpoint_object_manager=chkpt_object_manager,
            replication_manager=mock_replication_manager,
        )

    @pytest.fixture(autouse=True)
    def _setup_mocks(self, mocker: MockerFixture) -> None:
        """Sets up common mocks for all tests in this class."""
        self.mock_get_num_nodes = mocker.patch("ml_flashpoint.core.checkpoint_loader.get_num_of_nodes")
        self.mock_path_is_dir = mocker.patch("pathlib.Path.is_dir", return_value=True)
        self.mock_dist_get_rank = mocker.patch("torch.distributed.get_rank")
        self.mock_dist_get_node_local_rank = mocker.patch("torch.distributed.get_node_local_rank", return_value=0)
        self.mock_broadcast = mocker.patch("torch.distributed.broadcast_object_list")
        self.mock_all_gather = mocker.patch("torch.distributed.all_gather_object")

        def side_effect(out_list, in_obj):
            out_list[0] = in_obj

        self.mock_all_gather.side_effect = side_effect

        def side_effect(out_list, in_obj):
            out_list[0] = in_obj

        self.mock_all_gather.side_effect = side_effect
        self.mock_dist_get_world_size = mocker.patch("torch.distributed.get_world_size", return_value=1)

    def test_single_node_multiple_complete_checkpoints(
        self,
        loader: DefaultMLFlashpointCheckpointLoader,
        tmp_path: Path,
    ):
        """Tests that in a single-node setup, it returns all complete checkpoints."""
        # Given
        self.mock_get_num_nodes.return_value = 1
        self.mock_dist_get_rank.return_value = 0
        base_container = CheckpointContainerId(str(tmp_path))

        (tmp_path / "step-100_ckpt").mkdir()
        (tmp_path / "step-300_ckpt").mkdir()
        (tmp_path / "step-200_ckpt").mkdir()

        # When
        result = loader.get_candidate_checkpoints(base_container)

        # Then
        expected = [
            CheckpointContainerId(str(tmp_path / "step-300_ckpt")),
            CheckpointContainerId(str(tmp_path / "step-200_ckpt")),
            CheckpointContainerId(str(tmp_path / "step-100_ckpt")),
        ]
        assert result == expected
        self.mock_broadcast.assert_not_called()

    def test_single_node_mixed_checkpoints(
        self,
        loader: DefaultMLFlashpointCheckpointLoader,
        tmp_path: Path,
    ):
        """Tests that unfinished checkpoints that should be filtered out."""
        # Given
        self.mock_get_num_nodes.return_value = 1
        self.mock_dist_get_rank.return_value = 0
        base_container = CheckpointContainerId(str(tmp_path))

        (tmp_path / "step-100_ckpt").mkdir()
        (tmp_path / "step-300_ckpt").mkdir()
        (tmp_path / "step-300_ckpt_rank0__unfinished").touch()
        (tmp_path / "step-200_ckpt").mkdir()

        # When
        result = loader.get_candidate_checkpoints(base_container)

        # Then
        expected = [
            CheckpointContainerId(str(tmp_path / "step-200_ckpt")),
            CheckpointContainerId(str(tmp_path / "step-100_ckpt")),
        ]
        assert result == expected
        self.mock_broadcast.assert_not_called()

    def test_distributed_rank0(
        self,
        loader: DefaultMLFlashpointCheckpointLoader,
        tmp_path: Path,
    ):
        """Tests Rank 0 behavior in a distributed setup."""
        # Given
        self.mock_get_num_nodes.return_value = 2
        self.mock_dist_get_rank.return_value = 0
        self.mock_dist_get_world_size.return_value = 2
        base_container = CheckpointContainerId(str(tmp_path))

        (tmp_path / "step-100_ckpt").mkdir()
        (tmp_path / "step-200_ckpt").mkdir()

        def all_gather_side_effect(out_list, in_obj):
            # Simulates gathering from all ranks
            # Rank 0 (local) found checkpoints, Rank 1 found nothing
            out_list[0] = in_obj  # Rank 0's data
            out_list[1] = []  # Rank 1's data

        self.mock_all_gather.side_effect = all_gather_side_effect

        # When
        result = loader.get_candidate_checkpoints(base_container)

        # Then
        expected = [
            CheckpointContainerId(str(tmp_path / "step-200_ckpt")),
            CheckpointContainerId(str(tmp_path / "step-100_ckpt")),
        ]
        assert result == expected
        self.mock_all_gather.assert_called_once()
        self.mock_broadcast.assert_not_called()

    def test_distributed_non_rank0(
        self,
        loader: DefaultMLFlashpointCheckpointLoader,
        tmp_path: Path,
    ):
        """Tests non-Rank 0 behavior in a distributed setup."""
        # Given
        self.mock_get_num_nodes.return_value = 2
        self.mock_dist_get_rank.return_value = 1
        self.mock_dist_get_world_size.return_value = 2
        base_container = CheckpointContainerId(str(tmp_path))

        def all_gather_side_effect(out_list, in_obj):
            out_list[0] = [
                str(tmp_path / "step-200_ckpt"),
                str(tmp_path / "step-100_ckpt"),
            ]  # Rank 0's data (full paths)
            out_list[1] = in_obj  # Rank 1's data

        self.mock_all_gather.side_effect = all_gather_side_effect
        # Rank 1 finds nothing (tmp_path is empty)

        # When
        result = loader.get_candidate_checkpoints(base_container)

        # Then
        expected = [
            CheckpointContainerId(str(tmp_path / "step-200_ckpt")),
            CheckpointContainerId(str(tmp_path / "step-100_ckpt")),
        ]
        assert result == expected
        self.mock_all_gather.assert_called_once()
        self.mock_broadcast.assert_not_called()

    def test_multi_node_one_node_has_no_checkpoints(
        self,
        loader: DefaultMLFlashpointCheckpointLoader,
        tmp_path: Path,
    ):
        """Tests that the union includes checkpoints even if one node has none."""
        # Given
        self.mock_get_num_nodes.return_value = 2
        self.mock_dist_get_rank.return_value = 0  # This test is for rank 0 behavior
        self.mock_dist_get_world_size.return_value = 2
        base_container = CheckpointContainerId(str(tmp_path))

        # Local node has step 100 and 200
        (tmp_path / "step-100_ckpt").mkdir()
        (tmp_path / "step-200_ckpt").mkdir()

        # Mock broadcast side effect to simulate other ranks' contributions
        def broadcast_side_effect(obj_list, src):
            if src == 0:  # Rank 0 is sending its list
                obj_list[0] = [str(tmp_path / "step-200_ckpt"), str(tmp_path / "step-100_ckpt")]

        # Mock all_gather side effect to simulate other ranks' contributions
        def all_gather_side_effect(out_list, in_obj):
            out_list[0] = in_obj  # Rank 0 data
            out_list[1] = []  # Rank 1 data (empty)

        self.mock_all_gather.side_effect = all_gather_side_effect

        # When
        result = loader.get_candidate_checkpoints(base_container)

        # Then
        expected = [
            CheckpointContainerId(str(tmp_path / "step-200_ckpt")),
            CheckpointContainerId(str(tmp_path / "step-100_ckpt")),
        ]
        assert result == expected
        self.mock_all_gather.assert_called_once()
        self.mock_broadcast.assert_not_called()

    def test_multi_node_intersection(
        self,
        loader: DefaultMLFlashpointCheckpointLoader,
        tmp_path: Path,
    ):
        """Tests intersection of checkpoints across nodes."""
        # Given
        self.mock_get_num_nodes.return_value = 2
        self.mock_dist_get_rank.return_value = 0
        self.mock_dist_get_world_size.return_value = 2
        base_container = CheckpointContainerId(str(tmp_path))

        # Node 1 (Rank 0) has A, B
        (tmp_path / "step-100_ckpt").mkdir()
        (tmp_path / "step-200_ckpt").mkdir()

        def all_gather_side_effect(out_list, in_obj):
            out_list[0] = in_obj  # Rank 0: {100, 200}
            # Node 2 (Rank 1) has B, C
            out_list[1] = [str(tmp_path / "step-200_ckpt"), str(tmp_path / "step-300_ckpt")]

        self.mock_all_gather.side_effect = all_gather_side_effect

        # When
        result = loader.get_candidate_checkpoints(base_container)

        # Then
        # Intersection is {200}
        expected = [CheckpointContainerId(str(tmp_path / "step-200_ckpt"))]
        assert result == expected

    def test_multi_node_disjoint(
        self,
        loader: DefaultMLFlashpointCheckpointLoader,
        tmp_path: Path,
    ):
        """Tests disjoint sets of checkpoints across nodes."""
        # Given
        self.mock_get_num_nodes.return_value = 2
        self.mock_dist_get_rank.return_value = 0
        self.mock_dist_get_world_size.return_value = 2
        base_container = CheckpointContainerId(str(tmp_path))

        # Node 1 (Rank 0) has A
        (tmp_path / "step-100_ckpt").mkdir()

        def all_gather_side_effect(out_list, in_obj):
            out_list[0] = in_obj  # Rank 0: {100}
            # Node 2 (Rank 1) has B
            out_list[1] = [str(tmp_path / "step-200_ckpt")]

        self.mock_all_gather.side_effect = all_gather_side_effect

        # When
        result = loader.get_candidate_checkpoints(base_container)

        # Then
        expected = []
        assert result == expected

    def test_multi_node_subset(
        self,
        loader: DefaultMLFlashpointCheckpointLoader,
        tmp_path: Path,
    ):
        """Tests subset relationship of checkpoints across nodes."""
        # Given
        self.mock_get_num_nodes.return_value = 2
        self.mock_dist_get_rank.return_value = 0
        self.mock_dist_get_world_size.return_value = 2
        base_container = CheckpointContainerId(str(tmp_path))

        # Node 1 (Rank 0) has A, B
        (tmp_path / "step-100_ckpt").mkdir()
        (tmp_path / "step-200_ckpt").mkdir()

        def all_gather_side_effect(out_list, in_obj):
            out_list[0] = in_obj  # Rank 0: {100, 200}
            # Node 2 (Rank 1) has A
            out_list[1] = [str(tmp_path / "step-100_ckpt")]

        self.mock_all_gather.side_effect = all_gather_side_effect

        # When
        result = loader.get_candidate_checkpoints(base_container)

        # Then
        # Intersection is {100}
        expected = [CheckpointContainerId(str(tmp_path / "step-100_ckpt"))]
        assert result == expected

    def test_multi_node_with_unfinished_markers(
        self,
        loader: DefaultMLFlashpointCheckpointLoader,
        tmp_path: Path,
    ):
        """Tests that unfinished markers should cause checkpoints to be excluded."""
        # Given
        self.mock_get_num_nodes.return_value = 2
        self.mock_dist_get_rank.return_value = 0
        self.mock_dist_get_world_size.return_value = 2
        base_container = CheckpointContainerId(str(tmp_path))

        # Node 1 (Rank 0) has A, B
        (tmp_path / "step-100_ckpt").mkdir()
        (tmp_path / "step-200_ckpt").mkdir()
        (tmp_path / "step-200_ckpt_rank0__unfinished").touch()

        def all_gather_side_effect(out_list, in_obj):
            out_list[0] = in_obj
            out_list[1] = [str(tmp_path / "step-100_ckpt")]

        self.mock_all_gather.side_effect = all_gather_side_effect

        # When
        result = loader.get_candidate_checkpoints(base_container)

        # Then
        # Intersection is {100}
        expected = [CheckpointContainerId(str(tmp_path / "step-100_ckpt"))]
        assert result == expected

    def test_single_node_correct_numerical_sorting(
        self,
        loader: DefaultMLFlashpointCheckpointLoader,
        tmp_path: Path,
    ):
        """Tests that checkpoints are sorted numerically by step, not lexicographically."""
        # Given
        self.mock_get_num_nodes.return_value = 1
        self.mock_dist_get_rank.return_value = 0
        base_container = CheckpointContainerId(str(tmp_path))

        (tmp_path / "step-9_ckpt").mkdir()
        (tmp_path / "step-10_ckpt").mkdir()
        (tmp_path / "step-1_ckpt").mkdir()

        # When
        result = loader.get_candidate_checkpoints(base_container)

        # Then
        expected = [
            CheckpointContainerId(str(tmp_path / "step-10_ckpt")),
            CheckpointContainerId(str(tmp_path / "step-9_ckpt")),
            CheckpointContainerId(str(tmp_path / "step-1_ckpt")),
        ]
        assert result == expected

    def test_get_candidate_checkpoints_fallback_path(
        self,
        loader: DefaultMLFlashpointCheckpointLoader,
        tmp_path: Path,
        mocker,
    ):
        """Tests fallback to full path when relative_to fails."""
        # Given
        mocker.patch("ml_flashpoint.core.checkpoint_loader.get_num_of_nodes", return_value=2)
        mocker.patch("torch.distributed.get_rank", return_value=0)
        mocker.patch("torch.distributed.get_world_size", return_value=2)
        mock_broadcast = mocker.patch("torch.distributed.broadcast_object_list")

        base_container = CheckpointContainerId("/tmp/base")
        # Create a checkpoint outside base_container to force relative_to failure
        # We can't easily create a file outside tmp_path in a safe way if we want to use real files.
        # However, the test is about relative_to failure.
        # If we use real files, we need base_container to be a real path.

        # Let's use tmp_path as base_container, but mock relative_to to fail.
        base_container = CheckpointContainerId(str(tmp_path))
        (tmp_path / "step-100_ckpt").mkdir()

        mocker.patch("pathlib.Path.is_dir", return_value=True)

        # Mock Path.relative_to to raise ValueError only for our specific call
        # Pytest uses Path.relative_to internally, so we must be careful.
        orig_relative_to = Path.relative_to

        def side_effect_relative_to(self, other):
            if str(other) == str(base_container.data):
                raise ValueError("Not relative")
            return orig_relative_to(self, other)

        mocker.patch("pathlib.Path.relative_to", side_effect=side_effect_relative_to, autospec=True)
        # When
        loader.get_candidate_checkpoints(base_container)

        # Then
        # Mock all_gather to return the data
        def all_gather_side_effect(out_list, in_obj):
            out_list[0] = in_obj
            out_list[1] = []

        mock_all_gather = mocker.patch("torch.distributed.all_gather_object", side_effect=all_gather_side_effect)

        # When
        loader.get_candidate_checkpoints(base_container)

        # Then
        mock_all_gather.assert_called_once()
        mock_broadcast.assert_not_called()
        args, _ = mock_all_gather.call_args
        # args[1] is the input object (list of paths)
        assert len(args[1]) == 1


class TestRetrieveCheckpoint:
    @pytest.fixture
    def chkpt_object_manager(self):
        return CheckpointObjectManager()

    @pytest.fixture
    def loader(self, mocker, chkpt_object_manager):
        mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
        return DefaultMLFlashpointCheckpointLoader(
            checkpoint_object_manager=chkpt_object_manager,
            replication_manager=mock_replication_manager,
        )

    def test_retrieve_checkpoint_local_has_all_objects(self, loader, mocker):
        mocker.patch("torch.distributed.get_rank", return_value=0)
        mocker.patch("torch.distributed.get_world_size", return_value=1)
        mock_all_gather = mocker.patch("torch.distributed.all_gather_object")

        def side_effect(out_list, in_obj):
            out_list[0] = in_obj

        mock_all_gather.side_effect = side_effect

        # Plan is empty for rank 0
        plan = {0: []}
        assert loader.retrieve_checkpoint(plan) is True
        loader._replication_manager.sync_bulk_retrieve.assert_not_called()

    def test_retrieve_checkpoint_distributed_success(self, loader, mocker):
        mocker.patch("ml_flashpoint.core.checkpoint_loader.get_num_of_nodes", return_value=2)
        mocker.patch("torch.distributed.get_rank", return_value=0)
        mocker.patch("torch.distributed.get_world_size", return_value=2)
        mock_all_gather = mocker.patch("torch.distributed.all_gather_object")

        # Mock all_gather_object to populate success list with True
        def side_effect_gather(obj_list, obj):
            for i in range(len(obj_list)):
                obj_list[i] = True

        mock_all_gather.side_effect = side_effect_gather

        plan = {0: [(1, "/obj1")]}
        loader._replication_manager.sync_bulk_retrieve.return_value = True

        assert loader.retrieve_checkpoint(plan) is True
        mock_all_gather.assert_called_once()

    def test_retrieve_checkpoint_missing_objects_retrieved(self, loader, mocker):
        mocker.patch("torch.distributed.get_rank", return_value=0)
        mocker.patch("torch.distributed.get_world_size", return_value=1)
        mock_all_gather = mocker.patch("torch.distributed.all_gather_object")

        def side_effect(out_list, in_obj):
            out_list[0] = in_obj

        mock_all_gather.side_effect = side_effect

        # Plan says rank 0 needs to retrieve obj1 from rank 1
        plan = {0: [(1, "/obj1")]}

        loader._replication_manager.sync_bulk_retrieve.return_value = True

        assert loader.retrieve_checkpoint(plan) is True

        loader._replication_manager.sync_bulk_retrieve.assert_called_once_with(
            source_global_rank=1,
            object_ids_to_retrieve=[CheckpointObjectId("/obj1")],
            container_ids_to_retrieve=[],
        )

    def test_retrieve_checkpoint_failure(self, loader, mocker):
        mocker.patch("torch.distributed.get_rank", return_value=0)
        mocker.patch("torch.distributed.get_world_size", return_value=1)
        mock_all_gather = mocker.patch("torch.distributed.all_gather_object")

        def side_effect(out_list, in_obj):
            out_list[0] = in_obj

        mock_all_gather.side_effect = side_effect

        plan = {0: [(1, "/obj1")]}

        loader._replication_manager.sync_bulk_retrieve.return_value = False

        assert loader.retrieve_checkpoint(plan) is False


class TestGetCandidateCheckpointsOptimization:
    @pytest.fixture
    def chkpt_object_manager(self):
        return CheckpointObjectManager()

    @pytest.fixture(autouse=True)
    def loader(self, chkpt_object_manager, mocker) -> MLFlashpointCheckpointLoader:
        mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
        return DefaultMLFlashpointCheckpointLoader(
            checkpoint_object_manager=chkpt_object_manager,
            replication_manager=mock_replication_manager,
        )

    def test_get_candidate_checkpoints_optimization(self, loader, mocker, checkpoint_directory):
        """Verifies that relative paths are sent during broadcast."""
        # Setup directory structure
        base_dir = Path(checkpoint_directory)
        ckpt_dir = base_dir / "step-100_ckpt"
        ckpt_dir.mkdir()

        mocker.patch("ml_flashpoint.core.checkpoint_loader.get_num_of_nodes", return_value=2)
        mocker.patch("torch.distributed.get_rank", return_value=0)
        mocker.patch("torch.distributed.get_node_local_rank", return_value=0)
        mock_broadcast = mocker.patch("torch.distributed.broadcast_object_list")
        mock_all_gather = mocker.patch("torch.distributed.all_gather_object")
        mocker.patch("torch.distributed.get_world_size", return_value=2)

        mocker.patch("pathlib.Path.is_dir", return_value=True)

        base_container = CheckpointContainerId(str(base_dir))

        # Call method
        loader.get_candidate_checkpoints(base_container)

        # Verify broadcast called with relative path
        # Mock all_gather
        def all_gather_side_effect(out_list, in_obj):
            out_list[0] = [str(base_dir / "step-100_ckpt")]
            out_list[1] = []

        mock_all_gather.side_effect = all_gather_side_effect

        # Call method
        loader.get_candidate_checkpoints(base_container)

        mock_broadcast.assert_not_called()
        args, _ = mock_all_gather.call_args
        # args[1] is the input object (list of paths)
        assert args[1] == [str(base_dir / "step-100_ckpt")]


class TestGetCheckpointObjectsByNodeOptimization:
    @pytest.fixture
    def chkpt_object_manager(self):
        return CheckpointObjectManager()

    @pytest.fixture(autouse=True)
    def loader(self, chkpt_object_manager, mocker) -> MLFlashpointCheckpointLoader:
        mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
        return DefaultMLFlashpointCheckpointLoader(
            checkpoint_object_manager=chkpt_object_manager,
            replication_manager=mock_replication_manager,
        )

    @pytest.fixture(autouse=True)
    def _setup_mocks(self, mocker):
        mocker.patch("torch.distributed.get_node_local_rank", return_value=0)

    def test_get_checkpoint_objects_by_rank_optimization(self, loader, mocker, checkpoint_directory):
        """Verifies that only filenames are sent during all_gather."""
        # Setup directory structure
        base_dir = Path(checkpoint_directory)
        ckpt_dir = base_dir / "step-100_ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "obj1").touch()
        (ckpt_dir / "obj2").touch()

        mocker.patch("ml_flashpoint.core.checkpoint_loader.get_num_of_nodes", return_value=2)
        mocker.patch("torch.distributed.get_rank", return_value=0)
        mocker.patch("torch.distributed.get_world_size", return_value=2)
        mock_all_gather = mocker.patch("torch.distributed.all_gather_object")

        # Mock all_gather to return some data for rank 1
        def side_effect_all_gather(out_list, in_obj):
            out_list[0] = ["obj1", "obj2"]  # Rank 0 data
            out_list[1] = ["obj3"]  # Rank 1 data (filename only)

        mock_all_gather.side_effect = side_effect_all_gather

        container_id = CheckpointContainerId(str(ckpt_dir))

        # Call method
        result = loader.get_checkpoint_objects_by_rank(container_id)

        # Verify all_gather called with filenames only
        mock_all_gather.assert_called_once()
        args, _ = mock_all_gather.call_args
        # args[1] is the input object (list of filenames for rank 0)
        assert set(args[1]) == {"obj1", "obj2"}

        # Verify result has full paths
        assert 0 in result
        assert set(str(o) for o in result[0]) == {str(ckpt_dir / "obj1"), str(ckpt_dir / "obj2")}
        assert 1 in result
        assert set(str(o) for o in result[1]) == {str(ckpt_dir / "obj3")}

    def test_get_checkpoint_objects_by_rank_empty_rank(self, loader, mocker, checkpoint_directory):
        """Verifies handling of empty object list from a rank."""
        base_dir = Path(checkpoint_directory)
        ckpt_dir = base_dir / "step-100_ckpt"
        ckpt_dir.mkdir()

        mocker.patch("ml_flashpoint.core.checkpoint_loader.get_num_of_nodes", return_value=2)
        mocker.patch("torch.distributed.get_rank", return_value=0)
        mocker.patch("torch.distributed.get_world_size", return_value=2)
        mock_all_gather = mocker.patch("torch.distributed.all_gather_object")

        def side_effect_all_gather(out_list, in_obj):
            out_list[0] = []  # Rank 0 empty
            out_list[1] = []  # Rank 1 empty

        mock_all_gather.side_effect = side_effect_all_gather

        container_id = CheckpointContainerId(str(ckpt_dir))
        result = loader.get_checkpoint_objects_by_rank(container_id)

        assert result[0] == []
        assert result[1] == []

    def test_get_checkpoint_objects_by_rank_local(self, loader, mocker, checkpoint_directory):
        """Verifies local-only path."""
        base_dir = Path(checkpoint_directory)
        ckpt_dir = base_dir / "step-100_ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "obj1").touch()

        mocker.patch("ml_flashpoint.core.checkpoint_loader.get_num_of_nodes", return_value=1)
        mocker.patch("torch.distributed.get_rank", return_value=0)
        mocker.patch("torch.distributed.get_world_size", return_value=1)
        mock_all_gather = mocker.patch("torch.distributed.all_gather_object")

        def side_effect(out_list, in_obj):
            out_list[0] = in_obj

        mock_all_gather.side_effect = side_effect

        container_id = CheckpointContainerId(str(ckpt_dir))
        result = loader.get_checkpoint_objects_by_rank(container_id)

        assert 0 in result
        assert len(result[0]) == 1
        assert str(result[0][0]) == str(ckpt_dir / "obj1")
        assert loader._available_objects_cache[container_id][str(ckpt_dir / "obj1")] == [0]

    def test_get_checkpoint_objects_by_rank_unfinished_marker(self, loader, mocker, checkpoint_directory):
        """Verifies that objects are NOT ignored even if an unfinished marker exists (user request)."""
        base_dir = Path(checkpoint_directory)
        ckpt_dir = base_dir / "step-100_ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "model.src0.distcp").touch()
        (ckpt_dir / "other.txt").touch()

        # Create unfinished marker for rank 0
        (base_dir / "step-100_ckpt__0__unfinished").touch()

        mocker.patch("ml_flashpoint.core.checkpoint_loader.get_num_of_nodes", return_value=1)
        mocker.patch("torch.distributed.get_rank", return_value=0)
        mocker.patch("torch.distributed.get_world_size", return_value=1)
        mock_all_gather = mocker.patch("torch.distributed.all_gather_object")

        def side_effect(out_list, in_obj):
            out_list[0] = in_obj

        mock_all_gather.side_effect = side_effect

        container_id = CheckpointContainerId(str(ckpt_dir))
        result = loader.get_checkpoint_objects_by_rank(container_id)

        # Should return all files
        assert 0 in result
        assert len(result[0]) == 2
        local_filenames = [Path(obj.data).name for obj in result[0]]
        assert "model.src0.distcp" in local_filenames
        assert "other.txt" in local_filenames


class TestCheckpointLoaderLocalRank:
    @pytest.fixture
    def loader(self, mocker):
        chkpt_obj_manager = mocker.MagicMock(spec=CheckpointObjectManager)
        replication_manager = mocker.MagicMock(spec=ReplicationManager)
        return DefaultMLFlashpointCheckpointLoader(chkpt_obj_manager, replication_manager)

    def test_get_checkpoint_objects_by_rank_local_rank_0(self, mocker, loader, tmp_path):
        # Setup
        mock_get_node_local_rank = mocker.patch("ml_flashpoint.core.checkpoint_loader.dist.get_node_local_rank")
        mock_get_rank = mocker.patch("ml_flashpoint.core.checkpoint_loader.dist.get_rank")
        mock_get_world_size = mocker.patch("ml_flashpoint.core.checkpoint_loader.dist.get_world_size")
        mock_get_num_nodes = mocker.patch("ml_flashpoint.core.checkpoint_loader.get_num_of_nodes")
        mock_all_gather = mocker.patch("ml_flashpoint.core.checkpoint_loader.dist.all_gather_object")

        def side_effect(out_list, in_obj):
            out_list[0] = in_obj

        mock_all_gather.side_effect = side_effect

        mock_get_node_local_rank.return_value = 0
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 1
        mock_get_num_nodes.return_value = 1

        container_path = tmp_path / "ckpt"
        container_path.mkdir()
        (container_path / "file1").touch()

        container_id = CheckpointContainerId(str(container_path))

        # Execute
        result = loader.get_checkpoint_objects_by_rank(container_id)

        # Verify
        assert result[0] == [CheckpointObjectId(str(container_path / "file1"))]

    def test_get_checkpoint_objects_by_rank_local_rank_1(self, mocker, loader, tmp_path):
        # Setup
        mock_get_node_local_rank = mocker.patch("ml_flashpoint.core.checkpoint_loader.dist.get_node_local_rank")
        mock_get_rank = mocker.patch("ml_flashpoint.core.checkpoint_loader.dist.get_rank")
        mock_get_world_size = mocker.patch("ml_flashpoint.core.checkpoint_loader.dist.get_world_size")
        mock_get_num_nodes = mocker.patch("ml_flashpoint.core.checkpoint_loader.get_num_of_nodes")
        mock_all_gather = mocker.patch("ml_flashpoint.core.checkpoint_loader.dist.all_gather_object")

        mock_get_node_local_rank.return_value = 1
        mock_get_rank.return_value = 1
        mock_get_world_size.return_value = 2
        mock_get_num_nodes.return_value = 2

        container_path = tmp_path / "ckpt"
        container_path.mkdir()

        container_id = CheckpointContainerId(str(container_path))

        # Execute
        loader.get_checkpoint_objects_by_rank(container_id)

        mock_all_gather.assert_called_once()


class TestCheckpointLoaderSync:
    @pytest.fixture
    def loader(self, mocker):
        chkpt_obj_manager = mocker.MagicMock(spec=CheckpointObjectManager)
        replication_manager = mocker.MagicMock(spec=ReplicationManager)
        return DefaultMLFlashpointCheckpointLoader(chkpt_obj_manager, replication_manager)

    def test_get_checkpoint_objects_by_rank_sync(self, mocker, loader, tmp_path):
        mock_get_node_local_rank = mocker.patch("ml_flashpoint.core.checkpoint_loader.dist.get_node_local_rank")
        mock_get_rank = mocker.patch("ml_flashpoint.core.checkpoint_loader.dist.get_rank")
        mock_get_world_size = mocker.patch("ml_flashpoint.core.checkpoint_loader.dist.get_world_size")
        mock_get_num_nodes = mocker.patch("ml_flashpoint.core.checkpoint_loader.get_num_of_nodes")
        mock_all_gather = mocker.patch("ml_flashpoint.core.checkpoint_loader.dist.all_gather_object")

        mock_get_world_size.return_value = 4
        mock_get_num_nodes.return_value = 2

        mock_get_rank.return_value = 0
        mock_get_node_local_rank.return_value = 0

        container_path = tmp_path / "ckpt"
        container_path.mkdir()
        (container_path / "file1").touch()

        def side_effect_all_gather(obj_list, local_obj):
            obj_list[0] = ["file1"]
            obj_list[1] = ["file1"]
            obj_list[2] = ["file2"]
            obj_list[3] = []

        mock_all_gather.side_effect = side_effect_all_gather

        container_id = CheckpointContainerId(str(container_path))

        # Execute
        result = loader.get_checkpoint_objects_by_rank(container_id)

        # Verify

        assert 0 in result
        assert len(result[0]) == 1
        assert str(result[0][0]).endswith("file1")

        assert 1 in result
        # This assertion is expected to FAIL currently
        assert len(result[1]) == 1, f"Rank 1 should have objects, but got {result[1]}"
        assert str(result[1][0]).endswith("file1")


class TestCheckpointLoaderContext:
    @pytest.fixture
    def loader(self):
        ckpt_manager = CheckpointObjectManager()
        repl_manager = MagicMock(spec=ReplicationManager)
        return DefaultMLFlashpointCheckpointLoader(
            checkpoint_object_manager=ckpt_manager, replication_manager=repl_manager, recover_context=True
        )

    def test_get_checkpoint_objects_by_rank_finds_context(self, loader, mocker):
        """Test that get_checkpoint_objects_by_rank finds files in context/ dir when recover_context=True."""
        mocker.patch("torch.distributed.get_world_size", return_value=1)
        mocker.patch(
            "torch.distributed.all_gather_object",
            side_effect=lambda obj_list, local_obj: obj_list.__setitem__(0, local_obj),
        )

        container_path = "/tmp/ckpt/step-1"
        container_id = CheckpointContainerId(container_path)

        # Mock fs
        # base dir has "context" directory
        # context dir has "file1.txt"
        def mock_listdir(path):
            if str(path) == container_path:
                return ["context", "other_file"]
            if str(path) == str(Path(container_path) / "context"):
                return ["file1.txt", "file2.txt"]
            return []

        def mock_isdir(path):
            if str(path) == container_path:
                return True
            if str(path) == str(Path(container_path) / "context"):
                return True
            return False

        mocker.patch("os.listdir", side_effect=mock_listdir)
        mocker.patch("pathlib.Path.is_dir", new=mock_isdir)

        result = loader.get_checkpoint_objects_by_rank(container_id)

        assert 0 in result
        objs = result[0]
        # Should contain: .../context/file1.txt, .../context/file2.txt, .../context, .../other_file
        # Note: 'context' is from top-level listdir. 'context/file*.txt' is from recursive check.

        paths = [str(o.data) for o in objs]
        expected_context_file1 = str(Path(container_path) / "context" / "file1.txt")
        expected_context_file2 = str(Path(container_path) / "context" / "file2.txt")
        assert expected_context_file1 in paths
        assert expected_context_file2 in paths

    def test_compute_retrieval_plan_includes_context_optimized(self, loader, mocker):
        """
        Test that _compute_retrieval_plan includes context files ONLY for local rank 0 on each node.
        Scenario:
          - World Size: 4
          - Nodes: 2 (Ranks 0,1 on Node 0; Ranks 2,3 on Node 1)
          - Rank 0 has context files.
          - Rank 2 needs context files (different node).
          - Rank 1, 3 do NOT need context retrieval (same node as 0, 2).
        """
        checkpoint = CheckpointContainerId("/tmp/ckpt/step-1")

        # Mock metadata read (empty storage data)
        mock_metadata = MagicMock()
        mock_metadata.storage_data = {}
        mocker.patch.object(loader, "read_metadata", return_value=mock_metadata)

        mocker.patch("torch.distributed.get_world_size", return_value=4)
        mocker.patch("ml_flashpoint.core.checkpoint_loader.get_num_of_nodes", return_value=2)
        mocker.patch("torch.distributed.get_rank", return_value=0)

        ctx_file = str(Path(checkpoint.data) / "context" / "file1.txt")
        common_pt = str(Path(checkpoint.data) / "common.pt")
        metadata_file = str(Path(checkpoint.data) / ".metadata")

        # Available objects:
        # Rank 0 has everything (Context + Common + Metadata)
        # Others have nothing
        available_objects = {
            0: [CheckpointObjectId(ctx_file), CheckpointObjectId(common_pt), CheckpointObjectId(metadata_file)],
            1: [],
            2: [],
            3: [],
        }

        # Execute
        plan = loader._compute_retrieval_plan(checkpoint, available_objects)

        assert plan is not None

        # Rank 0: Already has files, no retrieval needed (or plan[0] is empty)
        assert 0 not in plan or not plan[0]

        # Rank 1: Local rank 1 on Node 0. Shared FS with Rank 0. Should NOT retrieve context.
        # It also doesn't need common.pt/metadata (only local rank 0 needs them).
        # So plan[1] should be empty/missing.
        assert 1 not in plan or not plan[1]

        # Rank 2: Local rank 0 on Node 1. Needs Context + Common + Metadata.
        assert 2 in plan
        retrieved_objs_2 = [path for src, path in plan[2]]
        assert ctx_file in retrieved_objs_2
        assert common_pt in retrieved_objs_2
        assert metadata_file in retrieved_objs_2

        # Rank 3: Local rank 1 on Node 1. Shared FS with Rank 2. Should NOT retrieve.
        assert 3 not in plan or not plan[3]
