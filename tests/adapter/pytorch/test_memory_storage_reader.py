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

import tempfile
from pathlib import Path

import pytest
import torch
from pytest_mock import MockerFixture
from torch.distributed.checkpoint.filesystem import FileSystem, _StorageInfo
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
)

from ml_flashpoint.adapter.pytorch.memory_storage_reader import MemoryStorageReader
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId, CheckpointObjectId
from ml_flashpoint.core.checkpoint_loader import DefaultMLFlashpointCheckpointLoader
from ml_flashpoint.replication.replication_manager import ReplicationManager


class TestMemoryStorageReader:
    @pytest.fixture
    def checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_initialization_sets_fields_correctly(self):
        test_checkpoint_loader = DefaultMLFlashpointCheckpointLoader(CheckpointObjectManager(), ReplicationManager())
        reader = MemoryStorageReader("/some/test/path", checkpoint_loader=test_checkpoint_loader)

        assert reader._path == "/some/test/path"
        assert reader._checkpoint_container_id == CheckpointContainerId("/some/test/path")
        assert reader._checkpoint_loader is test_checkpoint_loader
        assert reader._storage_data is None

    def test_read_data(self, checkpoint_directory, mocker: MockerFixture):
        # Arrange
        mock_loader = mocker.MagicMock()
        reader = MemoryStorageReader(path=checkpoint_directory, checkpoint_loader=mock_loader)

        # Mock storage_data
        reader._storage_data = {
            MetadataIndex("0"): mocker.MagicMock(relative_path="file1.pt"),
            MetadataIndex("1"): mocker.MagicMock(relative_path="file2.pt"),
            MetadataIndex("2"): mocker.MagicMock(relative_path="file1.pt"),
        }

        req1 = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=MetadataIndex("0"),
            storage_offsets=(0,),
            lengths=(1,),
            dest_index=(0,),
            dest_offsets=(0,),
        )
        req2 = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=MetadataIndex("1"),
            storage_offsets=(0,),
            lengths=(1,),
            dest_index=(0,),
            dest_offsets=(0,),
        )
        req3 = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=MetadataIndex("2"),
            storage_offsets=(0,),
            lengths=(1,),
            dest_index=(0,),
            dest_offsets=(0,),
        )
        plan = LoadPlan(items=[req1, req2, req3])
        planner = mocker.MagicMock()

        # Act
        reader.read_data(plan, planner)

        # Assert
        # 2 files only
        assert mock_loader.read_data.call_count == 2

        # We need to check the calls to read_data. The order is not guaranteed because of the ThreadPoolExecutor.
        # So we check the contents of the calls.
        call_args_list = mock_loader.read_data.call_args_list

        # Call for file1.pt
        file1_path = Path(checkpoint_directory) / "file1.pt"
        file1_reqs = [req1, req3]
        call_for_file1 = next(
            (call for call in call_args_list if call[0][0].data == str(file1_path)),
            None,
        )
        assert call_for_file1 is not None
        assert call_for_file1[0][1] == file1_reqs
        assert call_for_file1[0][2] == planner
        assert call_for_file1[0][3] == reader._storage_data

        # Call for file2.pt
        file2_path = Path(checkpoint_directory) / "file2.pt"
        file2_reqs = [req2]
        call_for_file2 = next(
            (call for call in call_args_list if call[0][0].data == str(file2_path)),
            None,
        )
        assert call_for_file2 is not None
        assert call_for_file2[0][1] == file2_reqs
        assert call_for_file2[0][2] == planner
        assert call_for_file2[0][3] == reader._storage_data

    def test_read_data_empty_plan(self, checkpoint_directory, mocker: MockerFixture):
        # Arrange
        mocker.patch("torch.distributed.get_rank", return_value=0)
        mock_loader = mocker.MagicMock()
        reader = MemoryStorageReader(path=checkpoint_directory, checkpoint_loader=mock_loader)
        reader._storage_data = {
            MetadataIndex("0"): mocker.MagicMock(relative_path="file1.pt"),
        }
        plan = LoadPlan(items=[])
        planner = mocker.MagicMock()
        planner.rank = 0

        # Act
        reader.read_data(plan, planner)

        # Assert
        mock_loader.read_data.assert_not_called()

    def test_read_data_propagates_error(self, checkpoint_directory, mocker: MockerFixture):
        # Arrange
        mock_loader = mocker.MagicMock()
        # Make the mock loader raise an exception
        mock_loader.read_data.side_effect = RuntimeError("Loader failed")

        reader = MemoryStorageReader(path=checkpoint_directory, checkpoint_loader=mock_loader)

        reader._storage_data = {
            MetadataIndex("0"): mocker.MagicMock(relative_path="file1.pt"),
        }

        req1 = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=MetadataIndex("0"),
            storage_offsets=(0,),
            lengths=(1,),
            dest_index=(0,),
            dest_offsets=(0,),
        )
        plan = LoadPlan(items=[req1])
        planner = mocker.MagicMock()

        # Act & Assert
        with pytest.raises(RuntimeError, match="Loader failed"):
            reader.read_data(plan, planner)

        # Ensure the mock loader's read_data was called
        assert mock_loader.read_data.called

    def test_read_data_with_actual_loader(self, checkpoint_directory, mocker: MockerFixture):
        # Arrange
        chkpt_object_manager = CheckpointObjectManager()
        loader = DefaultMLFlashpointCheckpointLoader(chkpt_object_manager, ReplicationManager())
        reader = MemoryStorageReader(path=checkpoint_directory, checkpoint_loader=loader)

        # Create a dummy checkpoint file
        file_path = Path(checkpoint_directory) / "file1.pt"
        chkpt_obj_id = CheckpointObjectId(str(file_path))
        tensor_to_save = torch.randn(2, 3)
        buffer_size = tensor_to_save.nbytes * 100  # Create enough extra room for storing the serialized form
        with chkpt_object_manager.create_buffer(chkpt_obj_id, buffer_size=buffer_size) as buffer:
            torch.save(tensor_to_save, buffer)
            tensor_data_size = buffer.tell()

        assert tensor_data_size > 0, (
            "tensor_data_size should be positive and equal to the size of the serialized tensor"
        )
        # Mock storage_data
        reader._storage_data = {
            MetadataIndex("0"): _StorageInfo(relative_path="file1.pt", offset=0, length=tensor_data_size),
        }

        req = ReadItem(
            type=LoadItemType.TENSOR,
            storage_index=MetadataIndex("0"),
            storage_offsets=(0, 0),
            lengths=(2, 3),
            dest_index=(0,),
            dest_offsets=(0,),
        )
        plan = LoadPlan(items=[req])

        # Mock planner
        planner = mocker.MagicMock(spec=LoadPlanner)
        target_tensor = torch.empty_like(tensor_to_save)

        def resolve_tensor_side_effect(read_item):
            if read_item == req:
                return target_tensor
            raise ValueError("Unexpected read_item")

        planner.resolve_tensor.side_effect = resolve_tensor_side_effect

        # Act
        reader.read_data(plan, planner)

        # Assert
        planner.resolve_tensor.assert_called_once_with(req)
        assert torch.equal(target_tensor, tensor_to_save)
        planner.commit_tensor.assert_called_once()
        call_args = planner.commit_tensor.call_args.args
        assert call_args[0] == req
        assert torch.equal(call_args[1], target_tensor)

    def test_reset(self, checkpoint_directory, mocker: MockerFixture):
        # Arrange
        initial_path = "/initial/path"
        loader = DefaultMLFlashpointCheckpointLoader(CheckpointObjectManager(), ReplicationManager())
        reader = MemoryStorageReader(path=initial_path, checkpoint_loader=loader)
        new_checkpoint_id = "/new/path"
        mock_generate_hfid = mocker.patch(
            "ml_flashpoint.adapter.pytorch.memory_storage_reader.generate_hfid",
            return_value="new_load_id",
        )

        # Act
        reader.reset(new_checkpoint_id)

        # Assert
        assert reader._path == new_checkpoint_id
        assert reader._checkpoint_container_id == CheckpointContainerId(new_checkpoint_id)
        mock_generate_hfid.assert_called_once_with("memreaderload")
        assert reader._load_id == "new_load_id"

    def test_reset_with_path_object(self, mocker: MockerFixture):
        """Tests that reset handles Path objects correctly."""
        # Given
        mock_loader = mocker.MagicMock()
        reader = MemoryStorageReader(path="/dummy", checkpoint_loader=mock_loader)
        new_path = Path("/new_path")
        mock_checkpoint_container_id = mocker.patch(
            "ml_flashpoint.adapter.pytorch.memory_storage_reader.CheckpointContainerId"
        )

        # When
        reader.reset(new_path)

        # Then
        assert reader._path == new_path
        mock_checkpoint_container_id.assert_called_once_with(str(new_path))

    def test_validate_checkpoint_id(self, mocker: MockerFixture):
        # Arrange
        mock_validate = mocker.patch.object(FileSystem, "validate_checkpoint_id")
        checkpoint_id = "/test/path"

        # Act
        MemoryStorageReader.validate_checkpoint_id(checkpoint_id)

        # Assert
        mock_validate.assert_called_once_with(checkpoint_id)

    def test_read_metadata(self, checkpoint_directory, mocker: MockerFixture):
        # Arrange
        mock_loader = mocker.MagicMock()
        reader = MemoryStorageReader(path=checkpoint_directory, checkpoint_loader=mock_loader)
        reader._load_id = "test_load_id"
        expected_metadata = Metadata(
            state_dict_metadata={
                "tensor1": TensorStorageMetadata(
                    size=torch.Size([10, 20]),
                    properties=TensorProperties(dtype=torch.float32),
                    chunks=[ChunkStorageMetadata(offsets=torch.Size([0, 0]), sizes=torch.Size([5, 10]))],
                ),
                "tensor2": TensorStorageMetadata(
                    size=torch.Size([30, 40]),
                    properties=TensorProperties(dtype=torch.float32),
                    chunks=[ChunkStorageMetadata(offsets=torch.Size([0, 0]), sizes=torch.Size([5, 10]))],
                ),
                "dir1/shard_0_0": BytesStorageMetadata(),
            },
            storage_meta=StorageMeta(),
        )
        mock_loader.read_metadata.return_value = expected_metadata

        # Act
        metadata = reader.read_metadata()

        # Assert
        mock_loader.read_metadata.assert_called_once_with(CheckpointContainerId(checkpoint_directory))
        assert metadata == expected_metadata
        assert metadata.storage_meta.load_id == "test_load_id"

    def test_set_up_storage_reader_success(self, checkpoint_directory, mocker: MockerFixture):
        # Arrange
        loader = DefaultMLFlashpointCheckpointLoader(CheckpointObjectManager(), ReplicationManager())
        reader = MemoryStorageReader(path=checkpoint_directory, checkpoint_loader=loader)
        storage_data = {
            MetadataIndex("0"): _StorageInfo(relative_path="file1.pt", offset=0, length=100),
        }
        metadata = Metadata(state_dict_metadata={}, storage_data=storage_data)

        # Act
        reader.set_up_storage_reader(metadata, is_coordinator=True)

        # Assert
        assert reader._storage_data == storage_data

    def test_set_up_storage_reader_no_storage_data(self, checkpoint_directory, mocker: MockerFixture):
        # Arrange
        loader = DefaultMLFlashpointCheckpointLoader(CheckpointObjectManager(), ReplicationManager())
        reader = MemoryStorageReader(path=checkpoint_directory, checkpoint_loader=loader)
        metadata = Metadata(state_dict_metadata={}, storage_data=None)

        # Act & Assert
        with pytest.raises(ValueError, match="metadata.storage_data cannot be None."):
            reader.set_up_storage_reader(metadata, is_coordinator=True)
