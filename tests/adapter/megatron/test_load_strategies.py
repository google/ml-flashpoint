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
import logging
import pickle
import tempfile
from pathlib import Path

import pytest
import torch
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.dist_checkpointing.strategies.torch import MCoreLoadPlanner
from torch.distributed._shard.sharded_tensor import ShardedTensor as TorchShardedTensor
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    TensorProperties,
    TensorStorageMetadata,
)

from ml_flashpoint.adapter.megatron.load_strategies import (
    MLFlashpointMegatronLoadStrategy,
)
from ml_flashpoint.adapter.pytorch.memory_storage_reader import MemoryStorageReader
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.core.checkpoint_loader import (
    DefaultMLFlashpointCheckpointLoader,
    MLFlashpointCheckpointLoader,
)
from ml_flashpoint.replication.replication_manager import ReplicationManager

_LOGGER = logging.getLogger(__name__)


class MockCheckpointLoader(MLFlashpointCheckpointLoader):
    def __init__(self, metadata):
        self.metadata = metadata

    def read_metadata(self, checkpoint_dir):
        return self.metadata

    def read_data(self, checkpoint_id, rank):
        pass

    def get_latest_complete_checkpoint(self, checkpoint_dir):
        pass

    def test_init_with_default_impl_args(mocker):
        mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
        mock_checkpoint_loader = mocker.MagicMock(spec=MLFlashpointCheckpointLoader)
        strategy = MLFlashpointMegatronLoadStrategy(
            replication_manager=mock_replication_manager, checkpoint_loader=mock_checkpoint_loader
        )
        assert strategy._replication_manager is mock_replication_manager
        assert strategy.checkpoint_loader is mock_checkpoint_loader


def test_load_tensors_metadata(mocker):
    metadata = Metadata(
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
    )

    expected_tensors_metadata = {
        "tensor1": ShardedTensor(
            key="tensor1",
            data=None,
            dtype=torch.float32,
            local_shape=(10, 20),
            global_shape=(10, 20),
            global_offset=(0, 0),
            axis_fragmentations=(1, 1),
            replica_id=0,
            prepend_axis_num=0,
            allow_shape_mismatch=False,
            flattened_range=None,
        ),
        "tensor2": ShardedTensor(
            key="tensor2",
            data=None,
            dtype=torch.float32,
            local_shape=(30, 40),
            global_shape=(30, 40),
            global_offset=(0, 0),
            axis_fragmentations=(1, 1),
            replica_id=0,
            prepend_axis_num=0,
            allow_shape_mismatch=False,
            flattened_range=None,
        ),
    }

    _LOGGER.info("metadata: %s", metadata)
    loader = MockCheckpointLoader(metadata)
    mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
    strategy = MLFlashpointMegatronLoadStrategy(checkpoint_loader=loader, replication_manager=mock_replication_manager)
    actual_tensors_metadata = strategy.load_tensors_metadata("/dummy_dir")
    _LOGGER.info("actual_tensors_metadata: %s", actual_tensors_metadata)
    assert isinstance(actual_tensors_metadata, dict)
    assert len(actual_tensors_metadata) == 2
    assert actual_tensors_metadata == expected_tensors_metadata


def test_load_sharded_metadata(mocker):
    metadata = Metadata(
        state_dict_metadata={
            "tensor1": TensorStorageMetadata(
                size=torch.Size([10, 20]),
                properties=TensorProperties(dtype=torch.float32),
                chunks=[ChunkStorageMetadata(offsets=torch.Size([0, 0]), sizes=torch.Size([5, 10]))],
            ),
            "dir1/shard_0_0": BytesStorageMetadata(),
        },
    )
    expected_sharded_metadata = {
        "dir1/shard_0_0": ShardedObject(key="dir1", data=None, global_shape=(0,), global_offset=(0,), replica_id=0),
        "tensor1": ShardedTensor(
            key="tensor1",
            data=None,
            dtype=torch.float32,
            local_shape=(10, 20),
            global_shape=(10, 20),
            global_offset=(0, 0),
            axis_fragmentations=(1, 1),
            replica_id=0,
            prepend_axis_num=0,
            allow_shape_mismatch=False,
            flattened_range=None,
        ),
    }
    _LOGGER.info("metadata: %s", metadata)
    loader = MockCheckpointLoader(metadata)
    mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
    strategy = MLFlashpointMegatronLoadStrategy(checkpoint_loader=loader, replication_manager=mock_replication_manager)
    actual_sharded_metadata = strategy.load_sharded_metadata("/dummy_dir")

    _LOGGER.info("sharded_metadata: %s", actual_sharded_metadata)

    assert isinstance(actual_sharded_metadata, dict)
    assert len(actual_sharded_metadata) == 2
    assert actual_sharded_metadata == expected_sharded_metadata


def test_invalid_sharded_metadata(mocker):
    metadata = Metadata(
        state_dict_metadata={
            "tensor1": TensorStorageMetadata(
                size=torch.Size([10, 20]),
                properties=TensorProperties(dtype=torch.float32),
                chunks=[ChunkStorageMetadata(offsets=torch.Size([0, 0]), sizes=torch.Size([5, 10]))],
            ),
            # invalid key
            "shard_0_0": BytesStorageMetadata(),
        },
    )
    _LOGGER.info("metadata: %s", metadata)
    loader = MockCheckpointLoader(metadata)
    mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
    strategy = MLFlashpointMegatronLoadStrategy(checkpoint_loader=loader, replication_manager=mock_replication_manager)
    with pytest.raises(ValueError):
        strategy.load_sharded_metadata("/.metadata")


@pytest.fixture
def checkpoint_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_load_metadata_with_default_loader(checkpoint_directory, mocker):
    metadata = Metadata(
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
            "dir1/shard_0_1": BytesStorageMetadata(),
        },
    )

    expected_tensors_metadata = {
        "tensor1": ShardedTensor(
            key="tensor1",
            data=None,
            dtype=torch.float32,
            local_shape=(10, 20),
            global_shape=(10, 20),
            global_offset=(0, 0),
            axis_fragmentations=(1, 1),
            replica_id=0,
            prepend_axis_num=0,
            allow_shape_mismatch=False,
            flattened_range=None,
        ),
        "tensor2": ShardedTensor(
            key="tensor2",
            data=None,
            dtype=torch.float32,
            local_shape=(30, 40),
            global_shape=(30, 40),
            global_offset=(0, 0),
            axis_fragmentations=(1, 1),
            replica_id=0,
            prepend_axis_num=0,
            allow_shape_mismatch=False,
            flattened_range=None,
        ),
    }
    expected_sharded_metadata = {
        "dir1/shard_0_0": ShardedObject(key="dir1", data=None, global_shape=(0,), global_offset=(0,), replica_id=0),
        "dir1/shard_0_1": ShardedObject(key="dir1", data=None, global_shape=(1,), global_offset=(0,), replica_id=0),
        "tensor1": ShardedTensor(
            key="tensor1",
            data=None,
            dtype=torch.float32,
            local_shape=(10, 20),
            global_shape=(10, 20),
            global_offset=(0, 0),
            axis_fragmentations=(1, 1),
            replica_id=0,
            prepend_axis_num=0,
            allow_shape_mismatch=False,
            flattened_range=None,
        ),
        "tensor2": ShardedTensor(
            key="tensor2",
            data=None,
            dtype=torch.float32,
            local_shape=(30, 40),
            global_shape=(30, 40),
            global_offset=(0, 0),
            axis_fragmentations=(1, 1),
            replica_id=0,
            prepend_axis_num=0,
            allow_shape_mismatch=False,
            flattened_range=None,
        ),
    }
    _LOGGER.info("metadata: %s", metadata)
    metadata_path = Path(checkpoint_directory) / ".metadata"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
    loader = DefaultMLFlashpointCheckpointLoader(
        checkpoint_object_manager=CheckpointObjectManager(),
        replication_manager=mock_replication_manager,
    )
    strategy = MLFlashpointMegatronLoadStrategy(checkpoint_loader=loader, replication_manager=mock_replication_manager)

    # load tensors metadata
    actual_tensors_metadata = strategy.load_tensors_metadata(checkpoint_directory)

    _LOGGER.info("actual_tensors_metadata: %s", actual_tensors_metadata)

    assert isinstance(actual_tensors_metadata, dict)
    assert len(actual_tensors_metadata) == 2
    assert "tensor1" in actual_tensors_metadata
    assert "tensor2" in actual_tensors_metadata
    assert actual_tensors_metadata == expected_tensors_metadata

    # load sharded metadata
    actual_sharded_metadata = strategy.load_sharded_metadata(checkpoint_directory)

    _LOGGER.info("actual_sharded_metadata: %s", actual_sharded_metadata)

    assert isinstance(actual_sharded_metadata, dict)
    assert len(actual_sharded_metadata) == 4
    assert actual_sharded_metadata == expected_sharded_metadata


def test_load_tensors_metadata_with_none_metadata(mocker):
    loader = MockCheckpointLoader(None)
    mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
    strategy = MLFlashpointMegatronLoadStrategy(checkpoint_loader=loader, replication_manager=mock_replication_manager)
    with pytest.raises(RuntimeError, match="Failed to load valid metadata"):
        strategy.load_tensors_metadata("/dummy_dir")


def test_load_tensors_metadata_with_missing_state_dict_metadata(mocker):
    metadata = object()
    loader = MockCheckpointLoader(metadata)
    mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
    strategy = MLFlashpointMegatronLoadStrategy(checkpoint_loader=loader, replication_manager=mock_replication_manager)
    with pytest.raises(RuntimeError, match="Failed to load valid metadata"):
        strategy.load_tensors_metadata("/dummy_dir")


@pytest.mark.parametrize("global_rank", [0, 1, 2])
def test_load_with_sharded_tensor_and_object(mocker, global_rank):
    """Tests that the `load` method correctly processes a sharded state dict
    containing both a ShardedTensor and a ShardedObject, returning them as a
    standard torch.Tensor and a list containing io.BytesIO, respectively."""
    # Given
    _setup_load_mocks(mocker, global_rank)

    metadata = Metadata(
        state_dict_metadata={
            "tensor1": TensorStorageMetadata(
                size=torch.Size([10, 20]),
                properties=TensorProperties(dtype=torch.float32),
                chunks=[ChunkStorageMetadata(offsets=torch.Size([0, 0]), sizes=torch.Size([5, 10]))],
            ),
            "obj1": BytesStorageMetadata(),
        },
    )
    mock_loader = MockCheckpointLoader(metadata)
    mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
    strategy = MLFlashpointMegatronLoadStrategy(
        checkpoint_loader=mock_loader, replication_manager=mock_replication_manager
    )
    sharded_state_dict = {
        "tensor1": ShardedTensor.from_rank_offsets(
            key="tensor1",
            data=torch.empty(10, 20, device="meta", dtype=torch.float32),
            replica_id=0,
        ).without_data(),
        "obj1": ShardedObject.empty_from_unique_key("obj1/shard_0_0"),
    }

    # When
    mlf_state_dict = strategy.load(sharded_state_dict, "/dummy_dir")

    # Then
    assert "tensor1" in mlf_state_dict
    assert "obj1" in mlf_state_dict
    assert isinstance(mlf_state_dict["tensor1"], list)
    assert isinstance(mlf_state_dict["tensor1"][0], torch.Tensor)
    assert isinstance(mlf_state_dict["obj1"], list)
    assert isinstance(mlf_state_dict["obj1"][0], io.BytesIO)


@pytest.mark.parametrize("global_rank", [0, 1, 2])
def test_load_calls_torch_dist_checkpoint_with_correct_planner_and_reader(mocker, global_rank):
    # Given
    load_patched_fn = _setup_load_mocks(mocker, global_rank)
    metadata = _get_test_torchdist_metadata()
    mock_loader = MockCheckpointLoader(metadata)
    mock_replication_manager = mocker.MagicMock(spec=ReplicationManager)
    strategy = MLFlashpointMegatronLoadStrategy(
        checkpoint_loader=mock_loader, replication_manager=mock_replication_manager
    )
    sharded_state_dict = {
        "tensor1": ShardedTensor.from_rank_offsets(
            key="tensor1",
            data=torch.empty(10, 20, device="meta", dtype=torch.float32),
            replica_id=0,
        ).without_data(),
        "obj1": ShardedObject.empty_from_unique_key("obj1/shard_0_0"),
    }

    # When
    strategy.load(sharded_state_dict, "/dummy_dir")

    # Then
    load_call_args = load_patched_fn.call_args
    assert isinstance(load_call_args[1]["planner"], MCoreLoadPlanner)
    storage_reader = load_call_args[1]["storage_reader"]
    assert isinstance(storage_reader, MemoryStorageReader)
    assert storage_reader._checkpoint_container_id == "/dummy_dir"
    assert storage_reader._checkpoint_loader is mock_loader


def _get_test_torchdist_metadata():
    metadata = Metadata(
        state_dict_metadata={
            "tensor1": TensorStorageMetadata(
                size=torch.Size([10, 20]),
                properties=TensorProperties(dtype=torch.float32),
                chunks=[ChunkStorageMetadata(offsets=torch.Size([0, 0]), sizes=torch.Size([5, 10]))],
            ),
            "obj1": BytesStorageMetadata(),
        },
    )
    return metadata


def _setup_load_mocks(mocker, global_rank):
    mocker.patch("torch.distributed.get_rank", return_value=global_rank)
    mocker.patch("torch.distributed.get_world_size", return_value=3)  # Mock world size for mcore_to_pyt_state_dict
    load_patched_fn = mocker.patch(
        "ml_flashpoint.adapter.megatron.load_strategies.torch_dist_checkpoint.load",
        return_value=None,
    )
    mocker.patch(
        "ml_flashpoint.adapter.megatron.load_strategies._replace_state_dict_keys_with_sharded_keys",
        side_effect=lambda x: (x, {}, {"tensor1": ["tensor1"], "obj1": ["obj1"]}),
    )
    mocker.patch(
        "ml_flashpoint.adapter.megatron.load_strategies._replace_sharded_keys_with_state_dict_keys",
        side_effect=lambda x, y, z: x,  # Return the dict as is
    )
    mocker.patch("ml_flashpoint.adapter.megatron.load_strategies._restore_dict_types")
    mock_pyt_state_dict = {
        "tensor1": mocker.MagicMock(spec=TorchShardedTensor, mcore_sh_ten=mocker.MagicMock(spec=ShardedTensor)),
        "obj1": [io.BytesIO(b"some data")],
    }
    mocker.patch(
        "ml_flashpoint.adapter.megatron.load_strategies.mcore_to_pyt_state_dict",
        return_value=mock_pyt_state_dict,
    )
    mocker.patch(
        "ml_flashpoint.adapter.megatron.load_strategies._unwrap_pyt_sharded_tensor",
        return_value=[torch.empty(10, 20, dtype=torch.float32)],
    )
    return load_patched_fn
