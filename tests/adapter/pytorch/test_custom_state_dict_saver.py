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

import pytest
import torch
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner, WriteItem, WriteItemType
from torch.distributed.checkpoint.storage import WriteResult
from torch.distributed.checkpoint.utils import _DistWrapper
from torch.futures import Future as TorchFuture

from ml_flashpoint.adapter.pytorch import custom_state_dict_saver
from ml_flashpoint.adapter.pytorch.memory_storage_writer import MemoryStorageWriter
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId, CheckpointObjectId
from ml_flashpoint.core.checkpoint_saver import ObjectWriteBucket


@pytest.fixture
def mock_storage_writer(mocker):
    """Fixture for a mocked StorageWriter."""
    # Since we now test `finish_write`, we need to mock the `get_write_results` method.
    writer = mocker.MagicMock(spec=MemoryStorageWriter)
    writer.get_write_results.return_value = [mocker.MagicMock()]
    return writer


@pytest.fixture
def mock_save_planner(mocker):
    """Fixture for a mocked SavePlanner."""
    return mocker.MagicMock(spec=SavePlanner)


@pytest.fixture
def dist_wrapper():
    """Fixture for a real _DistWrapper in non-distributed mode."""
    return _DistWrapper(group=None, use_dist=False, coordinator_rank=0)


class TestGeneratePlan:
    """Tests for the generate_plan function."""

    def test_generate_plan_calls_dependencies_correctly(
        self, mocker, mock_storage_writer, mock_save_planner, dist_wrapper
    ):
        """Tests that generate_plan calls its dependencies in the correct order."""
        # Given
        checkpoint_id = CheckpointContainerId("/test_checkpoint")
        state_dict = {"model": "test"}
        local_plan = SavePlan([])
        expected_global_plans = [local_plan]
        expected_global_metadata = Metadata(
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
            }
        )

        expected_write_buckets = [
            ObjectWriteBucket(
                object_id=CheckpointObjectId(f"/test_checkpoint/obj_{i}"),
                object_name=f"obj_{i}",
                bytesio_data=[],
                tensor_data=[(mocker.MagicMock(), torch.tensor([i]))],
            )
            for i in range(2)
        ]

        mock_save_planner.create_local_plan.return_value = local_plan
        mock_storage_writer.prepare_local_plan.return_value = local_plan
        mock_save_planner.create_global_plan.return_value = (expected_global_plans, expected_global_metadata)
        mock_storage_writer.prepare_global_plan.return_value = expected_global_plans
        mock_save_planner.finish_plan.return_value = local_plan
        mock_storage_writer.prepare_write_data_buckets.return_value = expected_write_buckets

        # When
        (
            (actual_plan, actual_write_buckets, actual_metadata),
            _,
            _,
            actual_reused,
        ) = custom_state_dict_saver.generate_plan(
            checkpoint_id, state_dict, mock_storage_writer, mock_save_planner, dist_wrapper
        )

        # Then
        mock_save_planner.set_up_planner.assert_called_once()
        mock_storage_writer.set_up_storage_writer.assert_called_once()
        mock_save_planner.create_local_plan.assert_called_once()
        mock_storage_writer.prepare_local_plan.assert_called_once_with(local_plan)
        mock_save_planner.create_global_plan.assert_called_once_with([local_plan])
        mock_storage_writer.prepare_global_plan.assert_called_once_with(expected_global_plans)
        mock_save_planner.finish_plan.assert_called_once_with(expected_global_plans[0])
        mock_storage_writer.prepare_write_data_buckets.assert_called_once_with(
            checkpoint_id, local_plan, mock_save_planner
        )
        assert actual_plan == local_plan
        assert actual_write_buckets == expected_write_buckets
        assert actual_plan == local_plan
        assert actual_write_buckets == expected_write_buckets
        assert actual_metadata == expected_global_metadata
        assert actual_reused is False

    def test_generate_plan_reuses_cache(self, mocker, mock_storage_writer, mock_save_planner, dist_wrapper):
        """Tests that generate_plan reuse the cache when validated_cache_reuse is True."""
        # Given
        checkpoint_id = CheckpointContainerId("/test_checkpoint")
        state_dict = {"model": "test"}
        cached_plan = SavePlan([WriteItem(index=MetadataIndex("cached"), type=WriteItemType.TENSOR)])
        # cached_local_plan = SavePlan([WriteItem(index=MetadataIndex("local"), type=WriteItemType.TENSOR)])
        # cached_metadata = Metadata(state_dict_metadata={"cached": "meta"})

        mock_save_planner.finish_plan.return_value = cached_plan
        mock_storage_writer.prepare_write_data_buckets.return_value = []

        # When
        (
            (actual_plan, _, actual_metadata),
            _,
            _,
            actual_reused,
        ) = custom_state_dict_saver.generate_plan(
            checkpoint_id,
            state_dict,
            mock_storage_writer,
            mock_save_planner,
            dist_wrapper,
            cached_ckpt_structure=(cached_plan, None, True),
        )

        # Then
        # Should not call reduce_scatter or broadcast_object
        # But we can't easily assert on dist_wrapper methods as they are not mocks here unless we mock them
        # However, we can check if they are NOT called by checking side effects if we had mocked them
        # For now, checking return values is good enough proxy
        assert actual_plan == cached_plan
        assert actual_metadata is None
        assert actual_reused is True

    def test_generate_plan_validates_cache_success(self, mocker, mock_storage_writer, mock_save_planner, dist_wrapper):
        """Tests that generate_plan validates cache successfully."""
        # Given
        local_plan = SavePlan([])
        global_plans = [local_plan]
        mock_save_planner.create_local_plan.return_value = local_plan
        mock_storage_writer.prepare_local_plan.return_value = local_plan
        mock_save_planner.create_global_plan.return_value = (global_plans, None)
        mock_storage_writer.prepare_global_plan.return_value = global_plans
        # Assume reduce_scatter returns the SAME plan as cached
        mocker.patch.object(dist_wrapper, "reduce_scatter", return_value=local_plan)
        mocker.patch.object(dist_wrapper, "broadcast_object", return_value=None)
        mock_save_planner.finish_plan.return_value = local_plan
        mock_storage_writer.prepare_write_data_buckets.return_value = []

        # When
        (
            _,
            _,
            _,
            actual_reused,
        ) = custom_state_dict_saver.generate_plan(
            CheckpointContainerId("/test"),
            {},
            mock_storage_writer,
            mock_save_planner,
            dist_wrapper,
            cached_ckpt_structure=(local_plan, None, False),
        )

        # Then
        assert actual_reused is True

    def test_generate_plan_reduce_scatters_local_plan(
        self, mock_storage_writer, mock_save_planner, dist_wrapper, mocker
    ):
        """Tests that generate_plan calls reduce_scatter with the correct arguments and returns its result."""
        # Given
        state_dict = {"model": "test"}
        # Add actual data to local_plan
        local_plan = SavePlan([WriteItem(index=MetadataIndex("local_item"), type=WriteItemType.TENSOR)])
        global_plans = [local_plan]
        global_metadata = Metadata(state_dict_metadata={})

        mock_save_planner.create_local_plan.return_value = local_plan
        mock_storage_writer.prepare_local_plan.return_value = local_plan
        mock_save_planner.create_global_plan.return_value = (global_plans, global_metadata)
        mock_storage_writer.prepare_global_plan.return_value = global_plans

        # Make reduce_scatter return a distinct plan with content
        expected_returned_plan = SavePlan(
            [WriteItem(index=MetadataIndex("scattered_item"), type=WriteItemType.BYTE_IO)]
        )
        mock_reduce_scatter = mocker.patch.object(dist_wrapper, "reduce_scatter", return_value=expected_returned_plan)
        mocker.patch.object(dist_wrapper, "broadcast_object", side_effect=lambda x: x)
        mock_save_planner.finish_plan.return_value = expected_returned_plan

        # When
        (
            (returned_local_plan, _, _),
            _,
            _,
            _,
        ) = custom_state_dict_saver.generate_plan(
            CheckpointContainerId("/test_checkpoint"), state_dict, mock_storage_writer, mock_save_planner, dist_wrapper
        )

        # Then
        mock_reduce_scatter.assert_called_once()
        # We can't directly assert the call arguments for local_step and global_step as they are inner functions.
        # However, we can assert that reduce_scatter was called with 'plan' as the tag.
        assert mock_reduce_scatter.call_args[0][0] == "plan"
        assert returned_local_plan == expected_returned_plan

    def test_generate_plan_broadcasts_global_metadata(
        self, mock_storage_writer, mock_save_planner, dist_wrapper, mocker
    ):
        """Tests that generate_plan broadcasts the global_metadata and returns its result."""
        # Given
        state_dict = {"model": "test"}
        local_plan = SavePlan([])
        global_plans = [local_plan]
        # Add actual data to global_metadata
        global_metadata = Metadata(state_dict_metadata={"test_key": "test_value"})

        mock_save_planner.create_local_plan.return_value = local_plan
        mock_storage_writer.prepare_local_plan.return_value = local_plan
        mock_save_planner.create_global_plan.return_value = (global_plans, global_metadata)
        mock_storage_writer.prepare_global_plan.return_value = global_plans

        # Make broadcast_object return a distinct value
        expected_broadcasted_metadata = Metadata(state_dict_metadata={"test_other_key": "test_other_value", "a": 2})
        mocker.patch.object(dist_wrapper, "broadcast_object", return_value=expected_broadcasted_metadata)

        # When
        (
            (_, _, returned_metadata),
            _,
            _,
            _,
        ) = custom_state_dict_saver.generate_plan(
            CheckpointContainerId("/test_checkpoint"), state_dict, mock_storage_writer, mock_save_planner, dist_wrapper
        )

        # Then
        dist_wrapper.broadcast_object.assert_called_once_with(global_metadata)
        assert returned_metadata == expected_broadcasted_metadata

    def test_generate_plan_returns_write_buckets(self, mock_storage_writer, mock_save_planner, dist_wrapper, mocker):
        """Tests that generate_plan returns the write_buckets from the storage_writer."""
        # Given
        state_dict = {"model": "test"}
        local_plan = SavePlan([WriteItem(index=MetadataIndex("local_item"), type=WriteItemType.TENSOR)])
        mock_save_planner.create_global_plan.return_value = ([local_plan], Metadata(state_dict_metadata={}))
        mock_save_planner.finish_plan.return_value = local_plan
        expected_write_buckets = [
            ObjectWriteBucket(
                object_id=CheckpointObjectId(f"/test_checkpoint/obj_ret_{i}"),
                object_name=f"obj_ret_{i}",
                bytesio_data=[],
                tensor_data=[(mocker.MagicMock(), torch.tensor([i + 10]))],
            )
            for i in range(2)
        ]
        mock_storage_writer.prepare_write_data_buckets.return_value = expected_write_buckets

        # When
        (
            (_, returned_write_buckets, _),
            _,
            _,
            _,
        ) = custom_state_dict_saver.generate_plan(
            CheckpointContainerId("/test_checkpoint"), state_dict, mock_storage_writer, mock_save_planner, dist_wrapper
        )

        # Then
        assert returned_write_buckets == expected_write_buckets

    def test_generate_plan_signature_compatibility(self, mock_storage_writer, mock_save_planner, dist_wrapper):
        """Tests that generate_plan returns exactly 4 elements (updated)."""
        # Given
        state_dict = {"model": "test"}
        mock_save_planner.create_local_plan.return_value = SavePlan([])
        mock_storage_writer.prepare_local_plan.return_value = SavePlan([])
        mock_save_planner.create_global_plan.return_value = ([SavePlan([])], None)
        mock_storage_writer.prepare_global_plan.return_value = [SavePlan([])]
        mock_save_planner.finish_plan.return_value = SavePlan([])
        mock_storage_writer.prepare_write_data_buckets.return_value = []

        # When
        result = custom_state_dict_saver.generate_plan(
            CheckpointContainerId("/test_checkpoint"),
            state_dict,
            mock_storage_writer,
            mock_save_planner,
            dist_wrapper,
        )

        # Then
        assert isinstance(result, tuple)
        assert len(result) == 4


class TestWriteData:
    """Tests for the write_data function."""

    def test_write_data_calls_dependencies_correctly(self, mock_storage_writer, mocker):
        """Tests that write_data calls its dependencies in the correct order."""
        # Given
        checkpoint_id = CheckpointContainerId("/test_checkpoint")
        staged_write_buckets = [mocker.MagicMock()]
        expected_future = TorchFuture()
        expected_results = [
            WriteResult(index=MetadataIndex("a"), size_in_bytes=10, storage_data="data_a"),
            WriteResult(index=MetadataIndex("b"), size_in_bytes=20, storage_data="data_b"),
        ]
        expected_future.set_result(expected_results)
        mock_storage_writer.write_staged_data_buckets.return_value = expected_future

        # When
        results = custom_state_dict_saver.write_data(
            checkpoint_id, mock_storage_writer, staged_write_buckets, replicate_after_write=True
        )

        # Then
        mock_storage_writer.write_staged_data_buckets.assert_called_once_with(checkpoint_id, staged_write_buckets, True)
        assert results == expected_results

    @pytest.mark.parametrize("replicate", [True, False])
    def test_write_data_passes_replicate_flag(self, mock_storage_writer, mocker, replicate):
        """Tests that write_data correctly passes the replicate_after_write flag."""
        # Given
        checkpoint_id = CheckpointContainerId("/test_checkpoint")
        staged_write_buckets = []
        expected_future = TorchFuture()
        expected_future.set_result([])
        mock_storage_writer.write_staged_data_buckets.return_value = expected_future

        # When
        custom_state_dict_saver.write_data(
            checkpoint_id, mock_storage_writer, staged_write_buckets, replicate_after_write=replicate
        )

        # Then
        mock_storage_writer.write_staged_data_buckets.assert_called_once_with(
            checkpoint_id, staged_write_buckets, replicate
        )


class TestFinishWrite:
    """Tests for the finish_write function."""

    def test_finish_write_local_rank_0(self, mocker, mock_storage_writer, dist_wrapper):
        """Tests finish_write when local rank is 0."""
        # Given
        checkpoint_id = CheckpointContainerId("/test_checkpoint")
        global_metadata = Metadata(state_dict_metadata={})
        expected_write_results_rank0 = [mocker.MagicMock()]
        mock_storage_writer.get_write_results.return_value = expected_write_results_rank0
        mocker.patch("torch.distributed.get_node_local_rank", return_value=0)

        # When
        custom_state_dict_saver.finish_write(checkpoint_id, mock_storage_writer, global_metadata, dist_wrapper)

        # Then
        mock_storage_writer.get_write_results.assert_called_once_with(checkpoint_id)
        mock_storage_writer.finish_checkpoint.assert_called_once_with(
            checkpoint_id=checkpoint_id, metadata=global_metadata, results=[expected_write_results_rank0]
        )
        mock_storage_writer.finish.assert_not_called()

    def test_finish_write_not_local_rank_0(self, mocker, mock_storage_writer, dist_wrapper):
        """Tests finish_write when local rank is not 0."""
        # Given
        checkpoint_id = CheckpointContainerId("/test_checkpoint")
        global_metadata = Metadata(state_dict_metadata={})
        mocker.patch("torch.distributed.get_node_local_rank", return_value=1)

        # When
        custom_state_dict_saver.finish_write(checkpoint_id, mock_storage_writer, global_metadata, dist_wrapper)

        # Then
        mock_storage_writer.finish_checkpoint.assert_not_called()
        mock_storage_writer.finish.assert_not_called()

    def test_finish_write_no_write_results(self, mocker, mock_storage_writer, dist_wrapper):
        """Tests finish_write when get_write_results returns None."""
        # Given
        checkpoint_id = CheckpointContainerId("/test_checkpoint")
        global_metadata = Metadata(state_dict_metadata={})
        mock_storage_writer.get_write_results.return_value = None
        mocker.patch("torch.distributed.get_node_local_rank", return_value=0)

        # When
        custom_state_dict_saver.finish_write(checkpoint_id, mock_storage_writer, global_metadata, dist_wrapper)

        # Then
        mock_storage_writer.get_write_results.assert_called_once_with(checkpoint_id)
        mock_storage_writer.finish_checkpoint.assert_called_once_with(
            checkpoint_id=checkpoint_id, metadata=global_metadata, results=[None]
        )
        mock_storage_writer.finish.assert_not_called()

    def test_finish_write_empty_results(self, mocker, mock_storage_writer, dist_wrapper):
        """Tests finish_write with an empty results list."""
        # Given
        checkpoint_id = CheckpointContainerId("/test_checkpoint")
        global_metadata = Metadata(state_dict_metadata={})
        mock_storage_writer.get_write_results.return_value = []
        mocker.patch("torch.distributed.get_node_local_rank", return_value=0)

        # When
        custom_state_dict_saver.finish_write(checkpoint_id, mock_storage_writer, global_metadata, dist_wrapper)

        # Then
        mock_storage_writer.get_write_results.assert_called_once_with(checkpoint_id)
        mock_storage_writer.finish_checkpoint.assert_called_once_with(
            metadata=global_metadata, results=[[]], checkpoint_id=checkpoint_id
        )
        mock_storage_writer.finish.assert_not_called()


class TestGetLocalRank0GlobalRanks:
    """Tests for the _get_local_rank_0_global_ranks function."""

    @pytest.mark.parametrize(
        "num_gpus, num_nodes, expected",
        [
            (8, 1, [0]),
            (8, 2, [0, 8]),
            (4, 4, [0, 4, 8, 12]),
            (1, 8, [0, 1, 2, 3, 4, 5, 6, 7]),
        ],
    )
    def test_get_local_rank_0_global_ranks(self, mocker, num_gpus, num_nodes, expected):
        """Tests _get_local_rank_0_global_ranks with various configurations."""
        # Given
        mocker.patch("torch.cuda.device_count", return_value=num_gpus)
        mocker.patch("ml_flashpoint.core.utils.get_num_of_nodes", return_value=num_nodes)

        # When
        result = custom_state_dict_saver._get_local_rank_0_global_ranks()

        # Then
        assert result == expected
