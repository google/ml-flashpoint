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
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from torch.distributed.checkpoint import metadata as torchdistmeta
from torch.distributed.checkpoint.planner import SavePlan, WriteItem, WriteItemType

from ml_flashpoint.adapter.megatron.save_strategies import MLFlashpointMegatronAsyncSaveStrategy
from ml_flashpoint.adapter.pytorch.memory_storage_writer import MemoryStorageWriter
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId, CheckpointObjectId
from ml_flashpoint.core.checkpoint_saver import (
    DefaultMLFlashpointCheckpointSaver,
    MLFlashpointCheckpointSaver,
    ObjectWriteBucket,
)
from ml_flashpoint.replication.replication_manager import ReplicationManager

_default_test_global_rank = 0
_default_test_local_rank = 0


@pytest.fixture
def checkpoint_saver() -> MLFlashpointCheckpointSaver:
    return DefaultMLFlashpointCheckpointSaver(
        global_rank_getter=lambda: _default_test_global_rank,
        local_rank_getter=lambda: _default_test_local_rank,
        global_barrier_func=lambda: None,
        ckpt_obj_manager=CheckpointObjectManager(),
        replication_manager=ReplicationManager(),
    )


@pytest.fixture
def storage_writer(mocker, checkpoint_saver) -> MemoryStorageWriter:
    # Using a real MemoryStorageWriter instance instead of a mock.
    # We can still spy on its methods if needed.
    # The mp_manager is mocked as it's not relevant to these tests.
    return MemoryStorageWriter(
        checkpoint_saver=checkpoint_saver,
        mp_manager=mocker.MagicMock(),
    )


class TestMLFlashpointMegatronAsyncSaveStrategy:
    def test_init(self, storage_writer, checkpoint_saver):
        # Given
        strategy = MLFlashpointMegatronAsyncSaveStrategy(storage_writer=storage_writer)

        # When/Then
        assert strategy._storage_writer is storage_writer
        assert strategy._checkpoint_saver is checkpoint_saver

    def test_can_handle_sharded_objects(self, storage_writer):
        # Given
        strategy = MLFlashpointMegatronAsyncSaveStrategy(storage_writer=storage_writer)

        # When/Then
        assert strategy.can_handle_sharded_objects() is True

    class TestAsyncSave:
        @pytest.fixture(autouse=True)
        def mock_dist(self, mocker):
            mocker.patch("torch.distributed.is_initialized", return_value=True)
            mocker.patch("torch.distributed.get_rank", return_value=_default_test_global_rank)
            mocker.patch("torch.distributed.get_node_local_rank", return_value=_default_test_local_rank)
            mocker.patch("torch.distributed.get_world_size", return_value=1)

        @pytest.fixture
        def checkpoint_id(self, tmp_path):
            return CheckpointContainerId(str(tmp_path / "test_checkpoint"))

        @pytest.fixture
        def dummy_save_plan(self):
            return SavePlan(
                [
                    WriteItem(index=torchdistmeta.MetadataIndex("tensor1"), type=WriteItemType.TENSOR),
                    WriteItem(index=torchdistmeta.MetadataIndex("tensor2"), type=WriteItemType.TENSOR),
                    WriteItem(index=torchdistmeta.MetadataIndex("dir1/shard_0_0"), type=WriteItemType.BYTE_IO),
                ]
            )

        @pytest.fixture
        def dummy_metadata(self):
            return torchdistmeta.Metadata(
                state_dict_metadata={
                    "tensor1": torchdistmeta.TensorStorageMetadata(
                        size=torch.Size([10, 20]),
                        properties=torchdistmeta.TensorProperties(dtype=torch.float32),
                        chunks=[
                            torchdistmeta.ChunkStorageMetadata(offsets=torch.Size([0, 0]), sizes=torch.Size([5, 10]))
                        ],
                    ),
                    "tensor2": torchdistmeta.TensorStorageMetadata(
                        size=torch.Size([30, 40]),
                        properties=torchdistmeta.TensorProperties(dtype=torch.float32),
                        chunks=[
                            torchdistmeta.ChunkStorageMetadata(offsets=torch.Size([0, 0]), sizes=torch.Size([5, 10]))
                        ],
                    ),
                    "dir1/shard_0_0": torchdistmeta.BytesStorageMetadata(),
                }
            )

        @pytest.fixture
        def dummy_write_buckets(self, mocker, checkpoint_id):
            return [
                ObjectWriteBucket(
                    object_id=CheckpointObjectId(f"{checkpoint_id.data}/obj_{i}"),
                    object_name=f"obj_{i}",
                    bytesio_data=[],
                    tensor_data=[(mocker.MagicMock(), torch.tensor([i]))],
                )
                for i in range(3)
            ]

        @pytest.fixture
        def async_save_setup(self, mocker, monkeypatch, storage_writer, checkpoint_id):
            monkeypatch.setenv("MLFLASHPOINT_DISABLE_DIST", "true")

            strategy = MLFlashpointMegatronAsyncSaveStrategy(storage_writer=storage_writer)

            sharded_state_dict = {
                "layer1.weight": ShardedTensor(
                    key="layer1.weight",
                    data=torch.rand(10, 5),
                    dtype=torch.float32,
                    local_shape=(10, 5),
                    global_shape=(10, 5),
                    global_offset=(0, 0),
                    axis_fragmentations=(1, 1),
                )
            }
            pyt_state_dict = {"tensor1": torch.tensor([1, 2, 3]), "non_tensor": "test_string"}

            mocker.patch(
                "ml_flashpoint.adapter.megatron.save_strategies._replace_state_dict_keys_with_sharded_keys",
                return_value=(sharded_state_dict, None, None),
            )
            mocker.patch(
                "ml_flashpoint.adapter.megatron.save_strategies.mcore_to_pyt_state_dict", return_value=pyt_state_dict
            )

            return strategy, checkpoint_id, sharded_state_dict, pyt_state_dict

        def test_async_save_initialization_calls_success(
            self, mocker, async_save_setup, storage_writer, checkpoint_saver, dummy_write_buckets
        ):
            """Tests the initialization calls within async_save, including StorageWriter re-initialization."""
            # Given
            mock_statedictsaver = mocker.patch("ml_flashpoint.adapter.megatron.save_strategies.statedictsaver")
            (
                strategy,
                checkpoint_id,
                sharded_state_dict,
                _,
            ) = async_save_setup
            mock_statedictsaver.generate_plan.return_value = (
                mocker.MagicMock(),
                dummy_write_buckets,
                mocker.MagicMock(),
            )

            mock_memory_storage_writer_cls = mocker.patch(
                "ml_flashpoint.adapter.megatron.save_strategies.MemoryStorageWriter"
            )
            mock_new_storage_writer_instance = mock_memory_storage_writer_cls.return_value

            initialize_checkpoint_spy = mocker.spy(checkpoint_saver, "initialize_checkpoint")

            # When
            strategy.async_save(sharded_state_dict, checkpoint_id.data)

            # Then
            initialize_checkpoint_spy.assert_called_once_with(checkpoint_id)

            mock_memory_storage_writer_cls.assert_called_once_with(
                checkpoint_saver=checkpoint_saver,
                mp_manager=storage_writer._mp_manager,
                thread_count=storage_writer._thread_count,
            )
            mock_new_storage_writer_instance.reset.assert_called_once_with(checkpoint_id.data)
            mock_new_storage_writer_instance.stage_write_data_buckets.assert_called_once_with(
                checkpoint_id, dummy_write_buckets, non_blocking=True
            )

        @pytest.mark.parametrize("expected_thread_count", [1, 2, 3, 5])
        def test_async_save_reinitializes_storage_writer_with_thread_count(
            self, mocker, async_save_setup, storage_writer, checkpoint_saver, dummy_write_buckets, expected_thread_count
        ):
            """Tests that the StorageWriter is re-initialized with the correct thread_count."""
            # Given
            mock_statedictsaver = mocker.patch("ml_flashpoint.adapter.megatron.save_strategies.statedictsaver")
            (
                strategy,
                checkpoint_id,
                sharded_state_dict,
                _,
            ) = async_save_setup
            mock_statedictsaver.generate_plan.return_value = (
                mocker.MagicMock(),
                dummy_write_buckets,
                mocker.MagicMock(),
            )

            # Set a specific thread_count on the original storage_writer
            storage_writer._thread_count = expected_thread_count

            mock_memory_storage_writer_cls = mocker.patch(
                "ml_flashpoint.adapter.megatron.save_strategies.MemoryStorageWriter"
            )

            # When
            strategy.async_save(sharded_state_dict, checkpoint_id.data)

            # Then
            mock_memory_storage_writer_cls.assert_called_once_with(
                checkpoint_saver=checkpoint_saver,
                mp_manager=storage_writer._mp_manager,
                thread_count=expected_thread_count,
            )

        def test_initialize_checkpoint_failure(self, mocker, async_save_setup, checkpoint_saver):
            """Tests that the process terminates gracefully if initialize_checkpoint fails."""
            # Given
            strategy, checkpoint_id, sharded_state_dict, _ = async_save_setup
            mocker.patch.object(checkpoint_saver, "initialize_checkpoint", side_effect=Exception("Init Failed"))

            # When / Then
            with pytest.raises(Exception, match="Init Failed"):
                strategy.async_save(sharded_state_dict, checkpoint_id.data)

        def test_async_save_generate_plan_call_success(self, mocker, async_save_setup, storage_writer):
            """Tests that generate_plan is called correctly within async_save."""
            # Given
            mock_statedictsaver = mocker.patch("ml_flashpoint.adapter.megatron.save_strategies.statedictsaver")
            MockMCoreSavePlanner = mocker.patch("ml_flashpoint.adapter.megatron.save_strategies.MCoreSavePlanner")
            (
                strategy,
                checkpoint_id,
                sharded_state_dict,
                pyt_state_dict,
            ) = async_save_setup
            mock_planner = MockMCoreSavePlanner.return_value
            mock_statedictsaver.generate_plan.return_value = (
                mocker.MagicMock(),
                mocker.MagicMock(),
                mocker.MagicMock(),
            )

            expected_kwarg_keys = {"checkpoint_id", "state_dict", "storage_writer", "planner", "world_dist_wrapper"}

            # When
            strategy.async_save(sharded_state_dict, checkpoint_id.data)

            # Then
            mock_statedictsaver.generate_plan.assert_called_once()
            _, kwargs = mock_statedictsaver.generate_plan.call_args
            actual_storage_writer_used = kwargs.get("storage_writer", None)
            assert set(kwargs.keys()) == expected_kwarg_keys
            assert kwargs["checkpoint_id"] == checkpoint_id
            assert kwargs["state_dict"] == pyt_state_dict
            assert actual_storage_writer_used is not None
            assert isinstance(actual_storage_writer_used, MemoryStorageWriter)
            assert actual_storage_writer_used._mp_manager is storage_writer._mp_manager
            assert kwargs["planner"] is mock_planner
            assert "world_dist_wrapper" in kwargs
            assert kwargs["world_dist_wrapper"].use_dist is False

        def test_generate_plan_failure(self, mocker, async_save_setup):
            """Tests that an exception in generate_plan is propagated."""
            # Given
            mock_statedictsaver = mocker.patch("ml_flashpoint.adapter.megatron.save_strategies.statedictsaver")
            strategy, checkpoint_id, sharded_state_dict, _ = async_save_setup
            mock_statedictsaver.generate_plan.side_effect = ValueError("Plan Failed")

            # When / Then
            with pytest.raises(ValueError, match="Plan Failed"):
                strategy.async_save(sharded_state_dict, checkpoint_id.data)

        def test_async_save_async_fn_call_success(
            self, mocker, async_save_setup, storage_writer, dummy_save_plan, dummy_metadata, dummy_write_buckets
        ):
            """Tests that the async_fn returned by async_save calls write_data."""
            # Given
            from ml_flashpoint.core.checkpoint_id_types import CheckpointObjectId
            from ml_flashpoint.core.checkpoint_saver import ObjectWriteBucket

            mock_statedictsaver = mocker.patch("ml_flashpoint.adapter.megatron.save_strategies.statedictsaver")
            strategy, checkpoint_id, sharded_state_dict, _ = async_save_setup
            mock_statedictsaver.generate_plan.return_value = (dummy_save_plan, dummy_write_buckets, dummy_metadata)
            staged_write_buckets = [
                ObjectWriteBucket(
                    object_id=CheckpointObjectId(f"/test_checkpoint/staged_obj_{i}"),
                    object_name=f"staged_obj_{i}",
                    bytesio_data=[],
                    tensor_data=[(mocker.MagicMock(), torch.tensor([i + 10]))],
                )
                for i in range(2)
            ]

            mock_memory_storage_writer_cls = mocker.patch(
                "ml_flashpoint.adapter.megatron.save_strategies.MemoryStorageWriter"
            )
            mock_new_storage_writer_instance = mock_memory_storage_writer_cls.return_value
            mock_new_storage_writer_instance.stage_write_data_buckets.return_value = staged_write_buckets

            # When
            actual_async_request = strategy.async_save(sharded_state_dict, checkpoint_id.data)
            actual_async_request.async_fn(**actual_async_request.async_fn_kwargs)

            # Then
            mock_statedictsaver.write_data.assert_called_once_with(
                checkpoint_id=checkpoint_id,
                storage_writer=mock_new_storage_writer_instance,
                staged_write_buckets=staged_write_buckets,
                replicate_after_write=False,
            )

        def test_async_save_async_fn_failure(self, mocker, async_save_setup, checkpoint_saver):
            """Tests that finalize_checkpoint is not called when async_fn fails."""
            # Given
            finalize_checkpoint_spy = mocker.spy(checkpoint_saver, "finalize_checkpoint")
            mock_statedictsaver = mocker.patch("ml_flashpoint.adapter.megatron.save_strategies.statedictsaver")
            strategy, checkpoint_id, sharded_state_dict, _ = async_save_setup
            mock_statedictsaver.generate_plan.return_value = (
                mocker.MagicMock(),
                mocker.MagicMock(),
                mocker.MagicMock(),
            )
            mock_statedictsaver.write_data.side_effect = Exception("Test Exception")

            # When
            actual_async_request = strategy.async_save(sharded_state_dict, checkpoint_id.data)
            with pytest.raises(Exception, match="Test Exception"):
                actual_async_request.async_fn(**actual_async_request.async_fn_kwargs)

            # Then
            finalize_checkpoint_spy.assert_not_called()

        def test_async_save_finalize_fns_calls(
            self,
            mocker,
            async_save_setup,
            storage_writer,
            checkpoint_saver,
            dummy_save_plan,
            dummy_metadata,
            dummy_write_buckets,
        ):
            """Tests that the finalize_fns returned by async_save call finish_write and finalize_checkpoint."""
            # Given
            finalize_checkpoint_spy = mocker.spy(checkpoint_saver, "finalize_checkpoint")
            mock_statedictsaver = mocker.patch("ml_flashpoint.adapter.megatron.save_strategies.statedictsaver")
            strategy, checkpoint_id, sharded_state_dict, _ = async_save_setup
            mock_statedictsaver.generate_plan.return_value = (dummy_save_plan, dummy_write_buckets, dummy_metadata)

            mock_memory_storage_writer_cls = mocker.patch(
                "ml_flashpoint.adapter.megatron.save_strategies.MemoryStorageWriter"
            )
            mock_storage_writer_instance = mock_memory_storage_writer_cls.return_value
            # We need to set _mp_manager on the mock because the test asserts on it later
            mock_storage_writer_instance._mp_manager = storage_writer._mp_manager
            mock_storage_writer_instance.stage_write_data_buckets.return_value = dummy_write_buckets

            expected_kwarg_keys = {"checkpoint_id", "storage_writer", "global_metadata", "world_dist_wrapper"}

            # When
            actual_async_request = strategy.async_save(sharded_state_dict, checkpoint_id.data)

            # Then
            assert len(actual_async_request.finalize_fns) == 3

            # Call 1st finalize function
            actual_async_request.finalize_fns[0]()

            # Then
            # Assert the actual storage_writer invoked replicate_written_objects with the entire set of object IDs
            expected_object_ids = {b.object_id for b in dummy_write_buckets}
            mock_storage_writer_instance.replicate_written_objects.assert_called_once_with(
                object_ids=expected_object_ids
            )

            # Call 2nd finalize function
            actual_async_request.finalize_fns[1]()

            # Then
            mock_statedictsaver.finish_write.assert_called_once()
            _, kwargs = mock_statedictsaver.finish_write.call_args
            actual_storage_writer_used = kwargs.get("storage_writer", None)
            assert set(kwargs.keys()) == expected_kwarg_keys
            assert kwargs["checkpoint_id"] == checkpoint_id
            assert actual_storage_writer_used is not None
            assert actual_storage_writer_used is mock_storage_writer_instance
            assert actual_storage_writer_used._mp_manager is storage_writer._mp_manager
            assert kwargs["global_metadata"] == dummy_metadata
            assert kwargs["world_dist_wrapper"].use_dist is False

            # Call 3rd finalize function
            actual_async_request.finalize_fns[2]()

            # Then
            finalize_checkpoint_spy.assert_called_once_with(checkpoint_id=checkpoint_id)

        def test_finalize_fns_failure(
            self, mocker, async_save_setup, checkpoint_saver, dummy_save_plan, dummy_metadata
        ):
            """Tests that a failure in the finish_write finalize_fn prevents finalize_checkpoint from running."""
            # Given
            finalize_checkpoint_spy = mocker.spy(checkpoint_saver, "finalize_checkpoint")
            mock_statedictsaver = mocker.patch("ml_flashpoint.adapter.megatron.save_strategies.statedictsaver")
            strategy, checkpoint_id, sharded_state_dict, _ = async_save_setup
            mock_statedictsaver.generate_plan.return_value = (dummy_save_plan, mocker.MagicMock(), dummy_metadata)
            mock_statedictsaver.finish_write.side_effect = ValueError("Finish Write Failed")

            # When
            actual_async_request = strategy.async_save(sharded_state_dict, checkpoint_id.data)
            with pytest.raises(ValueError, match="Finish Write Failed"):
                actual_async_request.finalize_fns[1]()

            # Then
            finalize_checkpoint_spy.assert_not_called()

        @pytest.mark.parametrize(
            "is_dist_initialized, dist_rank, expected_rank",
            [
                (True, 5, 5),
                (False, 0, -1),
            ],
        )
        def test_async_save_rank_determination(
            self,
            mocker,
            async_save_setup,
            is_dist_initialized,
            dist_rank,
            expected_rank,
        ):
            """Tests that the rank passed to async_fn is correct based on dist initialization."""
            # Given
            strategy, checkpoint_id, sharded_state_dict, _ = async_save_setup

            # Mock torch.distributed
            mocker.patch("torch.distributed.is_initialized", return_value=is_dist_initialized)
            if is_dist_initialized:
                mocker.patch("torch.distributed.get_rank", return_value=dist_rank)

            # Mock dependencies to ensure success path
            mock_statedictsaver = mocker.patch("ml_flashpoint.adapter.megatron.save_strategies.statedictsaver")
            mock_statedictsaver.generate_plan.return_value = (
                mocker.MagicMock(),
                mocker.MagicMock(),
                mocker.MagicMock(),
            )

            # When
            actual_async_request = strategy.async_save(sharded_state_dict, checkpoint_id.data)

            # Then
            assert actual_async_request.async_fn_kwargs["rank"] == expected_rank
