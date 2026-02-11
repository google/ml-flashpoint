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

import concurrent.futures
import dataclasses
import re

import pytest
import torch
from torch import multiprocessing as torch_mp
from torch.distributed.checkpoint import Metadata, SavePlan, WriteItem
from torch.distributed.checkpoint import metadata as torchdistmeta
from torch.distributed.checkpoint.planner import WriteItemType
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future as TorchFuture

from ml_flashpoint.adapter.pytorch.memory_storage_writer import MemoryStorageWriter, _StorageDataContext
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId, CheckpointObjectId
from ml_flashpoint.core.checkpoint_saver import MLFlashpointCheckpointSaver, ObjectWriteBucket

_EXPECTED_RESET_ERROR_MSG = re.escape("MemoryStorageWriter has not been reset. Call reset() before using this method.")


class DummySaver:
    pass


class TestMemoryStorageWriter:
    @staticmethod
    def _create_metadata(state_dict_metadata=None):
        return Metadata(state_dict_metadata=state_dict_metadata or {})

    @pytest.fixture(scope="class")
    # Takes 0.01s to create and to shut down the Manager; class scope reduces overall test time by ~1s.
    def mp_manager(self):
        mp_manager = torch_mp.Manager()
        yield mp_manager
        mp_manager.shutdown()

    def test_init(self, mocker, mp_manager):
        """Tests that the __init__ method sets the _checkpoint_saver attribute correctly."""
        # Given
        mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
        # When
        writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
        # Then
        assert writer._checkpoint_saver is mock_saver
        assert writer._mp_manager is mp_manager
        assert type(writer._write_events_per_checkpoint_id).__name__ == "DictProxy"
        assert len(writer._write_events_per_checkpoint_id) == 0
        assert writer._thread_count == 1

    @pytest.mark.parametrize(
        "thread_count, expected_thread_count",
        [
            (5, 5),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-10, 1),
        ],
    )
    def test_init_thread_count(self, mocker, mp_manager, thread_count, expected_thread_count):
        """Tests that the __init__ method sets the _thread_count attribute correctly."""
        # Given
        mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
        # When
        writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager, thread_count=thread_count)
        # Then
        assert writer._thread_count == expected_thread_count

    def test_validate_checkpoint_id(self):
        """Tests the validate_checkpoint_id class method."""
        # Valid cases
        assert MemoryStorageWriter.validate_checkpoint_id("/valid/path") is True
        assert MemoryStorageWriter.validate_checkpoint_id("/valid/path/with/number/123") is True

        # Invalid cases
        assert MemoryStorageWriter.validate_checkpoint_id("invalid/path") is False  # No leading slash
        assert MemoryStorageWriter.validate_checkpoint_id(123) is False  # Non-string
        assert MemoryStorageWriter.validate_checkpoint_id(None) is False  # None value

    def test_reset_valid_id(self, mocker, mp_manager):
        """Tests that the reset method initializes container_id and save_id."""
        # Given
        mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
        writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
        checkpoint_id = "/test_checkpoint"
        expected_save_id_prefix = "memwritersave"

        # When
        writer.reset(checkpoint_id)

        # Then
        assert writer._current_checkpoint_id.data == checkpoint_id
        assert isinstance(writer._current_save_id, str)
        assert writer._current_save_id.startswith(expected_save_id_prefix)
        assert len(writer._current_save_id) > len(expected_save_id_prefix)

        # Test calling reset again
        old_save_id = writer._current_save_id
        new_checkpoint_id = "/new_test_checkpoint"

        writer.reset(new_checkpoint_id)

        assert writer._current_checkpoint_id.data == new_checkpoint_id
        assert isinstance(writer._current_save_id, str)
        assert writer._current_save_id.startswith(expected_save_id_prefix)
        assert len(writer._current_save_id) > len(expected_save_id_prefix)
        assert writer._current_save_id != old_save_id

    def test_reset_invalid_id(self, mocker, mp_manager):
        """Tests that reset raises ValueError for an invalid checkpoint ID."""
        # Given
        mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
        writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
        invalid_checkpoint_id = "invalid/id"

        # When/Then
        with pytest.raises(ValueError, match="A CheckpointContainerId must begin with '/'"):
            writer.reset(invalid_checkpoint_id)

    def test_current_checkpoint_id_initial(self, mocker, mp_manager):
        """Tests that current_checkpoint_id is None initially."""
        # Given
        mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
        writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)

        # When/Then
        assert writer.current_checkpoint_id is None

    def test_current_checkpoint_id_after_reset(self, mocker, mp_manager):
        """Tests current_checkpoint_id after one call to reset."""
        # Given
        mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
        writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
        checkpoint_id = CheckpointContainerId("/test_checkpoint1")

        # When
        writer.reset(checkpoint_id.data)

        # Then
        assert isinstance(writer.current_checkpoint_id, CheckpointContainerId)
        assert writer.current_checkpoint_id == checkpoint_id

    def test_current_checkpoint_id_after_multiple_resets(self, mocker, mp_manager):
        """Tests current_checkpoint_id after multiple calls to reset."""
        # Given
        mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
        writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
        checkpoint_id1 = CheckpointContainerId("/test_checkpoint1")
        writer.reset(checkpoint_id1.data)

        # When
        checkpoint_id2 = CheckpointContainerId("/test_checkpoint2")
        writer.reset(checkpoint_id2.data)

        # Then
        assert isinstance(writer.current_checkpoint_id, CheckpointContainerId)
        assert writer.current_checkpoint_id == checkpoint_id2

    def test_path_property(self, mocker, mp_manager):
        """Tests that the path property returns the correct container_id data."""
        # Given

        mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
        writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
        checkpoint_id = "/test_checkpoint_path"

        # When
        writer.reset(checkpoint_id)

        # Then
        assert writer.path == checkpoint_id

    def test_path_before_reset(self, mocker, mp_manager):
        """Tests accessing path property before reset."""
        # Given
        mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
        writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)

        # When/Then
        assert writer.path is None

    def test_storage_meta_returns_correct_values(self, mocker, mp_manager):
        """Tests that the storage_meta method returns the correct StorageMeta object."""
        # Given
        mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
        writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
        checkpoint_id = "/test_checkpoint_meta"
        writer.reset(checkpoint_id)

        # When
        actual_storage_meta = writer.storage_meta()

        # Then
        assert actual_storage_meta == torchdistmeta.StorageMeta(
            checkpoint_id=checkpoint_id, save_id=writer._current_save_id
        )

    def test_storage_meta_before_reset_raises_error(self, mocker, mp_manager):
        """Tests calling storage_meta before reset."""
        # Given
        mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
        writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
        # When/Then
        with pytest.raises(ValueError, match=_EXPECTED_RESET_ERROR_MSG):
            writer.storage_meta()

    def test_set_up_storage_writer(self, mocker, mp_manager):
        """Tests that the set_up_storage_writer method runs without error."""
        # Given
        mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
        writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)

        # When/Then
        try:
            writer.set_up_storage_writer(is_coordinator=True)
            writer.set_up_storage_writer(is_coordinator=False)
        except Exception as e:
            pytest.fail(f"set_up_storage_writer raised an exception: {e}")

    class TestPrepareLocalPlan:
        def test_prepare_local_plan(self, mocker, mp_manager):
            """Tests that prepare_local_plan calls initialize_checkpoint and returns the plan."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = "/test_checkpoint_prepare_local"
            writer.reset(checkpoint_id)
            item1 = WriteItem(index=torchdistmeta.MetadataIndex("a"), type=WriteItemType.TENSOR)
            item2 = WriteItem(index=torchdistmeta.MetadataIndex("b"), type=WriteItemType.BYTE_IO)
            expected_plan = SavePlan(items=[item1, item2])

            # When
            actual_returned_plan = writer.prepare_local_plan(expected_plan)

            # Then
            mock_saver.initialize_checkpoint.assert_called_once_with(writer._current_checkpoint_id)
            assert actual_returned_plan is expected_plan

        def test_prepare_local_plan_exception(self, mocker, mp_manager):
            """Tests that exceptions from initialize_checkpoint propagate."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = "/test_checkpoint_prepare_local_exc"
            writer.reset(checkpoint_id)
            item1 = WriteItem(index=torchdistmeta.MetadataIndex("a"), type=WriteItemType.TENSOR)
            item2 = WriteItem(index=torchdistmeta.MetadataIndex("b"), type=WriteItemType.BYTE_IO)
            plan = SavePlan(items=[item1, item2])
            mock_saver.initialize_checkpoint.side_effect = RuntimeError("Init failed")

            # When/Then
            with pytest.raises(RuntimeError, match="Init failed"):
                writer.prepare_local_plan(plan)
            mock_saver.initialize_checkpoint.assert_called_once_with(writer._current_checkpoint_id)

        def test_prepare_local_plan_before_reset(self, mocker, mp_manager):
            """Tests calling prepare_local_plan before reset."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            item1 = WriteItem(index=torchdistmeta.MetadataIndex("a"), type=WriteItemType.TENSOR)
            item2 = WriteItem(index=torchdistmeta.MetadataIndex("b"), type=WriteItemType.BYTE_IO)
            plan = SavePlan(items=[item1, item2])
            # When/Then
            with pytest.raises(ValueError, match=_EXPECTED_RESET_ERROR_MSG):
                writer.prepare_local_plan(plan)

    class TestPrepareGlobalPlan:
        def test_prepare_global_plan(self, mocker, mp_manager):
            """Tests that prepare_global_plan adds the correct storage_data prefix to plans."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            item1 = WriteItem(index=torchdistmeta.MetadataIndex("a"), type=WriteItemType.TENSOR)
            item2 = WriteItem(index=torchdistmeta.MetadataIndex("b"), type=WriteItemType.TENSOR)
            plan1 = SavePlan(items=[item1])
            plan2 = SavePlan(items=[item2])
            plans = [plan1, plan2]

            # When
            actual_new_plans = writer.prepare_global_plan(plans)

            # Then
            assert len(actual_new_plans) == 2
            assert isinstance(actual_new_plans[0].storage_data, _StorageDataContext)
            assert actual_new_plans[0].storage_data.prefix == "__0_"
            assert isinstance(actual_new_plans[1].storage_data, _StorageDataContext)
            assert actual_new_plans[1].storage_data.prefix == "__1_"

            assert actual_new_plans[0].items == [item1]
            assert actual_new_plans[1].items == [item2]

    class TestStage:
        def test_stage_cpu_tensor(self, mocker, mp_manager):
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            state_dict = {"a": torch.tensor([1, 2, 3], device="cpu")}

            # When
            staged_dict = writer.stage(state_dict)

            # Then
            assert staged_dict["a"].device == torch.device("cpu")
            assert torch.equal(staged_dict["a"], state_dict["a"])

        def test_stage_cuda_tensor(self, mocker, mp_manager):
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            state_dict = {"a": torch.tensor([1, 2, 3], device="cuda")}

            # When
            actual_staged_dict = writer.stage(state_dict)

            # Then
            assert actual_staged_dict["a"].device == torch.device("cpu")
            assert torch.equal(actual_staged_dict["a"], state_dict["a"].cpu())

        def test_stage_mixed(self, mocker, mp_manager):
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            state_dict = {
                "a": torch.tensor([1, 2, 3], device="cpu"),
                "b": "a string",
                "c": 123,
            }
            if torch.cuda.is_available():
                state_dict["d"] = torch.tensor([4, 5, 6], device="cuda")

            # When
            actual_staged_dict = writer.stage(state_dict)

            # Then
            assert actual_staged_dict["a"].device == torch.device("cpu")
            assert torch.equal(actual_staged_dict["a"], state_dict["a"])
            assert actual_staged_dict["b"] == "a string"
            assert actual_staged_dict["c"] == 123
            if torch.cuda.is_available():
                assert actual_staged_dict["d"].device == torch.device("cpu")
                assert torch.equal(actual_staged_dict["d"], state_dict["d"].cpu())

        def test_stage_moves_all_to_cpu(self, mocker, mp_manager):
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)

            mock_empty_like = mocker.patch("torch.empty_like")

            test_tensor_cpu = torch.tensor(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device=torch.device("cpu")
            )

            mock_tensor_xla = mocker.MagicMock(spec=torch.Tensor)
            mock_tensor_xla.device = torch.device("xla:0")
            mock_tensor_xla.dtype = torch.float32

            mock_tensor_cuda = mocker.MagicMock(spec=torch.Tensor)
            mock_tensor_cuda.device = torch.device("cuda:0")
            mock_tensor_cuda.dtype = torch.float32

            # This is the new tensor that will be created on CPU
            mock_cpu_copy = mocker.MagicMock(spec=torch.Tensor)
            mock_cpu_copy.device = torch.device("cpu")
            mock_empty_like.return_value = mock_cpu_copy

            state_dict = {
                "cpu_tensor": test_tensor_cpu,
                "xla_tensor": mock_tensor_xla,
                "cuda_tensor": mock_tensor_cuda,
                "string_data": "hello",
            }

            # When
            actual_staged_dict = writer.stage(state_dict)

            # Then
            assert isinstance(actual_staged_dict, dict)
            assert len(actual_staged_dict) == len(state_dict)

            assert actual_staged_dict["string_data"] == "hello"

            # Assert all tensors in staged_dict are on CPU
            assert actual_staged_dict["cpu_tensor"].device == torch.device("cpu")
            assert actual_staged_dict["xla_tensor"].device == torch.device("cpu")
            assert actual_staged_dict["cuda_tensor"].device == torch.device("cpu")

            # Assert that the cpu_tensor is the same object
            assert torch.equal(actual_staged_dict["cpu_tensor"], test_tensor_cpu)

    class TestPrepareWriteDataBuckets:
        def test_prepare_write_data_buckets(self, mocker, mp_manager):
            """Tests that prepare_write_data_buckets initializes an event and calls the saver."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            expected_default_bucket_count = 1
            checkpoint_id = CheckpointContainerId("/test_checkpoint")
            plan = SavePlan(items=[], storage_data=_StorageDataContext(prefix="__0_"))
            planner = mocker.MagicMock()
            expected_buckets = _create_rich_object_write_buckets(checkpoint_id)
            mock_saver.prepare_write_data.return_value = expected_buckets

            # When
            actual_buckets = writer.prepare_write_data_buckets(checkpoint_id, plan, planner)

            # Then
            assert checkpoint_id in writer._write_events_per_checkpoint_id
            assert not writer._write_events_per_checkpoint_id[checkpoint_id].is_set()
            mock_saver.prepare_write_data.assert_called_once_with(
                checkpoint_id, plan.items, planner, plan.storage_data.prefix, bucket_count=expected_default_bucket_count
            )
            assert actual_buckets == expected_buckets

        @pytest.mark.parametrize("thread_count", [1, 4, 8])
        def test_prepare_write_data_buckets_with_thread_count(self, mocker, mp_manager, thread_count):
            """Tests that prepare_write_data_buckets calls the saver with the specified thread_count."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager, thread_count=thread_count)
            checkpoint_id = CheckpointContainerId("/test_checkpoint_with_thread_count")
            plan = SavePlan(items=[], storage_data=_StorageDataContext(prefix="__0_"))
            planner = mocker.MagicMock()
            expected_buckets = _create_rich_object_write_buckets(checkpoint_id)
            mock_saver.prepare_write_data.return_value = expected_buckets

            # When
            actual_buckets = writer.prepare_write_data_buckets(checkpoint_id, plan, planner)

            # Then
            mock_saver.prepare_write_data.assert_called_once_with(
                checkpoint_id, plan.items, planner, plan.storage_data.prefix, bucket_count=thread_count
            )
            assert actual_buckets == expected_buckets

    class TestStageWriteDataBuckets:
        def test_stage_write_data_buckets_cpu_tensor(self):
            """Tests that stage_write_data_buckets correctly moves CPU tensors to CPU (no-op)."""
            # Given
            checkpoint_id = CheckpointContainerId("/test_checkpoint")
            write_buckets = _create_rich_object_write_buckets(checkpoint_id)

            # When
            actual_staged_buckets = MemoryStorageWriter.stage_write_data_buckets(checkpoint_id, write_buckets)

            # Then
            assert len(actual_staged_buckets) == len(write_buckets)
            for i, actual_bucket in enumerate(actual_staged_buckets):
                original_bucket = write_buckets[i]
                assert actual_bucket.tensor_data[0][1].device == torch.device("cpu")
                assert torch.equal(actual_bucket.tensor_data[0][1], original_bucket.tensor_data[0][1])

        def test_stage_write_data_buckets_cuda_tensor(self):
            """Tests that stage_write_data_buckets correctly moves CUDA tensors to CPU."""
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")

            # Given
            checkpoint_id = CheckpointContainerId("/test_checkpoint")
            write_buckets = _create_rich_object_write_buckets(checkpoint_id)
            # Move tensors to CUDA for the test
            for bucket in write_buckets:
                bucket.tensor_data = [(item, tensor.to("cuda")) for item, tensor in bucket.tensor_data]

            # When
            actual_staged_buckets = MemoryStorageWriter.stage_write_data_buckets(checkpoint_id, write_buckets)

            # Then
            assert len(actual_staged_buckets) == len(write_buckets)
            for i, actual_bucket in enumerate(actual_staged_buckets):
                original_bucket = write_buckets[i]
                assert actual_bucket.tensor_data[0][1].device == torch.device("cpu")
                assert torch.equal(actual_bucket.tensor_data[0][1], original_bucket.tensor_data[0][1].cpu())

        def test_stage_write_data_buckets_non_blocking_synchronizes(self, mocker):
            """Tests that stage_write_data_buckets synchronizes when non_blocking is True and CUDA is available."""
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")

            # Given
            mock_cuda_synchronize = mocker.patch("torch.cuda.synchronize")
            checkpoint_id = CheckpointContainerId("/test_checkpoint")
            write_buckets = _create_rich_object_write_buckets(checkpoint_id)
            for bucket in write_buckets:
                bucket.tensor_data = [(item, tensor.to("cuda")) for item, tensor in bucket.tensor_data]

            # When
            MemoryStorageWriter.stage_write_data_buckets(checkpoint_id, write_buckets, non_blocking=True)

            # Then
            mock_cuda_synchronize.assert_called_once()

        def test_stage_write_data_buckets_non_blocking_no_cuda_no_synchronize(self, mocker):
            """Tests that stage_write_data_buckets does not synchronize when non_blocking is True
            but CUDA is not available."""
            # Given
            mocker.patch("torch.cuda.is_available", return_value=False)
            mock_cuda_synchronize = mocker.patch("torch.cuda.synchronize")
            checkpoint_id = CheckpointContainerId("/test_checkpoint")
            write_buckets = _create_rich_object_write_buckets(checkpoint_id)

            # When
            MemoryStorageWriter.stage_write_data_buckets(checkpoint_id, write_buckets, non_blocking=True)

            # Then
            mock_cuda_synchronize.assert_not_called()

    class TestWriteStagedDataBuckets:
        def test_write_staged_data_buckets(self, mocker, mp_manager):
            """Tests that write_staged_data_buckets calls checkpoint_saver.write_data and sets the event."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = CheckpointContainerId("/test_checkpoint")
            writer.reset(checkpoint_id.data)
            staged_write_buckets = _create_rich_object_write_buckets(checkpoint_id)
            expected_write_results = [
                WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            ]
            mock_saver.write_data.return_value = expected_write_results

            # Manually prepare write data buckets to initialize the event
            plan = SavePlan(items=[], storage_data=_StorageDataContext(prefix="__0_"))
            planner = mocker.MagicMock()
            writer.prepare_write_data_buckets(checkpoint_id, plan, planner)

            # When
            result_future = writer.write_staged_data_buckets(
                checkpoint_id, staged_write_buckets, replicate_after_write=False
            )

            # Then
            mock_saver.write_data.assert_called_once_with(
                checkpoint_id, write_buckets=staged_write_buckets, thread_count=1, replicate_after_write=False
            )
            assert writer._write_events_per_checkpoint_id[checkpoint_id].is_set()
            assert result_future.wait() == expected_write_results

        @pytest.mark.parametrize("thread_count", [1, 4, 8])
        def test_write_staged_data_buckets_with_explicit_thread_count(self, mocker, mp_manager, thread_count):
            """Tests that write_staged_data_buckets calls checkpoint_saver.write_data with the specified
            thread_count."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager, thread_count=thread_count)
            checkpoint_id = CheckpointContainerId("/test_checkpoint_explicit_thread_count")
            writer.reset(checkpoint_id.data)
            staged_write_buckets = _create_rich_object_write_buckets(checkpoint_id)
            expected_write_results = [
                WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            ]
            mock_saver.write_data.return_value = expected_write_results

            # Manually prepare write data buckets to initialize the event
            plan = SavePlan(items=[], storage_data=_StorageDataContext(prefix="__0_"))
            planner = mocker.MagicMock()
            writer.prepare_write_data_buckets(checkpoint_id, plan, planner)

            # When
            result_future = writer.write_staged_data_buckets(
                checkpoint_id, staged_write_buckets, replicate_after_write=False
            )

            # Then
            mock_saver.write_data.assert_called_once_with(
                checkpoint_id,
                write_buckets=staged_write_buckets,
                thread_count=thread_count,
                replicate_after_write=False,
            )
            assert writer._write_events_per_checkpoint_id[checkpoint_id].is_set()
            assert result_future.wait() == expected_write_results

        def test_write_staged_data_buckets_saver_exception(self, mocker, mp_manager):
            """Tests that exceptions from checkpoint_saver.write_data are propagated and event is NOT set."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = CheckpointContainerId("/test_checkpoint")
            writer.reset(checkpoint_id.data)
            staged_write_buckets = _create_rich_object_write_buckets(checkpoint_id)
            mock_saver.write_data.side_effect = RuntimeError("Saver write failed")

            plan = SavePlan(items=[], storage_data=_StorageDataContext(prefix="__0_"))
            planner = mocker.MagicMock()
            writer.prepare_write_data_buckets(checkpoint_id, plan, planner)

            # When/Then
            with pytest.raises(RuntimeError, match="Saver write failed"):
                writer.write_staged_data_buckets(checkpoint_id, staged_write_buckets, replicate_after_write=False)

            # Then
            mock_saver.write_data.assert_called_once_with(
                checkpoint_id, write_buckets=staged_write_buckets, thread_count=1, replicate_after_write=False
            )
            assert not writer._write_events_per_checkpoint_id[
                checkpoint_id
            ].is_set()  # Event should NOT be set on failure

        def test_write_staged_data_buckets_in_separate_process(self, mocker, mp_manager):
            """Tests that write_staged_data_buckets in a separate process correctly updates the shared results."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = CheckpointContainerId("/test_write_staged_multiprocess")
            writer.reset(checkpoint_id.data)

            staged_write_buckets = _create_rich_object_write_buckets(checkpoint_id)
            expected_write_results = [
                WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            ]
            mock_saver.write_data.return_value = expected_write_results

            # Manually prepare to initialize the event
            plan = writer.prepare_global_plan(
                [
                    SavePlan(
                        items=[
                            WriteItem(index=torchdistmeta.MetadataIndex("a"), type=WriteItemType.TENSOR),
                            WriteItem(index=torchdistmeta.MetadataIndex("b"), type=WriteItemType.BYTE_IO),
                        ]
                    )
                ]
            )[0]
            writer.prepare_write_data_buckets(checkpoint_id, plan, mocker.MagicMock())

            # When
            p = torch_mp.Process(
                target=writer.write_staged_data_buckets, args=(checkpoint_id, staged_write_buckets, False)
            )
            p.start()

            results = writer.get_write_results(checkpoint_id)
            p.join()

            # Then
            assert p.exitcode == 0
            assert results == expected_write_results

    class TestWriteData:
        """Tests for the write_data function."""

        def test_write_data_calls_dependencies_correctly(self, mocker, mp_manager):
            """Tests that write_data calls its dependencies in the correct order."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = CheckpointContainerId("/test_checkpoint_write_data")
            writer.reset(checkpoint_id.data)
            item1 = WriteItem(index=torchdistmeta.MetadataIndex("a"), type=WriteItemType.TENSOR)
            item2 = WriteItem(index=torchdistmeta.MetadataIndex("b"), type=WriteItemType.BYTE_IO)
            plan = SavePlan(items=[item1, item2], storage_data=_StorageDataContext(prefix="__0_"))
            planner = mocker.MagicMock()
            mock_write_buckets = _create_rich_object_write_buckets(checkpoint_id)

            expected_future = TorchFuture()
            expected_results = [
                WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            ]
            expected_future.set_result(expected_results)

            mocker.patch.object(writer, "prepare_write_data_buckets", return_value=mock_write_buckets)
            mocker.patch.object(writer, "write_staged_data_buckets", return_value=expected_future)

            # When
            result_future = writer.write_data(plan, planner)

            # Then
            writer.prepare_write_data_buckets.assert_called_once_with(checkpoint_id, plan, planner)
            writer.write_staged_data_buckets.assert_called_once_with(
                checkpoint_id, mock_write_buckets, replicate_after_write=True
            )
            assert result_future.wait() == expected_results

        def test_write_data_enforces_replication(self, mocker, mp_manager):
            """Tests that write_data explicitly sets replicate_after_write=True when calling
            write_staged_data_buckets."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = CheckpointContainerId("/test_checkpoint_write_data_repl")
            writer.reset(checkpoint_id.data)
            item = WriteItem(index=torchdistmeta.MetadataIndex("a"), type=WriteItemType.TENSOR)
            plan = SavePlan(items=[item], storage_data=_StorageDataContext(prefix="__0_"))
            planner = mocker.MagicMock()

            buckets = _create_rich_object_write_buckets(checkpoint_id)

            # Mock internal methods to isolate write_data logic
            mocker.patch.object(writer, "prepare_write_data_buckets", return_value=buckets)
            mock_write_staged = mocker.patch.object(writer, "write_staged_data_buckets", return_value=TorchFuture())

            # When
            writer.write_data(plan, planner)

            # Then
            mock_write_staged.assert_called_once_with(checkpoint_id, buckets, replicate_after_write=True)

        def test_write_data_missing_prefix(self, mocker, mp_manager):
            """Tests that write_data raises ValueError if storage_data is not set correctly."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = CheckpointContainerId("/test_checkpoint_write_data_fail")
            writer.reset(checkpoint_id.data)

            item = WriteItem(index=torchdistmeta.MetadataIndex("a"), type=WriteItemType.TENSOR)
            plan = SavePlan(items=[item])  # No storage_data prefix
            planner = mocker.MagicMock()

            # When/Then
            with pytest.raises(
                ValueError,
                match=re.escape("SavePlan.storage_data is not a valid _StorageDataContext or prefix is empty."),
            ):
                writer.write_data(plan, planner)
            mock_saver.write_data.assert_not_called()

        def test_write_data_empty_prefix(self, mocker, mp_manager):
            """Tests that write_data raises ValueError if storage_data.prefix is empty."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = CheckpointContainerId("/test_checkpoint_write_data_fail_empty")
            writer.reset(checkpoint_id.data)

            item = WriteItem(index=torchdistmeta.MetadataIndex("a"), type=WriteItemType.TENSOR)
            plan = SavePlan(items=[item])
            plan = dataclasses.replace(plan, storage_data=_StorageDataContext(prefix=""))  # Empty prefix
            planner = mocker.MagicMock()

            # When/Then
            with pytest.raises(
                ValueError,
                match=re.escape("SavePlan.storage_data is not a valid _StorageDataContext or prefix is empty."),
            ):
                writer.write_data(plan, planner)
            mock_saver.write_data.assert_not_called()

        def test_write_data_before_reset(self, mocker, mp_manager):
            """Tests calling write_data before reset."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)

            # When/Then
            with pytest.raises(ValueError, match=_EXPECTED_RESET_ERROR_MSG):
                writer.write_data(mocker.MagicMock(), mocker.MagicMock())

        def test_write_data_in_separate_process(self, mocker, mp_manager):
            """
            Tests that write_data in a separate process correctly updates the shared results.
            """
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = CheckpointContainerId("/test_write_data_multiprocess")
            writer.reset(checkpoint_id.data)

            # Mock the underlying saver's write_data method
            expected_results = [
                WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            ]
            mocker.patch.object(writer._checkpoint_saver, "write_data", return_value=expected_results)

            # Create a dummy plan to trigger write_data
            plan = SavePlan(items=[WriteItem(index=torchdistmeta.MetadataIndex("a"), type=WriteItemType.TENSOR)])
            plan = writer.prepare_global_plan([plan])[0]
            planner = mocker.MagicMock()

            # When
            # Start the write operation in a separate process
            write_process = torch_mp.Process(target=writer.write_data, args=(plan, planner))
            write_process.start()
            write_process.join()

            # Then
            assert write_process.exitcode == 0
            assert writer.get_write_results(checkpoint_id) == expected_results

    class TestFinish:
        def test_finish_success(self, mocker, mp_manager):
            """Tests that finish correctly populates metadata and calls write_metadata."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = "/test_checkpoint_finish"
            writer.reset(checkpoint_id)

            metadata = TestMemoryStorageWriter._create_metadata()
            wr1 = WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            wr2 = WriteResult(index=torchdistmeta.MetadataIndex("b"), size_in_bytes=20, storage_data="data_b")
            results = [[wr1], [wr2]]

            expected_storage_data = {
                torchdistmeta.MetadataIndex("a"): "data_a",
                torchdistmeta.MetadataIndex("b"): "data_b",
            }
            expected_storage_meta = writer.storage_meta()

            # When
            writer.finish(metadata, results)

            # Then
            assert metadata.storage_data == expected_storage_data
            assert metadata.storage_meta == expected_storage_meta
            mock_saver.write_metadata.assert_called_once_with(writer._current_checkpoint_id, metadata)

        def test_finish_empty_results(self, mocker, mp_manager):
            """Tests finish with an empty results list."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = "/test_checkpoint_finish_empty"
            writer.reset(checkpoint_id)

            metadata = TestMemoryStorageWriter._create_metadata()
            results = []

            expected_storage_data = {}
            expected_storage_meta = writer.storage_meta()

            # When
            writer.finish(metadata, results)

            # Then
            assert metadata.storage_data == expected_storage_data
            assert metadata.storage_meta == expected_storage_meta
            mock_saver.write_metadata.assert_called_once_with(writer._current_checkpoint_id, metadata)

        def test_finish_results_with_empty_list(self, mocker, mp_manager):
            """Tests finish with results containing empty inner lists."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = "/test_checkpoint_finish_inner_empty"
            writer.reset(checkpoint_id)

            metadata = TestMemoryStorageWriter._create_metadata()
            wr1 = WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            results = [[wr1], []]

            expected_storage_data = {torchdistmeta.MetadataIndex("a"): "data_a"}
            expected_storage_meta = writer.storage_meta()

            # When
            writer.finish(metadata, results)

            # Then
            assert metadata.storage_data == expected_storage_data
            assert metadata.storage_meta == expected_storage_meta
            mock_saver.write_metadata.assert_called_once_with(writer._current_checkpoint_id, metadata)

        def test_finish_write_metadata_exception(self, mocker, mp_manager):
            """Tests that exceptions from write_metadata propagate in finish."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = "/test_checkpoint_finish_exc"
            writer.reset(checkpoint_id)

            metadata = TestMemoryStorageWriter._create_metadata()
            results = []
            mock_saver.write_metadata.side_effect = RuntimeError("Metadata write failed")

            # When/Then
            with pytest.raises(RuntimeError, match="Metadata write failed"):
                writer.finish(metadata, results)
            mock_saver.write_metadata.assert_called_once_with(writer._current_checkpoint_id, metadata)

        def test_finish_before_reset(self, mocker, mp_manager):
            """Tests calling finish before reset."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            # When/Then
            with pytest.raises(ValueError, match=_EXPECTED_RESET_ERROR_MSG):
                writer.finish(TestMemoryStorageWriter._create_metadata(), [])

    class TestFinishCheckpoint:
        def test_finish_checkpoint_success(self, mocker, mp_manager):
            """Tests that finish_checkpoint correctly populates metadata and calls write_metadata."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = CheckpointContainerId("/test_checkpoint_finish")
            writer.reset(checkpoint_id.data)
            writer._write_results_per_checkpoint_id[checkpoint_id] = [
                WriteResult(index=torchdistmeta.MetadataIndex("dummy"), size_in_bytes=0, storage_data="")
            ]

            metadata = TestMemoryStorageWriter._create_metadata()
            wr1 = WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            wr2 = WriteResult(index=torchdistmeta.MetadataIndex("b"), size_in_bytes=20, storage_data="data_b")
            results = [[wr1], [wr2]]

            expected_storage_data = {
                torchdistmeta.MetadataIndex("a"): "data_a",
                torchdistmeta.MetadataIndex("b"): "data_b",
            }
            expected_storage_meta = writer.storage_meta()

            # When
            writer.finish_checkpoint(checkpoint_id, metadata, results)

            # Then
            assert metadata.storage_data == expected_storage_data
            assert metadata.storage_meta == expected_storage_meta
            mock_saver.write_metadata.assert_called_once_with(checkpoint_id, metadata)
            assert checkpoint_id not in writer._write_results_per_checkpoint_id

        def test_finish_checkpoint_empty_results(self, mocker, mp_manager):
            """Tests finish_checkpoint with an empty results list."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = CheckpointContainerId("/test_checkpoint_finish_empty")
            writer.reset(checkpoint_id.data)

            metadata = TestMemoryStorageWriter._create_metadata()
            results = []

            expected_storage_data = {}
            expected_storage_meta = writer.storage_meta()

            # When
            writer.finish_checkpoint(checkpoint_id, metadata, results)

            # Then
            assert metadata.storage_data == expected_storage_data
            assert metadata.storage_meta == expected_storage_meta
            mock_saver.write_metadata.assert_called_once_with(checkpoint_id, metadata)
            assert checkpoint_id not in writer._write_results_per_checkpoint_id

        def test_finish_checkpoint_results_with_none_list(self, mocker, mp_manager):
            """Tests finish_checkpoint with results containing None inner lists, expecting a RuntimeError."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = CheckpointContainerId("/test_checkpoint_finish_inner_none")
            writer.reset(checkpoint_id.data)
            writer._write_results_per_checkpoint_id[checkpoint_id] = []

            metadata = TestMemoryStorageWriter._create_metadata()
            wr1 = WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            results = [[wr1], None, []]

            # When/Then
            with pytest.raises(RuntimeError, match=re.escape("finish: write results[1] is None!")):
                writer.finish_checkpoint(checkpoint_id, metadata, results)
            mock_saver.write_metadata.assert_not_called()  # Should not be called if an error occurs earlier
            assert checkpoint_id in writer._write_results_per_checkpoint_id  # Should not be cleared on failure

        def test_finish_checkpoint_write_metadata_exception(self, mocker, mp_manager):
            """Tests that exceptions from write_metadata propagate in finish_checkpoint."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            checkpoint_id = CheckpointContainerId("/test_checkpoint_finish_exc")
            writer.reset(checkpoint_id.data)
            writer._write_results_per_checkpoint_id[checkpoint_id] = [
                WriteResult(index=torchdistmeta.MetadataIndex("dummy"), size_in_bytes=0, storage_data="")
            ]

            metadata = TestMemoryStorageWriter._create_metadata()
            results = []
            mock_saver.write_metadata.side_effect = RuntimeError("Metadata write failed")

            # When/Then
            with pytest.raises(RuntimeError, match="Metadata write failed"):
                writer.finish_checkpoint(checkpoint_id, metadata, results)
            mock_saver.write_metadata.assert_called_once_with(checkpoint_id, metadata)
            assert checkpoint_id in writer._write_results_per_checkpoint_id  # Should not be cleared on failure

    class TestWriteResultsPerCheckpointId:
        @pytest.fixture
        def writer(self, mocker, mp_manager):
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            return MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)

        def test_init(self, writer):
            """Tests that _write_results_per_checkpoint_id is initialized and empty."""
            # Given/When/Then
            assert len(writer._write_results_per_checkpoint_id) == 0

        def test_reset_does_not_clear(self, writer):
            """Tests that reset() does NOT clear _write_results_per_checkpoint_id."""
            # Given
            checkpoint_id1 = CheckpointContainerId("/test_checkpoint1")
            # Use a real WriteResult as Mocks are not pickleable by the Manager
            wr = WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            writer._write_results_per_checkpoint_id[checkpoint_id1] = [wr]
            # Manually initialize the event for checkpoint_id1, as prepare_write_data_buckets is not called
            writer._write_events_per_checkpoint_id[checkpoint_id1] = writer._mp_manager.Event()
            writer._write_events_per_checkpoint_id[checkpoint_id1].set()
            checkpoint_id2 = CheckpointContainerId("/test_checkpoint2")

            # When
            writer.reset(checkpoint_id2.data)

            # Then
            assert writer.get_write_results(checkpoint_id1) == [wr]
            assert writer._current_checkpoint_id == checkpoint_id2

        def test_write_data_stores_results(self, writer, mocker):
            """Tests that successful write_data stores results under the correct checkpoint_id."""
            # Given
            checkpoint_id = CheckpointContainerId("/test_checkpoint_write")
            writer.reset(checkpoint_id.data)
            item1 = WriteItem(index=torchdistmeta.MetadataIndex("a"), type=WriteItemType.TENSOR)
            item2 = WriteItem(index=torchdistmeta.MetadataIndex("b"), type=WriteItemType.BYTE_IO)
            plan = SavePlan(items=[item1, item2])
            plan = writer.prepare_global_plan([plan])[0]
            planner = mocker.MagicMock()
            write_result = WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            writer._checkpoint_saver.write_data.return_value = [write_result]

            # When
            writer.write_data(plan, planner).wait()

            # Then
            assert writer.get_write_results(checkpoint_id) == [write_result]

        @pytest.mark.parametrize("item_type", [WriteItemType.TENSOR, WriteItemType.BYTE_IO])
        def test_write_data_item_types(self, writer, mocker, item_type):
            """Tests that write_data handles different item types."""
            # Given
            checkpoint_id = CheckpointContainerId(f"/test_item_type_{item_type.name.lower()}")
            writer.reset(checkpoint_id.data)
            item1 = WriteItem(index=torchdistmeta.MetadataIndex("a"), type=item_type)
            item2 = WriteItem(
                index=torchdistmeta.MetadataIndex("b"),
                type=WriteItemType.TENSOR if item_type == WriteItemType.BYTE_IO else WriteItemType.BYTE_IO,
            )
            plan = SavePlan(items=[item1, item2])
            plan = writer.prepare_global_plan([plan])[0]
            planner = mocker.MagicMock()
            write_result = WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            writer._checkpoint_saver.write_data.return_value = [write_result]

            # When
            writer.write_data(plan, planner).wait()

            # Then
            assert writer.get_write_results(checkpoint_id) == [write_result]
            writer._checkpoint_saver.write_data.assert_called_once_with(
                checkpoint_id,
                write_buckets=writer._checkpoint_saver.prepare_write_data.return_value,
                thread_count=1,
                replicate_after_write=True,
            )

        def test_write_data_overwrites_results(self, writer, mocker):
            """Tests that multiple write_data calls for the same ID overwrites results (current behavior)."""
            # Given
            checkpoint_id = CheckpointContainerId("/test_checkpoint_append")
            writer.reset(checkpoint_id.data)
            item1 = WriteItem(index=torchdistmeta.MetadataIndex("a"), type=WriteItemType.TENSOR)
            plan1 = writer.prepare_global_plan([SavePlan(items=[item1])])[0]
            item2 = WriteItem(index=torchdistmeta.MetadataIndex("b"), type=WriteItemType.TENSOR)
            plan2 = writer.prepare_global_plan([SavePlan(items=[item2])])[0]
            planner = mocker.MagicMock()

            wr1 = WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            writer._checkpoint_saver.write_data.return_value = [wr1]
            writer.write_data(plan1, planner).wait()

            wr2 = WriteResult(index=torchdistmeta.MetadataIndex("b"), size_in_bytes=20, storage_data="data_b")
            writer._checkpoint_saver.write_data.return_value = [wr2]

            # When
            writer.write_data(plan2, planner).wait()

            # Then
            assert writer.get_write_results(checkpoint_id) == [wr2]  # Should only contain the last result

        def test_write_data_different_ids(self, writer, mocker):
            """Tests that multiple write_data calls for different IDs store results independently."""
            # Given
            checkpoint_id1 = CheckpointContainerId("/test_checkpoint_diff1")
            writer.reset(checkpoint_id1.data)
            item1 = WriteItem(index=torchdistmeta.MetadataIndex("a"), type=WriteItemType.TENSOR)
            plan1 = writer.prepare_global_plan([SavePlan(items=[item1])])[0]
            wr1 = WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            writer._checkpoint_saver.write_data.return_value = [wr1]
            writer.write_data(plan1, mocker.MagicMock()).wait()

            checkpoint_id2 = CheckpointContainerId("/test_checkpoint_diff2")
            writer.reset(checkpoint_id2.data)
            item2 = WriteItem(index=torchdistmeta.MetadataIndex("b"), type=WriteItemType.TENSOR)
            plan2 = writer.prepare_global_plan([SavePlan(items=[item2])])[0]
            wr2 = WriteResult(index=torchdistmeta.MetadataIndex("b"), size_in_bytes=20, storage_data="data_b")
            writer._checkpoint_saver.write_data.return_value = [wr2]

            # When
            writer.write_data(plan2, mocker.MagicMock()).wait()

            # Then
            assert writer.get_write_results(checkpoint_id1) == [wr1]
            assert writer.get_write_results(checkpoint_id2) == [wr2]

        def test_write_data_empty_saver_results(self, writer, mocker):
            """Tests that empty results from saver are stored as an empty list for the ID."""
            # Given
            checkpoint_id = CheckpointContainerId("/test_checkpoint_empty_res")
            writer.reset(checkpoint_id.data)
            item1 = WriteItem(index=torchdistmeta.MetadataIndex("a"), type=WriteItemType.TENSOR)
            item2 = WriteItem(index=torchdistmeta.MetadataIndex("b"), type=WriteItemType.BYTE_IO)
            plan = writer.prepare_global_plan([SavePlan(items=[item1, item2])])[0]
            writer._checkpoint_saver.write_data.return_value = []

            # When
            writer.write_data(plan, mocker.MagicMock()).wait()

            # Then
            assert writer.get_write_results(checkpoint_id) == []

        def test_write_data_saver_exception(self, writer, mocker):
            """Tests that saver exceptions do not modify results for the ID."""
            # Given
            checkpoint_id = CheckpointContainerId("/test_checkpoint_exc")
            writer.reset(checkpoint_id.data)
            item1 = WriteItem(index=torchdistmeta.MetadataIndex("a"), type=WriteItemType.TENSOR)
            item2 = WriteItem(index=torchdistmeta.MetadataIndex("b"), type=WriteItemType.BYTE_IO)
            plan = writer.prepare_global_plan([SavePlan(items=[item1, item2])])[0]
            writer._checkpoint_saver.write_data.side_effect = RuntimeError("Saver failed")

            # When
            with pytest.raises(RuntimeError, match="Saver failed"):
                writer.write_data(plan, mocker.MagicMock()).wait()

            # Then
            # The event is not set, so get_write_results will raise a RuntimeError
            with pytest.raises(
                RuntimeError, match=re.escape("Event was never set for checkpoint_id '%s'" % checkpoint_id)
            ):
                writer.get_write_results(checkpoint_id)

        def test_get_write_results(self, writer):
            """Tests that get_write_results returns correct results for a given ID, isolated from other IDs."""
            # Given
            checkpoint_id = CheckpointContainerId("/test_get_results")
            expected_wr1 = WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            expected_wr2 = WriteResult(index=torchdistmeta.MetadataIndex("b"), size_in_bytes=20, storage_data="data_b")
            writer._write_results_per_checkpoint_id[checkpoint_id] = [expected_wr1, expected_wr2]
            writer._write_events_per_checkpoint_id[checkpoint_id] = writer._mp_manager.Event()
            writer._write_events_per_checkpoint_id[checkpoint_id].set()

            # Add another checkpoint ID to ensure isolation
            other_checkpoint_id = CheckpointContainerId("/test_get_results_other")
            other_expected_wr1 = WriteResult(
                index=torchdistmeta.MetadataIndex("c"), size_in_bytes=30, storage_data="data_c"
            )
            other_expected_wr2 = WriteResult(
                index=torchdistmeta.MetadataIndex("d"), size_in_bytes=40, storage_data="data_d"
            )
            writer._write_results_per_checkpoint_id[other_checkpoint_id] = [other_expected_wr1, other_expected_wr2]
            writer._write_events_per_checkpoint_id[other_checkpoint_id] = writer._mp_manager.Event()
            writer._write_events_per_checkpoint_id[other_checkpoint_id].set()

            # When
            actual_results = writer.get_write_results(checkpoint_id)

            # Then
            assert actual_results == [expected_wr1, expected_wr2]
            # Also verify that the other checkpoint's results are still there and not affected
            assert writer.get_write_results(other_checkpoint_id) == [other_expected_wr1, other_expected_wr2]

        def test_get_write_results_not_found(self, writer):
            """Tests get_write_results raises KeyError if ID not found."""
            # Given/When/Then
            with pytest.raises(KeyError):
                writer.get_write_results(CheckpointContainerId("/non_existent_id"))

        def test_get_write_results_multiple(self, writer):
            """Tests that get_write_results returns combined results from the dict, isolated from other IDs."""
            # Given
            checkpoint_id = CheckpointContainerId("/test_get_multiple")
            expected_wr1 = WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            expected_wr2 = WriteResult(index=torchdistmeta.MetadataIndex("b"), size_in_bytes=20, storage_data="data_b")
            writer._write_results_per_checkpoint_id[checkpoint_id] = [expected_wr1, expected_wr2]
            writer._write_events_per_checkpoint_id[checkpoint_id] = writer._mp_manager.Event()
            writer._write_events_per_checkpoint_id[checkpoint_id].set()

            # Add another checkpoint ID to ensure isolation
            other_checkpoint_id = CheckpointContainerId("/test_get_multiple_other")
            other_expected_wr1 = WriteResult(
                index=torchdistmeta.MetadataIndex("c"), size_in_bytes=30, storage_data="data_c"
            )
            other_expected_wr2 = WriteResult(
                index=torchdistmeta.MetadataIndex("d"), size_in_bytes=40, storage_data="data_d"
            )
            writer._write_results_per_checkpoint_id[other_checkpoint_id] = [other_expected_wr1, other_expected_wr2]
            writer._write_events_per_checkpoint_id[other_checkpoint_id] = writer._mp_manager.Event()
            writer._write_events_per_checkpoint_id[other_checkpoint_id].set()

            # When
            actual_results = writer.get_write_results(checkpoint_id)

            # Then
            assert actual_results == [expected_wr1, expected_wr2]
            # Also verify that the other checkpoint's results are still there and not affected
            assert writer.get_write_results(other_checkpoint_id) == [other_expected_wr1, other_expected_wr2]

        def test_get_write_results_returns_copy(self, writer):
            """Tests that get_write_results returns a copy of the internal list, isolated from other IDs."""
            # Given
            checkpoint_id = CheckpointContainerId("/test_get_copy")
            expected_wr1 = WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            writer._write_results_per_checkpoint_id[checkpoint_id] = [expected_wr1]
            writer._write_events_per_checkpoint_id[checkpoint_id] = writer._mp_manager.Event()
            writer._write_events_per_checkpoint_id[checkpoint_id].set()

            # When
            actual_results = writer.get_write_results(checkpoint_id)
            # Add another element to see if it mutates the underlying collection
            actual_results.append(
                WriteResult(index=torchdistmeta.MetadataIndex("b"), size_in_bytes=20, storage_data="data_b")
            )

            # Then
            # Ensure the result doesn't have the extra element
            assert writer.get_write_results(checkpoint_id) == [expected_wr1]

        def test_finish_does_not_use_internal_results(self, writer):
            """Tests that finish uses the results argument, not _write_results_per_checkpoint_id."""
            # Given
            checkpoint_id = CheckpointContainerId("/test_finish_uses_arg")
            writer.reset(checkpoint_id.data)
            wr1 = WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            writer._write_results_per_checkpoint_id[checkpoint_id] = [wr1]  # Internal dict has this

            metadata = TestMemoryStorageWriter._create_metadata()
            wr2 = WriteResult(index=torchdistmeta.MetadataIndex("b"), size_in_bytes=20, storage_data="data_b")
            results_arg = [[wr2]]  # Argument to finish has this

            expected_storage_data = {torchdistmeta.MetadataIndex("b"): "data_b"}  # Should only reflect results_arg
            expected_storage_meta = writer.storage_meta()

            # When
            writer.finish(metadata, results_arg)

            # Then
            assert metadata.storage_data == expected_storage_data
            assert metadata.storage_meta == expected_storage_meta
            writer._checkpoint_saver.write_metadata.assert_called_once_with(writer._current_checkpoint_id, metadata)

        def test_finish_duplicate_indices(self, writer):
            """Tests that finish handles duplicate indices in results, last one wins."""
            # Given
            checkpoint_id = CheckpointContainerId("/test_finish_duplicate")
            writer.reset(checkpoint_id.data)
            metadata = TestMemoryStorageWriter._create_metadata()
            idx = torchdistmeta.MetadataIndex("a")
            wr1 = WriteResult(index=idx, size_in_bytes=10, storage_data="data_a1")
            wr2 = WriteResult(index=idx, size_in_bytes=20, storage_data="data_a2")
            results_arg = [[wr1, wr2]]

            expected_storage_data = {idx: "data_a2"}  # Last one wins

            # When
            writer.finish(metadata, results_arg)

            # Then
            assert metadata.storage_data == expected_storage_data

        def test_finish_clears_results(self, writer):
            """Tests that finish clears the results for the current ID from the internal dict."""
            # Given
            checkpoint_id = CheckpointContainerId("/test_finish_clears")
            writer.reset(checkpoint_id.data)
            wr1 = WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            writer._write_results_per_checkpoint_id[checkpoint_id] = [wr1]
            writer._write_events_per_checkpoint_id[checkpoint_id] = writer._mp_manager.Event()
            writer._write_events_per_checkpoint_id[checkpoint_id].set()
            metadata = TestMemoryStorageWriter._create_metadata()

            # When
            writer.finish(metadata, [[wr1]])

            # Then
            assert writer.get_write_results(checkpoint_id) is None

        def test_write_data_in_separate_process(self, writer, mocker):
            """Tests that write_data in a separate process correctly updates the shared results."""
            # Given
            checkpoint_id = CheckpointContainerId("/test_multiprocess")
            writer.reset(checkpoint_id.data)
            items = [
                WriteItem(index=torchdistmeta.MetadataIndex("a"), type=WriteItemType.TENSOR),
                WriteItem(index=torchdistmeta.MetadataIndex("b"), type=WriteItemType.BYTE_IO),
            ]
            plan = SavePlan(items=items)
            # Run through prepare so it attaches the necessary storage data context
            plan = writer.prepare_global_plan([plan])[0]
            planner = mocker.MagicMock()
            write_results = [
                WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a"),
                WriteResult(index=torchdistmeta.MetadataIndex("b"), size_in_bytes=20, storage_data="data_b"),
            ]
            # The mock needs to be picklable to be sent to the other process.
            # A simple MagicMock is not. We can use a simple function for the side effect.
            mocker.patch.object(writer._checkpoint_saver, "write_data", return_value=write_results)

            # When
            p = torch_mp.Process(target=writer.write_data, args=(plan, planner))
            p.start()
            p.join()

            # Then
            assert p.exitcode == 0
            assert writer.get_write_results(checkpoint_id) == write_results

        def test_replicate_written_objects(self, mocker, mp_manager):
            """Tests that replicate_written_objects calls async_replicate_object for each ID and returns futures."""
            # Given
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            writer = MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)
            object_ids = {
                CheckpointObjectId.from_container(CheckpointContainerId("/c1"), "obj1"),
                CheckpointObjectId.from_container(CheckpointContainerId("/c1"), "obj2"),
                CheckpointObjectId.from_container(CheckpointContainerId("/c2"), "obj3"),
            }
            expected_future1 = concurrent.futures.Future()
            expected_future2 = concurrent.futures.Future()
            expected_future3 = concurrent.futures.Future()
            expected_future4 = concurrent.futures.Future()
            expected_future5 = concurrent.futures.Future()
            expected_future6 = concurrent.futures.Future()
            expected_futures = [
                expected_future1,
                expected_future2,
                expected_future3,
                expected_future4,
                expected_future5,
                expected_future6,
            ]
            # Mock return value for sequence of calls (each element is an invocation's return value, which
            # is a list[Future])
            mock_saver.async_replicate_object.side_effect = [
                [expected_future1, expected_future2],
                [expected_future3, expected_future4, expected_future5],
                [expected_future6],
            ]

            # When
            actual_futures = writer.replicate_written_objects(object_ids)

            # Then
            assert actual_futures == expected_futures
            assert mock_saver.async_replicate_object.call_count == 3
            # Order is not guaranteed for set iteration
            called_args = {call.args[0] for call in mock_saver.async_replicate_object.call_args_list}
            assert called_args == object_ids

    class TestGetWriteResults:
        @pytest.fixture(scope="class")
        def mp_manager(self):
            mp_manager = torch_mp.Manager()
            yield mp_manager
            mp_manager.shutdown()

        @pytest.fixture
        def writer(self, mocker, mp_manager):
            mock_saver = mocker.MagicMock(spec=MLFlashpointCheckpointSaver)
            return MemoryStorageWriter(checkpoint_saver=mock_saver, mp_manager=mp_manager)

        def test_get_write_results_waits_for_event_happy_path(self, writer):
            """
            Tests that get_write_results waits for the write event to be set by another thread
            before returning the results.
            """
            # Given
            import threading
            import time

            checkpoint_id = CheckpointContainerId("/test_get_write_results_wait")
            writer.reset(checkpoint_id.data)

            expected_results = [
                WriteResult(index=torchdistmeta.MetadataIndex("a"), size_in_bytes=10, storage_data="data_a")
            ]

            # This function will run in a separate thread to simulate an async write completing
            def set_event_and_results():
                time.sleep(0.3)  # Simulate some work
                writer._write_results_per_checkpoint_id[checkpoint_id] = expected_results
                writer._write_events_per_checkpoint_id[checkpoint_id].set()

            # Manually initialize the event
            writer._write_events_per_checkpoint_id[checkpoint_id] = writer._mp_manager.Event()

            # When
            setter_thread = threading.Thread(target=set_event_and_results)
            setter_thread.start()

            # This call should block until the thread above sets the event
            results = writer.get_write_results(checkpoint_id)

            # Then
            setter_thread.join()
            assert results == expected_results

        def test_get_write_results_waits_for_event_timeout_path(self, writer):
            """
            Tests that get_write_results raises a RuntimeError if the write event is not set
            within the timeout period.
            """
            # Given
            checkpoint_id = CheckpointContainerId("/test_get_write_results_timeout")
            writer.reset(checkpoint_id.data)

            # Do NOT call write_staged_data_buckets, so the event is never set.
            # The event for this checkpoint_id will be initialized but never set.
            writer._write_events_per_checkpoint_id[checkpoint_id] = writer._mp_manager.Event()

            # When/Then
            with pytest.raises(
                RuntimeError,
                match=re.escape(
                    "Event was never set for checkpoint_id '%s', meaning we cannot confirm that the write has "
                    "completed, and its results are available." % checkpoint_id
                ),
            ):
                writer.get_write_results(checkpoint_id)

            # Verify that write_data was not called, as no write operation was initiated
            writer._checkpoint_saver.write_data.assert_not_called()

    class TestPickling:
        def test_pickling_excludes_mp_manager(self, mp_manager):
            """Tests that pickling excludes the _mp_manager attribute."""
            # Given
            import pickle

            dummy_saver = DummySaver()
            writer = MemoryStorageWriter(checkpoint_saver=dummy_saver, mp_manager=mp_manager)

            # When
            pickled = pickle.dumps(writer)
            unpickled = pickle.loads(pickled)

            # Then
            assert unpickled._mp_manager is None
            assert unpickled._checkpoint_saver is not None
            assert isinstance(unpickled._checkpoint_saver, DummySaver)


def _create_rich_object_write_buckets(checkpoint_id: CheckpointContainerId, count: int = 3):
    """Helper to create a list of ObjectWriteBuckets with rich data."""
    buckets = []
    for i in range(count):
        buckets.append(
            ObjectWriteBucket(
                object_id=CheckpointObjectId(f"{checkpoint_id.data}/obj_{i}"),
                object_name=f"obj_{i}",
                bytesio_data=[
                    (
                        WriteItem(index=torchdistmeta.MetadataIndex(f"bytes_{i}"), type=WriteItemType.BYTE_IO),
                        b"some_bytes_data",
                    )
                ],
                tensor_data=[
                    (
                        WriteItem(index=torchdistmeta.MetadataIndex(f"tensor_{i}"), type=WriteItemType.TENSOR),
                        torch.tensor([i, i + 1, i + 2]),
                    )
                ],
            )
        )
    return buckets


def _assert_no_write_results_and_raises_error(writer, checkpoint_id):
    with pytest.raises(
        RuntimeError,
        match=re.escape("Event was never set for checkpoint_id '%s'" % checkpoint_id),
    ):
        writer.get_write_results(checkpoint_id, wait_timeout_sec=0)
