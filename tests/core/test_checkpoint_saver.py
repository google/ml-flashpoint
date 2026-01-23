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

import builtins
import concurrent.futures
import io
import json
import os
import pickle
import shutil
import struct
import tempfile

import pytest
import torch
from torch.distributed.checkpoint import metadata as torchdistmeta
from torch.distributed.checkpoint.planner import (
    BytesIOWriteData,
    MetadataIndex,
    TensorWriteData,
    WriteItem,
    WriteItemType,
)

from ml_flashpoint.checkpoint_object_manager.buffer_io import BufferIO
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.checkpoint_object_manager.object_manager import object_manager_ext
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId, CheckpointObjectId
from ml_flashpoint.core.checkpoint_saver import DefaultMLFlashpointCheckpointSaver, WriteItemResolver
from ml_flashpoint.core.defaults import MAGIC_BYTES
from ml_flashpoint.replication.replication_manager import ReplicationManager


def _load_tensor_maybe_optimized(data):
    if isinstance(data, bytes):
        data = io.BytesIO(data)

    pos = data.tell()
    magic = data.read(8)
    data.seek(pos)

    if magic == MAGIC_BYTES:
        data.seek(pos + 8)
        len_bytes = data.read(4)
        header_len = struct.unpack("<I", len_bytes)[0]
        json_bytes = data.read(header_len)
        metadata = json.loads(json_bytes)

        dtype = getattr(torch, metadata["dtype"])
        shape = metadata["shape"]

        raw_data = data.read()
        tensor = torch.frombuffer(bytearray(raw_data), dtype=dtype).reshape(shape)
        return tensor.clone()
    else:
        return torch.load(data, weights_only=False)


class StubWriteItemResolver(WriteItemResolver):
    def __init__(self, data_map):
        self.data_map = data_map

    def resolve_data(self, write_item: WriteItem):
        if write_item.index in self.data_map:
            return self.data_map[write_item.index]
        raise KeyError(f"Index {write_item.index} not found in data map")


class TestDefaultMLFlashpointCheckpointSaver:
    @pytest.fixture
    def temp_dir_path(self):
        _temp_dir = tempfile.mkdtemp()
        yield _temp_dir
        shutil.rmtree(_temp_dir)

    @pytest.fixture
    def chkpt_object_manager(self):
        return CheckpointObjectManager()

    @pytest.fixture
    def replication_manager(self, mocker):
        return mocker.MagicMock(spec=ReplicationManager)

    @staticmethod
    def _tensor_write_data_for(tensor: torch.Tensor):
        return TensorWriteData(
            chunk=None, properties=torchdistmeta.TensorProperties(dtype=tensor.dtype), size=tensor.shape
        )

    def test_write_data_with_optimized_save_true(
        self, chkpt_object_manager, replication_manager, temp_dir_path, mocker
    ):
        # Given
        saver = DefaultMLFlashpointCheckpointSaver(
            global_rank_getter=lambda: 0,
            local_rank_getter=lambda: 0,
            global_barrier_func=lambda: None,
            ckpt_obj_manager=chkpt_object_manager,
            replication_manager=replication_manager,
            use_optimized_save=True,
        )
        checkpoint_id = CheckpointContainerId(os.path.join(temp_dir_path, "ckpt_opt_true"))
        tensor_data = torch.tensor([1, 2, 3], dtype=torch.int32)
        item_index = MetadataIndex(fqn="item_0")
        resolver = StubWriteItemResolver({item_index: tensor_data})
        write_items = [
            WriteItem(index=item_index, type=WriteItemType.TENSOR, tensor_data=self._tensor_write_data_for(tensor_data))
        ]

        # Prepare bucket
        buckets = saver.prepare_write_data(
            checkpoint_id, write_items, resolver, object_name_prefix="data", bucket_count=1
        )

        # When
        saver.write_data(checkpoint_id, buckets, replicate_after_write=False)

        # Then
        # Read using BufferIO to skip metadata
        object_id = buckets[0].object_id
        buffer_io = chkpt_object_manager.get_buffer(object_id)
        assert buffer_io is not None
        with buffer_io:
            magic = buffer_io.read(8)
            assert magic == b"MLF_TENS"

    def test_write_data_with_optimized_save_false(self, chkpt_object_manager, replication_manager, temp_dir_path):
        # Given
        saver = DefaultMLFlashpointCheckpointSaver(
            global_rank_getter=lambda: 0,
            local_rank_getter=lambda: 0,
            global_barrier_func=lambda: None,
            ckpt_obj_manager=chkpt_object_manager,
            replication_manager=replication_manager,
            use_optimized_save=False,
        )
        checkpoint_id = CheckpointContainerId(os.path.join(temp_dir_path, "ckpt_opt_false"))
        tensor_data = torch.tensor([1, 2, 3], dtype=torch.int32)
        item_index = MetadataIndex(fqn="item_0")
        resolver = StubWriteItemResolver({item_index: tensor_data})
        write_items = [
            WriteItem(index=item_index, type=WriteItemType.TENSOR, tensor_data=self._tensor_write_data_for(tensor_data))
        ]

        # Prepare bucket
        buckets = saver.prepare_write_data(
            checkpoint_id, write_items, resolver, object_name_prefix="data", bucket_count=1
        )

        # When
        saver.write_data(checkpoint_id, buckets, replicate_after_write=False)

        # Then
        # Read using BufferIO to skip metadata
        object_id = buckets[0].object_id
        buffer_io = chkpt_object_manager.get_buffer(object_id)
        assert buffer_io is not None
        with buffer_io:
            magic = buffer_io.read(8)
            assert magic != MAGIC_BYTES

            # Reset position for loading
            buffer_io.seek(0)
            data = buffer_io.read()

        # Should be loadable by torch.load
        data_io = io.BytesIO(data)
        loaded_tensor = torch.load(data_io, weights_only=False)
        assert torch.equal(loaded_tensor, tensor_data)

    @pytest.mark.parametrize(
        "checkpoint_id_suffix",
        [
            "checkpoint_test",
            "checkpoint_test/",
        ],
    )
    def test_get_dirty_marker_file_path(
        self, checkpoint_id_suffix, chkpt_object_manager, replication_manager, temp_dir_path
    ):
        # Given
        local_rank = 123
        saver = DefaultMLFlashpointCheckpointSaver(
            global_rank_getter=lambda: 0,
            local_rank_getter=lambda: local_rank,
            global_barrier_func=lambda: None,
            ckpt_obj_manager=chkpt_object_manager,
            replication_manager=replication_manager,
        )
        checkpoint_id_str = f"{temp_dir_path}/{checkpoint_id_suffix}"
        checkpoint_id = CheckpointContainerId(checkpoint_id_str)
        expected_path = f"{checkpoint_id_str.rstrip('/')}__{local_rank}__unfinished"

        # When/Then
        assert saver._get_dirty_marker_file_path(checkpoint_id) == expected_path

    def test_get_dirty_marker_file_path_root_error(self, mocker, chkpt_object_manager, replication_manager):
        # Given
        saver = DefaultMLFlashpointCheckpointSaver(
            global_rank_getter=lambda: 0,
            local_rank_getter=lambda: 123,
            global_barrier_func=lambda: None,
            ckpt_obj_manager=chkpt_object_manager,
            replication_manager=replication_manager,
        )
        mock_checkpoint_id = mocker.MagicMock(spec=CheckpointContainerId)
        mocker.patch.object(mock_checkpoint_id, "__str__", return_value="/")

        # When/Then
        with pytest.raises(ValueError, match="CheckpointContainerId cannot be the root path '/'"):
            saver._get_dirty_marker_file_path(mock_checkpoint_id)

    class TestInitializeCheckpoint:
        @pytest.mark.parametrize("local_rank, global_rank", [(0, 0), (1, 0), (0, 1), (1, 1), (2, 5)])
        def test_initialize_checkpoint_creates_dirty_marker(
            self,
            local_rank,
            global_rank,
            temp_dir_path: str,
            chkpt_object_manager,
            replication_manager,
        ):
            # Given
            saver = DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=lambda: global_rank,
                local_rank_getter=lambda: local_rank,
                global_barrier_func=lambda: None,
                ckpt_obj_manager=chkpt_object_manager,  # Not needed for this test
                replication_manager=replication_manager,
            )
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_init_{local_rank}_{global_rank}")

            # When
            saver.initialize_checkpoint(checkpoint_id)

            # Then
            expected_dirty_marker_path = f"{checkpoint_id}__{local_rank}__unfinished"
            assert os.path.exists(expected_dirty_marker_path)

        def test_initialize_checkpoint_dirty_marker_fail(
            self,
            mocker,
            temp_dir_path,
            chkpt_object_manager,
            replication_manager,
        ):
            # Given
            local_rank = 0
            saver = DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=lambda: 0,
                local_rank_getter=lambda: local_rank,
                global_barrier_func=lambda: None,
                ckpt_obj_manager=chkpt_object_manager,
                replication_manager=replication_manager,
            )
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_init_fail")

            # Mock open to raise an exception only for the dirty marker file
            original_open = builtins.open

            def mock_open(file, mode):
                if file == f"{checkpoint_id}__{local_rank}__unfinished":
                    raise IOError("Test error: Failed to create dirty marker")
                return original_open(file, mode)

            mocker.patch("builtins.open", side_effect=mock_open)

            # When/Then
            with pytest.raises(IOError, match="Test error: Failed to create dirty marker"):
                saver.initialize_checkpoint(checkpoint_id)

            # Assert that the checkpoint directory was NOT created
            assert not os.path.exists(checkpoint_id.data)

        @pytest.mark.parametrize("local_rank", [0, 1, 8])
        def test_initialize_checkpoint_creates_container_dir_when_not_exists_local_rank_0(
            self,
            local_rank,
            temp_dir_path,
            chkpt_object_manager,
            replication_manager,
        ):
            # Given
            saver = DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=lambda: 0,
                local_rank_getter=lambda: local_rank,
                global_barrier_func=lambda: None,
                ckpt_obj_manager=chkpt_object_manager,  # Not needed for this test
                replication_manager=replication_manager,
            )
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_init_dir_{local_rank}")

            # When
            saver.initialize_checkpoint(checkpoint_id)

            # Then
            # Check for directory creation only on local rank 0
            if local_rank == 0:
                assert os.path.exists(checkpoint_id.data)
                assert os.path.isdir(checkpoint_id.data)
            else:
                assert not os.path.exists(checkpoint_id.data)

        @pytest.mark.parametrize("local_rank", [0, 1])
        def test_initialize_checkpoint_leaves_container_dir_when_exists(
            self,
            local_rank,
            temp_dir_path,
            chkpt_object_manager,
            replication_manager,
        ):
            # Given
            saver = DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=lambda: 0,
                local_rank_getter=lambda: local_rank,
                global_barrier_func=lambda: None,
                ckpt_obj_manager=chkpt_object_manager,  # Not needed for this test
                replication_manager=replication_manager,
            )
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_init_dir_{local_rank}")
            # The checkpoint container dir already exists
            os.makedirs(checkpoint_id.data)
            chkpt_data_object_id = CheckpointObjectId.from_container(checkpoint_id, "test_file.txt")
            chkpt_data_lines = ["hello\n", "world"]
            # And it could even have something in it
            with open(chkpt_data_object_id.data, "w") as file:
                file.writelines(chkpt_data_lines)

            # When
            saver.initialize_checkpoint(checkpoint_id)

            # Then
            # The checkpoint directory and its contents are still in place
            assert os.path.exists(checkpoint_id.data)
            assert os.path.isdir(checkpoint_id.data)
            with open(chkpt_data_object_id.data, "r") as file:
                assert chkpt_data_lines == file.readlines()

        @pytest.mark.parametrize("local_rank", [0, 1, 8])
        def test_initialize_checkpoint_idempotent(
            self,
            temp_dir_path,
            chkpt_object_manager,
            local_rank,
            replication_manager,
        ):
            # Given
            saver = DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=lambda: 0,
                local_rank_getter=lambda: local_rank,
                global_barrier_func=lambda: None,
                ckpt_obj_manager=chkpt_object_manager,  # Not needed for this test
                replication_manager=replication_manager,
            )
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_init_idem_{local_rank}")
            expected_dirty_marker_path = f"{checkpoint_id}__{local_rank}__unfinished"

            # When
            saver.initialize_checkpoint(checkpoint_id)  # First call
            saver.initialize_checkpoint(checkpoint_id)  # Second call

            # Then
            assert os.path.exists(expected_dirty_marker_path)
            if local_rank == 0:
                assert os.path.exists(checkpoint_id.data)
                assert os.path.isdir(checkpoint_id.data)
            else:
                assert not os.path.exists(checkpoint_id.data)

    class TestFinalizeCheckpoint:
        @pytest.mark.parametrize(
            "global_rank, local_rank, dirty_marker_file_exists",
            [
                (0, 0, True),
                (0, 1, True),
                (1, 0, True),
                (1, 1, True),
                (5, 2, True),
                (0, 0, False),
                (0, 1, False),
                (1, 0, False),
                (1, 1, False),
                (5, 2, False),
            ],
        )
        def test_finalize_checkpoint_removes_dirty_marker(
            self,
            global_rank,
            local_rank,
            dirty_marker_file_exists,
            temp_dir_path,
            chkpt_object_manager,
            replication_manager,
        ):
            # Given
            saver = DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=lambda: global_rank,
                local_rank_getter=lambda: local_rank,
                global_barrier_func=lambda: None,
                ckpt_obj_manager=chkpt_object_manager,  # Not needed for this test
                replication_manager=replication_manager,
            )
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_finalizetest_{local_rank}_{global_rank}")
            random_chkpt_file_path = CheckpointObjectId.from_container(checkpoint_id, "test_file.txt")
            chkpt_data_lines = ["hello\n", "world"]
            expected_dirty_marker_path = f"{checkpoint_id}__{local_rank}__unfinished"

            # Create file within checkpoint container for validations later (and its parent dir)
            os.makedirs(checkpoint_id.data)
            with open(random_chkpt_file_path.data, "w") as file:
                file.writelines(chkpt_data_lines)
            if dirty_marker_file_exists:
                # Create the marker file directly to isolate the test
                with open(expected_dirty_marker_path, "w") as _:
                    pass  # empty marker file
                assert os.path.exists(expected_dirty_marker_path)

            # When
            saver.finalize_checkpoint(checkpoint_id)

            # Then
            # The marker file is not present
            assert not os.path.exists(expected_dirty_marker_path)
            # The checkpoint directory and its contents are still in place
            assert os.path.exists(checkpoint_id.data)
            assert os.path.exists(random_chkpt_file_path.data)
            with open(random_chkpt_file_path.data, "r") as file:
                assert chkpt_data_lines == file.readlines()

        def test_finalize_checkpoint_calls_barrier_and_removes_older_in_order(
            self,
            mocker,
            temp_dir_path,
            chkpt_object_manager,
            replication_manager,
        ):
            # Given
            # Using a manager to assert on the sequence of calls.
            manager = mocker.Mock()
            mock_barrier_func = manager.barrier
            mock_remove_dirty_marker = mocker.patch(
                "ml_flashpoint.core.checkpoint_saver.DefaultMLFlashpointCheckpointSaver._remove_dirty_checkpoint_marker"
            )
            manager.attach_mock(mock_remove_dirty_marker, "remove_dirty")
            mock_remove_older = mocker.patch(
                "ml_flashpoint.core.checkpoint_saver.DefaultMLFlashpointCheckpointSaver._remove_older_checkpoints"
            )
            mock_future = mocker.MagicMock()
            mock_remove_older.return_value = mock_future
            manager.attach_mock(mock_remove_older, "remove_older")

            saver = DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=lambda: 0,
                local_rank_getter=lambda: 0,
                global_barrier_func=mock_barrier_func,
                ckpt_obj_manager=chkpt_object_manager,
                replication_manager=replication_manager,
            )
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_finalize_barrier")

            # When
            returned_future = saver.finalize_checkpoint(checkpoint_id)

            # Then
            expected_calls = [
                mocker.call.remove_dirty(checkpoint_id),
                mocker.call.barrier(),
                mocker.call.remove_older(older_than=checkpoint_id),
            ]
            assert manager.mock_calls == expected_calls
            assert returned_future is mock_future

        @pytest.mark.parametrize("local_rank", [0, 1, 5])
        def test_finalize_checkpoint_removes_older_dirs_only_on_local_rank_0(
            self,
            temp_dir_path,
            chkpt_object_manager,
            local_rank,
            replication_manager,
        ):
            # Given
            checkpoint_base_dir = os.path.join(temp_dir_path, "checkpoints")
            older_ckpt_path1 = os.path.join(checkpoint_base_dir, "step-100_ckpt")
            older_ckpt_path2 = os.path.join(checkpoint_base_dir, "step-200_ckpt")
            current_ckpt_path = os.path.join(checkpoint_base_dir, "step-300_ckpt")
            newer_ckpt_path = os.path.join(checkpoint_base_dir, "step-400_ckpt")

            os.makedirs(older_ckpt_path1)
            os.makedirs(older_ckpt_path2)
            os.makedirs(current_ckpt_path)
            os.makedirs(newer_ckpt_path)

            saver = DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=lambda: 0,
                local_rank_getter=lambda: local_rank,
                global_barrier_func=lambda: None,
                ckpt_obj_manager=chkpt_object_manager,
                replication_manager=replication_manager,
            )
            checkpoint_id = CheckpointContainerId(current_ckpt_path)

            # When
            future = saver.finalize_checkpoint(checkpoint_id)
            if local_rank == 0:
                assert future is not None
                future.wait()
            else:
                assert future is None

            # Then
            if local_rank == 0:
                assert not os.path.exists(older_ckpt_path1)
                assert not os.path.exists(older_ckpt_path2)
            else:
                assert os.path.exists(older_ckpt_path1)
                assert os.path.exists(older_ckpt_path2)

            # Current and newer checkpoints should always exist
            assert os.path.exists(current_ckpt_path)
            assert os.path.exists(newer_ckpt_path)

    class TestStageData:
        @pytest.fixture
        def saver(self, chkpt_object_manager, replication_manager):
            return DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=lambda: 0,
                local_rank_getter=lambda: 0,
                global_barrier_func=lambda: None,
                ckpt_obj_manager=chkpt_object_manager,  # Not needed for this test
                replication_manager=replication_manager,
            )

        def test_stage_data_cpu_tensor(self, temp_dir_path, saver):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_stage_cpu")
            state_dict = {"a": torch.tensor([1, 2, 3], device="cpu"), "b": torch.randn(size=(3, 4, 5), device="cpu")}
            staged_dict = saver.stage_data(checkpoint_id, state_dict)
            assert staged_dict["a"].device == torch.device("cpu")
            assert torch.equal(staged_dict["a"], state_dict["a"])

        def test_stage_data_cuda_tensor(self, temp_dir_path, saver):
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_stage_cuda")
            state_dict = {"a": torch.tensor([1, 2, 3], device="cuda")}
            staged_dict = saver.stage_data(checkpoint_id, state_dict)
            assert staged_dict["a"].device == torch.device("cpu")
            assert torch.equal(staged_dict["a"], state_dict["a"].cpu())

        def test_stage_data_mixed(self, temp_dir_path, saver):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_stage_mixed")
            state_dict = {
                "a": torch.tensor([1, 2, 3], device="cpu"),
                "b": "a string",
                "c": 123,
                "d": torch.randn(size=(3, 4, 5), dtype=torch.float16, device="cpu"),
            }
            if torch.cuda.is_available():
                state_dict["g1"] = torch.tensor([4, 5, 6], device="cuda")
                state_dict["g2"] = torch.randn(size=(3, 4, 5), dtype=torch.float32, device="cuda")

            staged_dict = saver.stage_data(checkpoint_id, state_dict)
            assert staged_dict["a"].device == torch.device("cpu")
            assert torch.equal(staged_dict["a"], state_dict["a"])
            assert staged_dict["b"] == "a string"
            assert staged_dict["c"] == 123
            assert torch.equal(staged_dict["d"], state_dict["d"])
            if torch.cuda.is_available():
                assert staged_dict["g1"].device == torch.device("cpu")
                assert torch.equal(staged_dict["g1"], state_dict["g1"].cpu())
                assert staged_dict["g2"].device == torch.device("cpu")
                assert torch.equal(staged_dict["g2"], state_dict["g2"].cpu())

        def test_stage_data_defaults_to_non_blocking(self, temp_dir_path, saver, mocker):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_stage_defaults")
            mock_tensor = mocker.MagicMock(spec=torch.Tensor)
            mock_tensor.device = torch.device("cuda:0")
            state_dict = {"a": mock_tensor}

            saver.stage_data(checkpoint_id, state_dict)

            mock_tensor.to.assert_called_once_with(device="cpu", non_blocking=True)

        @pytest.mark.parametrize("non_blocking", [True, False])
        def test_stage_data_non_blocking_modes(self, temp_dir_path, saver, non_blocking):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_stage_nonblock")
            state_dict = {"a": torch.tensor([1, 2, 3], device="cpu")}
            if torch.cuda.is_available():
                state_dict["b"] = torch.tensor([4, 5, 6], device="cuda")

            staged_dict = saver.stage_data(checkpoint_id, state_dict, non_blocking=non_blocking)
            assert staged_dict["a"].device == torch.device("cpu")
            if torch.cuda.is_available():
                assert staged_dict["b"].device == torch.device("cpu")
                assert torch.equal(staged_dict["b"], state_dict["b"].cpu())

        @pytest.mark.parametrize("non_blocking", [True, False])
        def test_stage_data_moves_all_to_cpu_mocked(self, temp_dir_path, saver, mocker, non_blocking):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_stage_mocked")

            # Mock torch.Tensor.to
            mocker.patch.object(torch.Tensor, "to", wraps=torch.Tensor.to)

            # Create mock tensors with different devices
            mock_tensors_cpu = [mocker.MagicMock(spec=torch.Tensor) for _ in range(2)]
            for t in mock_tensors_cpu:
                t.device = torch.device("cpu")
                t.to.return_value = t

            mock_tensors_xla = [mocker.MagicMock(spec=torch.Tensor) for _ in range(2)]
            for t in mock_tensors_xla:
                t.device = torch.device("xla:0")
                t.to.return_value = mock_tensors_cpu[0]  # Simulate moving to CPU

            mock_tensors_cuda = [mocker.MagicMock(spec=torch.Tensor) for _ in range(2)]
            for t in mock_tensors_cuda:
                t.device = torch.device("cuda:0")
                t.to.return_value = mock_tensors_cpu[0]  # Simulate moving to CPU

            state_dict = {
                "cpu_tensor_0": mock_tensors_cpu[0],
                "cpu_tensor_1": mock_tensors_cpu[1],
                "xla_tensor_0": mock_tensors_xla[0],
                "xla_tensor_1": mock_tensors_xla[1],
                "cuda_tensor_0": mock_tensors_cuda[0],
                "cuda_tensor_1": mock_tensors_cuda[1],
                "string_data": "hello",
                "int_data": 123,
            }

            staged_dict = saver.stage_data(checkpoint_id, state_dict, non_blocking=non_blocking)

            # Assert that .to() was called correctly for all tensors
            for t in mock_tensors_cpu:
                t.to.assert_called_once_with(device="cpu", non_blocking=non_blocking)
            for t in mock_tensors_xla:
                t.to.assert_called_once_with(device="cpu", non_blocking=non_blocking)
            for t in mock_tensors_cuda:
                t.to.assert_called_once_with(device="cpu", non_blocking=non_blocking)

            # Assert that the staged dict contains the tensors returned by .to()
            assert staged_dict["cpu_tensor_0"] is mock_tensors_cpu[0]
            assert staged_dict["cpu_tensor_1"] is mock_tensors_cpu[1]
            assert staged_dict["xla_tensor_0"] is mock_tensors_cpu[0]
            assert staged_dict["xla_tensor_1"] is mock_tensors_cpu[0]
            assert staged_dict["cuda_tensor_0"] is mock_tensors_cpu[0]
            assert staged_dict["cuda_tensor_1"] is mock_tensors_cpu[0]

        @pytest.mark.parametrize("non_blocking", [True, False])
        @pytest.mark.parametrize("cuda_available", [True, False])
        def test_stage_data_cuda_synchronization(self, temp_dir_path, saver, mocker, non_blocking, cuda_available):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_stage_sync")

            mocker.patch("torch.cuda.is_available", return_value=cuda_available)
            mock_cuda_synchronize = mocker.patch("torch.cuda.synchronize")

            state_dict = {"a": torch.tensor([[1, 2, 3], [4, 5, 6]])}

            saver.stage_data(checkpoint_id, state_dict, non_blocking=non_blocking)

            if non_blocking and cuda_available:
                mock_cuda_synchronize.assert_called_once()
            else:
                # Should not invoke it unnecessarily, as it can cause unnecessary slowdowns
                mock_cuda_synchronize.assert_not_called()

    class TestPrepareWriteData:
        @pytest.fixture
        def saver(self, chkpt_object_manager, replication_manager):
            return DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=lambda: 0,
                local_rank_getter=lambda: 0,
                global_barrier_func=lambda: None,
                ckpt_obj_manager=chkpt_object_manager,
                replication_manager=replication_manager,
            )

        @staticmethod
        def _tensor_write_data_for(tensor: torch.Tensor):
            return TensorWriteData(
                chunk=None, properties=torchdistmeta.TensorProperties(dtype=tensor.dtype), size=tensor.shape
            )

        def test_prepare_write_data_single_tensor(self, saver, temp_dir_path):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_prepare_single_tensor")
            tensor_data = torch.tensor([1, 2, 3])
            item_index = MetadataIndex(fqn="item_0")
            data_map = {item_index: tensor_data}
            resolver = StubWriteItemResolver(data_map)
            write_items = [
                WriteItem(
                    index=item_index, type=WriteItemType.TENSOR, tensor_data=self._tensor_write_data_for(tensor_data)
                )
            ]

            write_buckets = saver.prepare_write_data(
                checkpoint_id, write_items, resolver, object_name_prefix="data", bucket_count=1
            )

            assert len(write_buckets) == 1
            bucket = write_buckets[0]
            assert bucket.object_name == "data_0_src0.distcp"
            assert len(bucket.tensor_data) == 1
            assert len(bucket.bytesio_data) == 0
            assert torch.equal(bucket.tensor_data[0][1], tensor_data)

        def test_prepare_write_data_single_byteio(self, saver, temp_dir_path):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_prepare_single_byteio")
            data_binary = b"test byte data"
            bytesio_data = io.BytesIO(data_binary)
            item_index = MetadataIndex(fqn="item_0")
            data_map = {item_index: bytesio_data}
            resolver = StubWriteItemResolver(data_map)
            write_items = [
                WriteItem(
                    index=item_index, type=WriteItemType.BYTE_IO, bytes_io_data=BytesIOWriteData(len(data_binary))
                )
            ]

            write_buckets = saver.prepare_write_data(
                checkpoint_id, write_items, resolver, object_name_prefix="data", bucket_count=1
            )

            assert len(write_buckets) == 1
            bucket = write_buckets[0]
            assert bucket.object_name == "data_0_src0.distcp"
            assert len(bucket.bytesio_data) == 1
            assert len(bucket.tensor_data) == 0
            assert bucket.bytesio_data[0][1].getvalue() == data_binary

        def test_save_tensor_optimized_writes_correct_header_and_data(self, saver, chkpt_object_manager, mocker):
            """Test that _save_tensor_optimized writes the expected zero-copy format."""
            # Given
            tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
            buffer_io_mock = mocker.MagicMock(spec=BufferIO)
            # Create a real memory view for the mock to return so torch.frombuffer works
            real_buffer = bytearray(tensor.nbytes)
            buffer_io_mock.get_buffer_slice.return_value = memoryview(real_buffer)

            # When
            saver._save_tensor_optimized(tensor, buffer_io_writer=buffer_io_mock)

            # Then
            # 1. Verify Header Write
            buffer_io_mock.write.assert_called_once()
            header_bytes = buffer_io_mock.write.call_args[0][0]

            # Parse header manually to verify
            magic = header_bytes[:8]
            assert magic == b"MLF_TENS"

            len_bytes = header_bytes[8:12]
            header_len = struct.unpack("<I", len_bytes)[0]

            json_bytes = header_bytes[12:]
            assert len(json_bytes) == header_len

            metadata = json.loads(json_bytes)
            assert metadata["dtype"] == "int32"
            assert metadata["shape"] == [3]

            # 2. Verify Data Copy
            buffer_io_mock.get_buffer_slice.assert_called_once_with(tensor.nbytes)

            # Verify data in our side-buffer matches
            written_tensor = torch.frombuffer(real_buffer, dtype=torch.int32)
            assert torch.equal(written_tensor, tensor)

        @pytest.mark.parametrize("bucket_count", [1, 2])
        def test_prepare_write_data_multiple_tensors(self, saver, temp_dir_path, bucket_count):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_prepare_multi_tensor")
            tensor0 = torch.tensor([1, 2, 3])
            tensor1 = torch.tensor([4, 5, 6])
            item_index0 = MetadataIndex(fqn="item_0")
            item_index1 = MetadataIndex(fqn="item_1")
            data_map = {item_index0: tensor0, item_index1: tensor1}
            resolver = StubWriteItemResolver(data_map)
            write_items = [
                WriteItem(
                    index=item_index0, type=WriteItemType.TENSOR, tensor_data=self._tensor_write_data_for(tensor0)
                ),
                WriteItem(
                    index=item_index1, type=WriteItemType.TENSOR, tensor_data=self._tensor_write_data_for(tensor1)
                ),
            ]

            write_buckets = saver.prepare_write_data(
                checkpoint_id, write_items, resolver, object_name_prefix="data", bucket_count=bucket_count
            )

            assert len(write_buckets) == bucket_count
            all_tensors = [item[1] for bucket in write_buckets for item in bucket.tensor_data]
            assert len(all_tensors) == 2
            assert any(torch.equal(t, tensor0) for t in all_tensors)
            assert any(torch.equal(t, tensor1) for t in all_tensors)

        @pytest.mark.parametrize("bucket_count", [1, 2, 3])
        def test_prepare_write_data_mixed_types(self, saver, temp_dir_path, bucket_count):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_prepare_mixed")
            tensor0 = torch.tensor([1, 2, 3])
            bytes1 = b"test byte data"
            bytesio1 = io.BytesIO(bytes1)
            tensor2 = torch.tensor([7, 8, 9])
            item_index0 = MetadataIndex(fqn="item_0_tensor")
            item_index1 = MetadataIndex(fqn="item_1_bytes")
            item_index2 = MetadataIndex(fqn="item_2_tensor")
            data_map = {item_index0: tensor0, item_index1: bytesio1, item_index2: tensor2}
            resolver = StubWriteItemResolver(data_map)
            write_items = [
                WriteItem(
                    index=item_index0, type=WriteItemType.TENSOR, tensor_data=self._tensor_write_data_for(tensor0)
                ),
                WriteItem(index=item_index1, type=WriteItemType.BYTE_IO, bytes_io_data=BytesIOWriteData(len(bytes1))),
                WriteItem(
                    index=item_index2, type=WriteItemType.TENSOR, tensor_data=self._tensor_write_data_for(tensor2)
                ),
            ]

            write_buckets = saver.prepare_write_data(
                checkpoint_id, write_items, resolver, object_name_prefix="data", bucket_count=bucket_count
            )

            assert len(write_buckets) > 0
            all_tensors = [item[1] for bucket in write_buckets for item in bucket.tensor_data]
            all_bytes = [item[1] for bucket in write_buckets for item in bucket.bytesio_data]
            assert len(all_tensors) == 2
            assert len(all_bytes) == 1
            assert any(torch.equal(t, tensor0) for t in all_tensors)
            assert any(torch.equal(t, tensor2) for t in all_tensors)
            assert all_bytes[0].getvalue() == bytes1

        def test_prepare_write_data_empty_items(self, saver, temp_dir_path):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_prepare_empty")
            resolver = StubWriteItemResolver({})
            write_items = []

            write_buckets = saver.prepare_write_data(
                checkpoint_id, write_items, resolver, object_name_prefix="data", bucket_count=1
            )

            assert write_buckets == []

        def test_prepare_write_data_resolver_exception(self, saver, temp_dir_path):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_prepare_resolver_exc")
            tensor_data = torch.tensor([1, 2, 3])
            item_index = MetadataIndex(fqn="item_0")
            resolver = StubWriteItemResolver({})  # Empty data map
            write_items = [
                WriteItem(
                    index=item_index, type=WriteItemType.TENSOR, tensor_data=self._tensor_write_data_for(tensor_data)
                )
            ]

            with pytest.raises(KeyError):
                saver.prepare_write_data(
                    checkpoint_id, write_items, resolver, object_name_prefix="data", bucket_count=1
                )

        def test_prepare_write_data_clone_if_needed_logic(self, saver, temp_dir_path, mocker):
            # Given
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_clone_logic")

            # 1. CPU tensor, contiguous, not a view
            cpu_tensor_contig = torch.randn(10)

            # 2. CPU tensor, non-contiguous
            cpu_tensor_non_contig = torch.randn(10, 2).t()

            # 3. CPU tensor, view (contiguous but view of larger storage)
            base_tensor = torch.randn(20)
            cpu_tensor_view = base_tensor[2:5]  # length 3, contiguous

            # 4. Mock CUDA tensor
            cuda_tensor = mocker.MagicMock(spec=torch.Tensor)
            cuda_tensor.device.type = "cuda"
            # prepare_write_data calls .detach() on the resolved data
            cuda_tensor.detach.return_value = cuda_tensor

            item_indices = {
                "contig": MetadataIndex(fqn="contig"),
                "non_contig": MetadataIndex(fqn="non_contig"),
                "view": MetadataIndex(fqn="view"),
                "cuda": MetadataIndex(fqn="cuda"),
            }

            data_map = {
                item_indices["contig"]: cpu_tensor_contig,
                item_indices["non_contig"]: cpu_tensor_non_contig,
                item_indices["view"]: cpu_tensor_view,
                item_indices["cuda"]: cuda_tensor,
            }
            resolver = StubWriteItemResolver(data_map)

            write_items = [
                WriteItem(
                    index=item_indices["contig"],
                    type=WriteItemType.TENSOR,
                    tensor_data=self._tensor_write_data_for(cpu_tensor_contig),
                ),
                WriteItem(
                    index=item_indices["non_contig"],
                    type=WriteItemType.TENSOR,
                    tensor_data=self._tensor_write_data_for(cpu_tensor_non_contig),
                ),
                WriteItem(
                    index=item_indices["view"],
                    type=WriteItemType.TENSOR,
                    tensor_data=self._tensor_write_data_for(cpu_tensor_view),
                ),
                WriteItem(index=item_indices["cuda"], type=WriteItemType.TENSOR, tensor_data=mocker.MagicMock()),
            ]

            # When
            write_buckets = saver.prepare_write_data(
                checkpoint_id, write_items, resolver, object_name_prefix="data", bucket_count=1
            )

            # Then
            bucket = write_buckets[0]
            resolved_tensors = {item[0].index.fqn: item[1] for item in bucket.tensor_data}

            # 1. Contiguous CPU tensor (not view) should be contiguous and share the same data pointer
            # (Note: .detach() creates a new tensor object but shares storage)
            assert resolved_tensors["contig"].is_contiguous()
            assert resolved_tensors["contig"].data_ptr() == cpu_tensor_contig.data_ptr()

            # 2. Non-contiguous CPU tensor should become contiguous (and thus a different storage/data pointer)
            assert resolved_tensors["non_contig"].is_contiguous()
            assert resolved_tensors["non_contig"].data_ptr() != cpu_tensor_non_contig.data_ptr()
            assert torch.equal(resolved_tensors["non_contig"], cpu_tensor_non_contig)

            # 3. CPU tensor view should be cloned (different storage/data pointer)
            assert resolved_tensors["view"].is_contiguous()
            assert resolved_tensors["view"].data_ptr() != cpu_tensor_view.data_ptr()
            assert torch.equal(resolved_tensors["view"], cpu_tensor_view)
            # Verify it's not sharing storage anymore
            assert (
                resolved_tensors["view"].untyped_storage().size()
                == resolved_tensors["view"].numel() * resolved_tensors["view"].itemsize
            )

            # 4. CUDA tensor should be the EXACT same object (not cloned, not contiguous called on it)
            assert resolved_tensors["cuda"] is cuda_tensor
            cuda_tensor.contiguous.assert_not_called()

    class TestWriteData:
        @pytest.fixture
        def saver(self, chkpt_object_manager, replication_manager):
            return DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=lambda: 0,
                local_rank_getter=lambda: 0,
                global_barrier_func=lambda: None,
                ckpt_obj_manager=chkpt_object_manager,
                replication_manager=replication_manager,
            )

        @pytest.mark.parametrize("thread_count", [1, 2, 3])
        def test_write_data_single_tensor(self, saver, temp_dir_path, chkpt_object_manager, thread_count):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_write_data_tensor_th{thread_count}")
            os.makedirs(checkpoint_id.data, exist_ok=True)

            tensor_data = torch.tensor([1, 2, 3])
            item_index = MetadataIndex(fqn="item_0")
            data_map = {item_index: tensor_data}
            resolver = StubWriteItemResolver(data_map)

            write_items = [
                WriteItem(
                    index=item_index, type=WriteItemType.TENSOR, tensor_data=self._tensor_write_data_for(tensor_data)
                ),
            ]
            write_buckets = saver.prepare_write_data(
                checkpoint_id, write_items, resolver, "data", bucket_count=thread_count
            )

            actual_results = saver.write_data(
                checkpoint_id,
                write_buckets,
                thread_count=thread_count,
                replicate_after_write=False,
            )

            assert actual_results is not None
            assert len(actual_results) == 1

            self._assert_write_result(
                actual_results[0],
                checkpoint_id,
                item_index,
                tensor_data,
                "data_0_src0.distcp",
                0,
                chkpt_object_manager,
            )

        @pytest.mark.parametrize("thread_count", [1, 2, 3])
        def test_write_data_single_byteio(self, saver, temp_dir_path, chkpt_object_manager, thread_count):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_write_data_byteio_th{thread_count}")
            os.makedirs(checkpoint_id.data, exist_ok=True)

            data_binary = b"test byte data"
            bytesio_data = io.BytesIO(data_binary)
            item_index = MetadataIndex(fqn="item_0")
            data_map = {item_index: bytesio_data}
            resolver = StubWriteItemResolver(data_map)

            write_items = [
                WriteItem(
                    index=item_index, type=WriteItemType.BYTE_IO, bytes_io_data=BytesIOWriteData(len(data_binary))
                ),
            ]
            write_buckets = saver.prepare_write_data(
                checkpoint_id, write_items, resolver, "data", bucket_count=thread_count
            )

            actual_results = saver.write_data(
                checkpoint_id,
                write_buckets,
                thread_count=thread_count,
                replicate_after_write=False,
            )

            assert actual_results is not None

            expected_object_id = CheckpointObjectId.from_container(checkpoint_id, "data_0_src0.distcp")
            assert os.path.exists(expected_object_id.data)

            with chkpt_object_manager.get_buffer(expected_object_id) as buffer_reader:
                loaded_bytes = buffer_reader.read()
                assert loaded_bytes == data_binary

            assert len(actual_results) == 1
            assert actual_results[0].index == item_index
            assert actual_results[0].storage_data.relative_path == "data_0_src0.distcp"
            assert actual_results[0].size_in_bytes == len(data_binary)
            assert actual_results[0].storage_data.offset == 0

        @pytest.mark.parametrize("thread_count", [1, 2, 3])
        def test_write_data_multiple_tensors(self, saver, temp_dir_path, chkpt_object_manager, thread_count):
            checkpoint_id = CheckpointContainerId(
                f"{temp_dir_path}/checkpoint_write_data_multi_tensor_th{thread_count}"
            )
            os.makedirs(checkpoint_id.data, exist_ok=True)

            expected_tensor0 = torch.tensor([1, 2, 3])
            expected_tensor1 = torch.tensor([4, 5, 6])
            item_index0 = MetadataIndex(fqn="item_0")
            item_index1 = MetadataIndex(fqn="item_1")
            data_map = {item_index0: expected_tensor0, item_index1: expected_tensor1}
            resolver = StubWriteItemResolver(data_map)

            write_items = [
                WriteItem(
                    index=item_index0,
                    type=WriteItemType.TENSOR,
                    tensor_data=self._tensor_write_data_for(expected_tensor0),
                ),
                WriteItem(
                    index=item_index1,
                    type=WriteItemType.TENSOR,
                    tensor_data=self._tensor_write_data_for(expected_tensor1),
                ),
            ]
            write_buckets = saver.prepare_write_data(
                checkpoint_id, write_items, resolver, "data", bucket_count=thread_count
            )

            actual_results = saver.write_data(
                checkpoint_id,
                write_buckets,
                thread_count=thread_count,
                replicate_after_write=False,
            )

            assert actual_results is not None
            actual_results.sort(key=lambda x: x.index.fqn)

            assert len(actual_results) == 2
            res0 = actual_results[0]
            assert res0 is not None
            res1 = actual_results[1]
            assert res1 is not None

            assert res0.index == item_index0
            assert res1.index == item_index1

            # Dynamically find the expected path for each item from the prepared buckets
            res0_expected_path = None
            res1_expected_path = None
            for b in write_buckets:
                for item, _ in b.tensor_data:
                    if item.index == item_index0:
                        res0_expected_path = b.object_name
                    if item.index == item_index1:
                        res1_expected_path = b.object_name
            assert res0_expected_path is not None
            assert res1_expected_path is not None
            assert res0.storage_data.relative_path == res0_expected_path
            assert res1.storage_data.relative_path == res1_expected_path

            self._assert_write_result(
                res0, checkpoint_id, item_index0, expected_tensor0, res0_expected_path, 0, chkpt_object_manager
            )
            # If in the same file, offset is after the first item. Otherwise, it's a new file, offset is 0.
            expected_offset1 = res0.size_in_bytes if res0_expected_path == res1_expected_path else 0
            self._assert_write_result(
                res1,
                checkpoint_id,
                item_index1,
                expected_tensor1,
                res1_expected_path,
                expected_offset1,
                chkpt_object_manager,
            )

        @pytest.mark.parametrize("thread_count", [1, 2, 3])
        def test_write_data_mixed_types(self, saver, temp_dir_path, chkpt_object_manager, thread_count):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_write_data_mixed_th{thread_count}")
            os.makedirs(checkpoint_id.data, exist_ok=True)

            expected_tensor0 = torch.tensor([1, 2, 3])
            expected_bytes1 = b"test byte data"
            bytesio1 = io.BytesIO(expected_bytes1)
            expected_tensor2 = torch.tensor([7, 8, 9])

            item_index0 = MetadataIndex(fqn="item_0_tensor")
            item_index1 = MetadataIndex(fqn="item_1_bytes")
            item_index2 = MetadataIndex(fqn="item_2_tensor")

            data_map = {
                item_index0: expected_tensor0,
                item_index1: bytesio1,
                item_index2: expected_tensor2,
            }
            resolver = StubWriteItemResolver(data_map)

            write_items = [
                WriteItem(
                    index=item_index0,
                    type=WriteItemType.TENSOR,
                    tensor_data=self._tensor_write_data_for(expected_tensor0),
                ),
                WriteItem(
                    index=item_index1, type=WriteItemType.BYTE_IO, bytes_io_data=BytesIOWriteData(len(expected_bytes1))
                ),
                WriteItem(
                    index=item_index2,
                    type=WriteItemType.TENSOR,
                    tensor_data=self._tensor_write_data_for(expected_tensor2),
                ),
            ]
            write_buckets = saver.prepare_write_data(
                checkpoint_id, write_items, resolver, "data", bucket_count=thread_count
            )

            actual_results = saver.write_data(
                checkpoint_id,
                write_buckets,
                thread_count=thread_count,
                replicate_after_write=False,
            )

            assert actual_results is not None
            actual_results.sort(key=lambda x: x.index.fqn)

            assert len(actual_results) == 3

            res0 = actual_results[0]  # expected_tensor0
            res1 = actual_results[1]  # bytesio1
            res2 = actual_results[2]  # expected_tensor2

            assert res0.index == item_index0
            assert res1.index == item_index1
            assert res2.index == item_index2

            # Assertions on storage_data paths
            res0_expected_path = None
            res1_expected_path = None
            res2_expected_path = None
            for b in write_buckets:
                for item, _ in b.tensor_data:
                    if item.index == item_index0:
                        res0_expected_path = b.object_name
                    if item.index == item_index2:
                        res2_expected_path = b.object_name
                for item, _ in b.bytesio_data:
                    if item.index == item_index1:
                        res1_expected_path = b.object_name
            assert res0_expected_path is not None
            assert res1_expected_path is not None
            assert res2_expected_path is not None
            assert res0.storage_data.relative_path == res0_expected_path
            assert res1.storage_data.relative_path == res1_expected_path
            assert res2.storage_data.relative_path == res2_expected_path

            assert res0.size_in_bytes > 0
            assert res1.size_in_bytes == len(expected_bytes1)
            assert res2.size_in_bytes > 0

            # The offsets are verified within the results_by_file loop below, as they depend on the bucketing strategy.
            results_by_file = {}
            for res in actual_results:
                fname = res.storage_data.relative_path
                if fname not in results_by_file:
                    results_by_file[fname] = []
                results_by_file[fname].append(res)

            for fname, results in results_by_file.items():
                results.sort(key=lambda x: x.storage_data.offset)
                obj_id = CheckpointObjectId.from_container(checkpoint_id, fname)
                assert os.path.exists(obj_id.data)

                with chkpt_object_manager.get_buffer(obj_id) as buffer_reader:
                    current_offset = 0
                    for res in results:
                        # Assert that offsets are sequential within each file
                        assert res.storage_data.offset == current_offset

                        sinfo = res.storage_data
                        buffer_reader.seek(sinfo.offset)

                        if res.index == item_index1:  # expected_bytes1
                            loaded_bytes = buffer_reader.read(sinfo.length)
                            assert loaded_bytes == expected_bytes1
                        else:  # Tensor items
                            tensor_bytes = buffer_reader.read(sinfo.length)
                            loaded_tensor = _load_tensor_maybe_optimized(tensor_bytes)
                            if res.index == item_index0:
                                assert torch.equal(loaded_tensor, expected_tensor0)
                            elif res.index == item_index2:
                                assert torch.equal(loaded_tensor, expected_tensor2)
                        current_offset += sinfo.length

        @pytest.mark.parametrize("thread_count", [1, 2, 3])
        def test_write_data_io_error(self, saver, temp_dir_path, mocker, thread_count):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_write_data_io_error_th{thread_count}")
            os.makedirs(checkpoint_id.data, exist_ok=True)

            tensor_data = torch.tensor([1, 2, 3])
            item_index = MetadataIndex(fqn="item_0")
            data_map = {item_index: tensor_data}
            resolver = StubWriteItemResolver(data_map)

            write_items = [
                WriteItem(
                    index=item_index, type=WriteItemType.TENSOR, tensor_data=self._tensor_write_data_for(tensor_data)
                ),
            ]
            write_buckets = saver.prepare_write_data(
                checkpoint_id, write_items, resolver, "data", bucket_count=thread_count
            )

            # Mock the checkpoint object manager to raise an IOError during buffer creation
            mocker.patch.object(saver._chkpt_obj_manager, "create_buffer", side_effect=IOError("Test IOError"))
            with pytest.raises(IOError):
                saver.write_data(
                    checkpoint_id,
                    write_buckets,
                    thread_count=thread_count,
                    replicate_after_write=False,
                )

        @pytest.mark.parametrize("thread_count", [1, 2, 3])
        def test_write_data_empty_buckets(self, saver, temp_dir_path, thread_count):
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_write_data_empty_th{thread_count}")
            os.makedirs(checkpoint_id.data, exist_ok=True)

            actual_results = saver.write_data(
                checkpoint_id,
                write_buckets=[],
                thread_count=thread_count,
                replicate_after_write=False,
            )

            assert actual_results == []

            # Check that no files were created
            assert not os.listdir(checkpoint_id.data)

        @pytest.mark.parametrize("preexisting_content", [b"", b"some old data"])
        def test_write_data_overwrite(self, saver, temp_dir_path, chkpt_object_manager, preexisting_content):
            # Given
            thread_count = 1
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_write_data_overwrite")
            os.makedirs(checkpoint_id.data, exist_ok=True)

            tensor_data = torch.tensor([1, 2, 3])
            item_index = MetadataIndex(fqn="item_0")
            data_map = {item_index: tensor_data}
            resolver = StubWriteItemResolver(data_map)

            write_items = [
                WriteItem(
                    index=item_index, type=WriteItemType.TENSOR, tensor_data=self._tensor_write_data_for(tensor_data)
                ),
            ]
            write_buckets = saver.prepare_write_data(
                checkpoint_id, write_items, resolver, "data", bucket_count=thread_count
            )

            # Create a file with the same name that will be written to
            expected_object_name = write_buckets[0].object_name
            object_id = CheckpointObjectId.from_container(checkpoint_id, expected_object_name)
            with open(object_id.data, "wb") as f:
                f.write(preexisting_content)

            # When
            actual_results = saver.write_data(
                checkpoint_id,
                write_buckets,
                thread_count=thread_count,
                replicate_after_write=False,
            )

            # Then
            assert actual_results is not None
            assert len(actual_results) == 1

            # Assert that the object name written to is the same as the one we pre-created
            assert actual_results[0].storage_data.relative_path == expected_object_name

            self._assert_write_result(
                actual_results[0],
                checkpoint_id,
                item_index,
                tensor_data,
                expected_object_name,
                0,
                chkpt_object_manager,
            )

            # Additionally, assert that the content of the file has changed from the preexisting content
            object_id = CheckpointObjectId.from_container(checkpoint_id, write_buckets[0].object_name)
            with chkpt_object_manager.get_buffer(object_id) as buffer_reader:
                loaded_tensor = _load_tensor_maybe_optimized(buffer_reader.read())
                assert torch.equal(loaded_tensor, tensor_data)

        @pytest.mark.parametrize("thread_count", [0, -1, -5])
        def test_write_data_thread_count_less_than_1(self, saver, temp_dir_path, chkpt_object_manager, thread_count):
            # Given
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_write_data_tensor_th{thread_count}")
            os.makedirs(checkpoint_id.data, exist_ok=True)

            tensor_data1 = torch.tensor([1, 2, 3])
            tensor_data2 = torch.tensor([4, 5, 6])
            item_index1 = MetadataIndex(fqn="item_0")
            item_index2 = MetadataIndex(fqn="item_1")
            data_map = {item_index1: tensor_data1, item_index2: tensor_data2}
            resolver = StubWriteItemResolver(data_map)

            write_items = [
                WriteItem(
                    index=item_index1, type=WriteItemType.TENSOR, tensor_data=self._tensor_write_data_for(tensor_data1)
                ),
                WriteItem(
                    index=item_index2, type=WriteItemType.TENSOR, tensor_data=self._tensor_write_data_for(tensor_data2)
                ),
            ]
            write_buckets = saver.prepare_write_data(checkpoint_id, write_items, resolver, "data", bucket_count=1)

            # When
            actual_results = saver.write_data(
                checkpoint_id,
                write_buckets,
                thread_count=thread_count,
                replicate_after_write=False,
            )

            # Then
            assert actual_results is not None
            actual_results.sort(key=lambda x: x.index.fqn)

            assert len(actual_results) == 2

            res1 = actual_results[0]
            self._assert_write_result(
                res1, checkpoint_id, item_index1, tensor_data1, "data_0_src0.distcp", 0, chkpt_object_manager
            )

            res2 = actual_results[1]
            self._assert_write_result(
                res2,
                checkpoint_id,
                item_index2,
                tensor_data2,
                "data_0_src0.distcp",
                res1.size_in_bytes,
                chkpt_object_manager,
            )

        def test_write_data_triggers_replication(self, saver, temp_dir_path, replication_manager):
            # Given
            thread_count = 1
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_write_data_replication")
            os.makedirs(checkpoint_id.data, exist_ok=True)

            tensor_data = torch.tensor([1, 2, 3])
            item_index = MetadataIndex(fqn="item_0")
            data_map = {item_index: tensor_data}
            resolver = StubWriteItemResolver(data_map)

            write_items = [
                WriteItem(
                    index=item_index, type=WriteItemType.TENSOR, tensor_data=self._tensor_write_data_for(tensor_data)
                ),
            ]
            write_buckets = saver.prepare_write_data(
                checkpoint_id, write_items, resolver, "data", bucket_count=thread_count
            )

            # When
            saver.write_data(
                checkpoint_id,
                write_buckets,
                thread_count=thread_count,
                replicate_after_write=True,
            )

            # Then
            assert replication_manager.async_replicate.call_count == 1
            args, _ = replication_manager.async_replicate.call_args
            buffer_io = args[0]

            assert isinstance(buffer_io, BufferIO)

            assert len(write_buckets) == 1
            expected_object_name = write_buckets[0].object_name
            expected_object_id = CheckpointObjectId.from_container(checkpoint_id, expected_object_name)
            assert buffer_io.buffer_obj.get_id() == str(expected_object_id)
            buffer_io.close()

        def test_write_data_no_replication(self, saver, temp_dir_path, replication_manager):
            # Given
            thread_count = 1
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_write_data_no_replication")
            os.makedirs(checkpoint_id.data, exist_ok=True)

            tensor_data = torch.tensor([1, 2, 3])
            item_index = MetadataIndex(fqn="item_0")
            data_map = {item_index: tensor_data}
            resolver = StubWriteItemResolver(data_map)

            write_items = [
                WriteItem(
                    index=item_index, type=WriteItemType.TENSOR, tensor_data=self._tensor_write_data_for(tensor_data)
                ),
            ]
            write_buckets = saver.prepare_write_data(
                checkpoint_id, write_items, resolver, "data", bucket_count=thread_count
            )

            # When
            saver.write_data(
                checkpoint_id,
                write_buckets,
                thread_count=thread_count,
                replicate_after_write=False,
            )

            # Then
            replication_manager.async_replicate.assert_not_called()

        @staticmethod
        def _tensor_write_data_for(tensor: torch.Tensor):
            return TensorWriteData(
                chunk=None, properties=torchdistmeta.TensorProperties(dtype=tensor.dtype), size=tensor.shape
            )

        @staticmethod
        def _assert_write_result(
            result,
            checkpoint_id,
            expected_index,
            expected_tensor,
            expected_object_name,
            expected_offset,
            chkpt_obj_manager,
        ):
            assert result.index == expected_index
            assert result.storage_data.relative_path == expected_object_name
            assert result.size_in_bytes > 0
            assert result.storage_data.offset == expected_offset

            object_id = CheckpointObjectId.from_container(checkpoint_id, result.storage_data.relative_path)
            assert os.path.exists(object_id.data)
            with chkpt_obj_manager.get_buffer(object_id) as buffer_reader:
                buffer_reader.seek(result.storage_data.offset)
                loaded_tensor = _load_tensor_maybe_optimized(buffer_reader.read(result.storage_data.length))
                assert torch.equal(loaded_tensor, expected_tensor)

    class TestWriteMetadata:
        @staticmethod
        def _get_test_metadata():
            return torchdistmeta.Metadata(
                state_dict_metadata={
                    "model.layer1.weight": torchdistmeta.TensorStorageMetadata(
                        properties=torchdistmeta.TensorProperties(dtype=torch.float32, layout=torch.strided),
                        size=torch.Size([10, 20]),
                        chunks=[],
                    ),
                    "model.layer2.bias": torchdistmeta.TensorStorageMetadata(
                        properties=torchdistmeta.TensorProperties(dtype=torch.float16, layout=torch.strided),
                        size=torch.Size([30]),
                        chunks=[],
                    ),
                    "some_bytes": torchdistmeta.BytesStorageMetadata(),
                },
                storage_data={
                    "model.layer1.weight": {
                        "relative_path": "data_0_src0.distcp",
                        "offset": 0,
                        "length": 800,
                    },
                    "model.layer2.bias": {
                        "relative_path": "data_1_src0.distcp",
                        "offset": 100,
                        "length": 60,
                    },
                    "some_bytes": {
                        "relative_path": "data_1_src0.distcp",
                        "offset": 160,
                        "length": 20,
                    },
                },
            )

        def test_write_metadata_default_name(self, temp_dir_path, chkpt_object_manager, replication_manager):
            # Given
            saver = DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=lambda: 0,
                local_rank_getter=lambda: 0,
                global_barrier_func=lambda: None,
                ckpt_obj_manager=chkpt_object_manager,  # Not needed for this test
                replication_manager=replication_manager,
            )
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_1")
            metadata = self._get_test_metadata()

            # When
            saver.write_metadata(checkpoint_id, metadata)

            # Then
            metadata_path = os.path.join(checkpoint_id.data, ".metadata")
            assert os.path.exists(metadata_path)

            with open(metadata_path, "rb") as f:
                loaded_metadata = pickle.load(f)
            assert loaded_metadata == metadata

            # Check that the tmp file is not there
            assert not os.path.exists(metadata_path + ".tmp")

        def test_write_metadata_custom_name(self, temp_dir_path, chkpt_object_manager, replication_manager):
            # Given
            saver = DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=lambda: 0,
                local_rank_getter=lambda: 0,
                global_barrier_func=lambda: None,
                ckpt_obj_manager=chkpt_object_manager,  # Not needed for this test
                replication_manager=replication_manager,
            )
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_2")
            metadata = self._get_test_metadata()
            custom_name = "custom.meta"

            # When
            saver.write_metadata(checkpoint_id, metadata, md_object_name=custom_name)

            # Then
            metadata_path = os.path.join(checkpoint_id.data, custom_name)
            assert os.path.exists(metadata_path)

            with open(metadata_path, "rb") as f:
                loaded_metadata = pickle.load(f)
            assert loaded_metadata == metadata
            assert not os.path.exists(metadata_path + ".tmp")

        def test_write_metadata_error_when_file_exists_as_dir(
            self,
            temp_dir_path,
            chkpt_object_manager,
            replication_manager,
        ):
            # Given
            saver = DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=lambda: 0,
                local_rank_getter=lambda: 0,
                global_barrier_func=lambda: None,
                ckpt_obj_manager=chkpt_object_manager,  # Not needed for this test
                replication_manager=replication_manager,
            )
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_3")
            metadata = self._get_test_metadata()

            # Simulate an error during os.rename by creating a directory with the same name
            metadata_path = os.path.join(checkpoint_id.data, ".metadata")
            os.makedirs(metadata_path)

            # When/Then
            with pytest.raises(OSError):
                saver.write_metadata(checkpoint_id, metadata)

            # Check that the tmp file is retained, to aid with debugging
            assert os.path.exists(metadata_path + ".tmp")

        def test_write_metadata_error_when_write_fails(
            self,
            mocker,
            temp_dir_path,
            chkpt_object_manager,
            replication_manager,
        ):
            # Given
            saver = DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=lambda: 0,
                local_rank_getter=lambda: 0,
                global_barrier_func=lambda: None,
                ckpt_obj_manager=chkpt_object_manager,  # Not needed for this test
                replication_manager=replication_manager,
            )
            checkpoint_id = CheckpointContainerId(f"{temp_dir_path}/checkpoint_write_fail")
            metadata = self._get_test_metadata()
            metadata_path = os.path.join(checkpoint_id.data, ".metadata")
            # Mock the write operation to raise an exception
            mocker.patch("pickle.dump", side_effect=Exception("Simulated write error"))

            # When/Then
            with pytest.raises(Exception, match="Simulated write error"):
                saver.write_metadata(checkpoint_id, metadata)

            # Check that the tmp file exists because the write started
            assert os.path.exists(metadata_path + ".tmp")
            # Check that the final metadata file does not exist because rename was not reached
            assert not os.path.exists(metadata_path)

    class TestRemoveOlderCheckpoints:
        @pytest.fixture
        def saver(self, chkpt_object_manager, replication_manager):
            return DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=lambda: 0,
                local_rank_getter=lambda: 0,
                global_barrier_func=lambda: None,
                ckpt_obj_manager=chkpt_object_manager,
                replication_manager=replication_manager,
            )

        def test_remove_older_checkpoints_some_older_some_newer(self, saver, temp_dir_path):
            # Given
            checkpoint_base_dir = os.path.join(temp_dir_path, "checkpoints")
            os.makedirs(checkpoint_base_dir)
            checkpoint_dirs = [
                os.path.join(checkpoint_base_dir, "step-100_ckpt"),
                os.path.join(checkpoint_base_dir, "step-200_ckpt"),
                os.path.join(checkpoint_base_dir, "step-300_ckpt"),
                os.path.join(checkpoint_base_dir, "step-400_ckpt"),
            ]
            for d in checkpoint_dirs:
                os.makedirs(d)

            # When
            future = saver._remove_older_checkpoints(
                older_than=CheckpointContainerId(os.path.join(checkpoint_base_dir, "step-300_ckpt")),
            )
            future.wait()  # Wait for deletion to complete

            # Then
            assert not os.path.exists(os.path.join(checkpoint_base_dir, "step-100_ckpt"))
            assert not os.path.exists(os.path.join(checkpoint_base_dir, "step-200_ckpt"))
            assert os.path.exists(os.path.join(checkpoint_base_dir, "step-300_ckpt"))
            assert os.path.exists(os.path.join(checkpoint_base_dir, "step-400_ckpt"))

        def test_remove_older_checkpoints_no_older(self, saver, temp_dir_path):
            # Given
            checkpoint_base_dir = os.path.join(temp_dir_path, "checkpoints")
            os.makedirs(checkpoint_base_dir)
            checkpoint_dirs = [
                os.path.join(checkpoint_base_dir, "step-100_ckpt"),
                os.path.join(checkpoint_base_dir, "step-200_ckpt"),
                os.path.join(checkpoint_base_dir, "step-300_ckpt"),
            ]
            for d in checkpoint_dirs:
                os.makedirs(d)

            # When
            future = saver._remove_older_checkpoints(
                older_than=CheckpointContainerId(os.path.join(checkpoint_base_dir, "step-100_ckpt")),
            )
            future.wait()  # Wait for deletion to complete

            # Then
            assert os.path.exists(os.path.join(checkpoint_base_dir, "step-100_ckpt"))
            assert os.path.exists(os.path.join(checkpoint_base_dir, "step-200_ckpt"))
            assert os.path.exists(os.path.join(checkpoint_base_dir, "step-300_ckpt"))

        def test_remove_older_checkpoints_with_other_files(self, saver, temp_dir_path):
            # Given
            checkpoint_base_dir = os.path.join(temp_dir_path, "checkpoints")
            os.makedirs(checkpoint_base_dir)
            checkpoint_dirs = [
                os.path.join(checkpoint_base_dir, "step-100_ckpt"),
                os.path.join(checkpoint_base_dir, "step-200_ckpt"),
            ]
            for d in checkpoint_dirs:
                os.makedirs(d)

            other_file = os.path.join(checkpoint_base_dir, "some_file.txt")
            with open(other_file, "w") as f:
                f.write("hello")
            other_dir = os.path.join(checkpoint_base_dir, "not_a_checkpoint")
            os.makedirs(other_dir)

            # When
            future = saver._remove_older_checkpoints(
                older_than=CheckpointContainerId(os.path.join(checkpoint_base_dir, "step-300_ckpt")),
            )
            future.wait()

            # Then
            assert not os.path.exists(os.path.join(checkpoint_base_dir, "step-100_ckpt"))
            assert not os.path.exists(os.path.join(checkpoint_base_dir, "step-200_ckpt"))
            assert os.path.exists(other_file)
            assert os.path.exists(other_dir)

        def test_remove_older_checkpoints_empty_dir(self, saver, temp_dir_path):
            # Given
            checkpoint_base_dir = os.path.join(temp_dir_path, "checkpoints")
            os.makedirs(checkpoint_base_dir)

            # When
            future = saver._remove_older_checkpoints(
                older_than=CheckpointContainerId(os.path.join(checkpoint_base_dir, "step-100_ckpt")),
            )
            future.wait()

            # Then
            # No directories were deleted, and no error was raised.
            assert os.path.exists(checkpoint_base_dir)
            assert len(os.listdir(checkpoint_base_dir)) == 0

        def test_remove_older_checkpoints_delete_fails(self, saver, temp_dir_path, mocker):
            # Given
            checkpoint_base_dir = os.path.join(temp_dir_path, "checkpoints")
            os.makedirs(checkpoint_base_dir)
            checkpoint_dirs = [
                os.path.join(checkpoint_base_dir, "step-100_ckpt"),
                os.path.join(checkpoint_base_dir, "step-200_ckpt"),
            ]
            for d in checkpoint_dirs:
                os.makedirs(d)

            mocker.patch(
                "ml_flashpoint.core.checkpoint_saver.object_manager_ext.delete_directories_async",
                side_effect=IOError("Delete failed"),
            )

            # When/Then
            with pytest.raises(IOError, match="Delete failed"):
                # The future is returned, but the exception is raised immediately by the mock
                saver._remove_older_checkpoints(
                    older_than=CheckpointContainerId(os.path.join(checkpoint_base_dir, "step-300_ckpt")),
                )

            object_manager_ext.delete_directories_async.assert_called_once()
            args, _ = object_manager_ext.delete_directories_async.call_args
            actual_requested_deleted = set(args[0])
            expected_requested_deleted = {
                os.path.join(checkpoint_base_dir, "step-100_ckpt"),
                os.path.join(checkpoint_base_dir, "step-200_ckpt"),
            }
            assert actual_requested_deleted == expected_requested_deleted

        def test_remove_older_checkpoints_with_various_invalid_formats(self, saver, temp_dir_path):
            # Given
            checkpoint_base_dir = os.path.join(temp_dir_path, "checkpoints")
            os.makedirs(checkpoint_base_dir)
            checkpoint_dirs_to_delete = [
                os.path.join(checkpoint_base_dir, "step-100_ckpt"),
                os.path.join(checkpoint_base_dir, "step-200_ckpt"),
            ]
            checkpoint_dirs_to_keep = [
                os.path.join(checkpoint_base_dir, "step-300_ckpt"),
                os.path.join(checkpoint_base_dir, "step-400_ckpt"),
            ]
            invalid_format_dirs = [
                os.path.join(checkpoint_base_dir, "step-invalid_ckpt"),
                os.path.join(checkpoint_base_dir, "step-300a_ckpt"),
                os.path.join(checkpoint_base_dir, "not-a-step-300_ckpt"),
                os.path.join(checkpoint_base_dir, "step-300"),
                os.path.join(checkpoint_base_dir, "step-300_ckpt_extra"),
                os.path.join(checkpoint_base_dir, "step--100_ckpt"),  # Negative step
                os.path.join(checkpoint_base_dir, "step-3.14_ckpt"),  # Non-integer step
                os.path.join(checkpoint_base_dir, "latest"),
            ]
            for d in checkpoint_dirs_to_delete + checkpoint_dirs_to_keep + invalid_format_dirs:
                os.makedirs(d)

            # When
            future = saver._remove_older_checkpoints(
                older_than=CheckpointContainerId(os.path.join(checkpoint_base_dir, "step-300_ckpt")),
            )
            future.wait()

            # Then
            for d in checkpoint_dirs_to_delete:
                assert not os.path.exists(d)
            for d in checkpoint_dirs_to_keep:
                assert os.path.exists(d)
            for d in invalid_format_dirs:
                assert os.path.exists(d)

        def test_remove_older_checkpoints_older_than_is_invalid(self, saver, temp_dir_path, mocker):
            # Given
            checkpoint_base_dir = os.path.join(temp_dir_path, "checkpoints")
            os.makedirs(checkpoint_base_dir)
            checkpoint_dirs = [
                os.path.join(checkpoint_base_dir, "step-100_ckpt"),
                os.path.join(checkpoint_base_dir, "step-200_ckpt"),
            ]
            for d in checkpoint_dirs:
                os.makedirs(d)

            mocker.patch("ml_flashpoint.core.checkpoint_saver.object_manager_ext.delete_directories_async")

            # When
            future = saver._remove_older_checkpoints(
                older_than=CheckpointContainerId(os.path.join(checkpoint_base_dir, "invalid-step")),
            )

            # Then
            # No deletion is requested, so the future should complete immediately without error
            assert future is None
            assert os.path.exists(os.path.join(checkpoint_base_dir, "step-100_ckpt"))
            assert os.path.exists(os.path.join(checkpoint_base_dir, "step-200_ckpt"))

        @pytest.mark.parametrize(
            "older_than_step, all_checkpoint_steps, expected_deleted_steps",
            [
                # Basic cases
                (3, [1, 2, 3, 4, 5], [1, 2]),
                (10, [1, 5, 10, 15], [1, 5]),
                # Digit length crossovers
                (10, [1, 5, 9, 10, 11], [1, 5, 9]),
                (100, [1, 10, 99, 100, 101], [1, 10, 99]),
                (9, [1, 5, 8, 9, 10], [1, 5, 8]),
                (99, [1, 10, 98, 99, 100], [1, 10, 98]),
                # No older checkpoints
                (1, [1, 2, 3], []),
                (5, [5, 6, 7], []),
                # All older checkpoints
                (5, [1, 2, 3, 4], [1, 2, 3, 4]),
                # Mixed valid and invalid formats
                (
                    10,
                    [
                        1,
                        5,
                        9,
                        10,
                        11,
                        "invalid",
                        "step-abc_ckpt",
                        "step-10_ckpt_extra",
                    ],
                    [1, 5, 9],
                ),
            ],
        )
        def test_remove_older_checkpoints_digit_length_crossover(
            self,
            saver,
            temp_dir_path,
            older_than_step,
            all_checkpoint_steps,
            expected_deleted_steps,
        ):
            # Given
            checkpoint_base_dir = os.path.join(temp_dir_path, "checkpoints")
            os.makedirs(checkpoint_base_dir)

            # Create all checkpoint directories, including invalid ones
            all_created_dirs = []
            for step_val in all_checkpoint_steps:
                if isinstance(step_val, int):
                    dir_name = CheckpointContainerId.format_version_container(step_val)
                else:
                    dir_name = str(step_val)  # For invalid formats
                full_path = os.path.join(checkpoint_base_dir, dir_name)
                os.makedirs(full_path, exist_ok=True)
                all_created_dirs.append(full_path)

            older_than_checkpoint_id = CheckpointContainerId(
                os.path.join(checkpoint_base_dir, CheckpointContainerId.format_version_container(older_than_step))
            )

            # When
            future = saver._remove_older_checkpoints(older_than=older_than_checkpoint_id)
            if future is not None:
                future.wait()  # Wait for deletion to complete

            # Then
            expected_deleted_paths = {
                os.path.join(checkpoint_base_dir, CheckpointContainerId.format_version_container(s))
                for s in expected_deleted_steps
            }

            for d in all_created_dirs:
                if d in expected_deleted_paths:
                    assert not os.path.exists(d)
                else:
                    assert os.path.exists(d)

    def test_async_replicate_object(self, chkpt_object_manager, replication_manager, temp_dir_path):
        # Given
        saver = DefaultMLFlashpointCheckpointSaver(
            global_rank_getter=lambda: 0,
            local_rank_getter=lambda: 0,
            global_barrier_func=lambda: None,
            ckpt_obj_manager=chkpt_object_manager,
            replication_manager=replication_manager,
        )
        object_id = CheckpointObjectId.from_container(CheckpointContainerId(f"{temp_dir_path}/c1"), "obj1")

        # Create a real file using the manager to ensure proper format
        os.makedirs(os.path.dirname(object_id.data), exist_ok=True)
        with chkpt_object_manager.create_buffer(object_id, 1024) as f:
            f.write(b"test_async_replicate_object data")

        expected_futures = [concurrent.futures.Future(), concurrent.futures.Future(), concurrent.futures.Future()]
        replication_manager.async_replicate.return_value = expected_futures

        # When
        actual_futures = saver.async_replicate_object(object_id)

        # Then
        assert actual_futures is expected_futures

        assert replication_manager.async_replicate.call_count == 1
        args, _ = replication_manager.async_replicate.call_args
        buffer_io = args[0]
        assert isinstance(buffer_io, BufferIO)
        assert buffer_io.buffer_obj.get_id() == str(object_id)
        assert buffer_io.read() == b"test_async_replicate_object data"
        buffer_io.close()
