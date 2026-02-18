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
from unittest.mock import MagicMock

import pytest
import torch
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.strategies.async_utils import (
    AsyncRequest as MegatronAsyncRequest,
)
from megatron.core.dist_checkpointing.strategies.common import COMMON_STATE_FNAME, TorchCommonLoadStrategy
from nemo.lightning.io.pl import MegatronCheckpointIO

from ml_flashpoint.adapter.nemo.checkpoint_io import (
    MLFlashpointAsyncFinalizableCheckpointIO,
    MLFlashpointCheckpointIO,
    _init_pool_in_worker,
    _is_ml_flashpoint_checkpoint,
    _teardown_pool_in_worker,
)
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import (
    CheckpointObjectManager,
)
from ml_flashpoint.core.checkpoint_id_types import (
    CheckpointContainerId,
)


@pytest.fixture
def checkpoint_io_components(tmp_path, mocker):
    """Creates mocks and a MLFlashpointCheckpointIO instance for testing."""
    mock_alt_checkpoint_io = mocker.MagicMock(name="alt_checkpoint_io")

    # Use a real CheckpointObjectManager
    real_chkpt_obj_manager = CheckpointObjectManager()

    mock_save_strategy = mocker.MagicMock(name="save_strategy")
    mock_load_strategy = mocker.MagicMock(name="load_strategy")
    flashpoint_base_path = str(tmp_path / "test_base_path")

    checkpoint_io = MLFlashpointCheckpointIO(
        flashpoint_base_path=flashpoint_base_path,
        alt_checkpoint_io=mock_alt_checkpoint_io,
        chkpt_obj_manager=real_chkpt_obj_manager,
        save_strategy=mock_save_strategy,
        load_strategy=mock_load_strategy,
        trainer=mocker.MagicMock(),
        async_save=True,  # Default to async for most tests
    )

    components = {
        "checkpoint_io": checkpoint_io,
        "base_path": flashpoint_base_path,
        "alt_checkpoint_io": mock_alt_checkpoint_io,
        "chkpt_obj_manager": real_chkpt_obj_manager,  # Return real object
        "save_strategy": mock_save_strategy,
        "load_strategy": mock_load_strategy,
    }

    # Ensure each value in the dict is not None as a sanity check.
    for value in components.values():
        assert value is not None

    return components


def test_is_ml_flashpoint_checkpoint_true(checkpoint_io_components):
    """Tests that _is_ml_flashpoint_checkpoint returns True for an MLF path."""
    # Given
    checkpoint_io = checkpoint_io_components["checkpoint_io"]
    base_path = checkpoint_io_components["base_path"]
    ckpt_version_path = base_path + "/checkpoint1"

    # When
    result = _is_ml_flashpoint_checkpoint(checkpoint_io.flashpoint_base_dir, ckpt_version_path)

    # Then
    assert result is True


def test_is_ml_flashpoint_checkpoint_false(checkpoint_io_components, tmp_path):
    """Tests that _is_ml_flashpoint_checkpoint returns False for a non-MLF path."""
    # Given
    checkpoint_io = checkpoint_io_components["checkpoint_io"]
    ckpt_version_path = str(tmp_path / "different_path")

    # When
    result = _is_ml_flashpoint_checkpoint(checkpoint_io.flashpoint_base_dir, ckpt_version_path)

    # Then
    assert result is False


class TestMLFlashpointCheckpointIO:
    """Tests for the MLFlashpointCheckpointIO class."""

    def test_initializer(self, checkpoint_io_components, tmp_path, mocker):
        """Tests that dependency imports and class instantiation are successful."""
        # Given
        checkpoint_io = MLFlashpointCheckpointIO(
            flashpoint_base_path=checkpoint_io_components["base_path"],
            alt_checkpoint_io=checkpoint_io_components["alt_checkpoint_io"],
            chkpt_obj_manager=checkpoint_io_components["chkpt_obj_manager"],
            save_strategy=checkpoint_io_components["save_strategy"],
            load_strategy=checkpoint_io_components["load_strategy"],
            trainer=mocker.MagicMock(),
            async_save=False,  # Default to async for most tests
        )

        # Then
        assert checkpoint_io.flashpoint_base_dir == CheckpointContainerId(checkpoint_io_components["base_path"])
        assert checkpoint_io.fallback_checkpoint_io is checkpoint_io_components["alt_checkpoint_io"]
        assert checkpoint_io.chkpt_obj_manager is checkpoint_io_components["chkpt_obj_manager"]
        assert checkpoint_io.save_strategy is checkpoint_io_components["save_strategy"]
        assert checkpoint_io.load_strategy is checkpoint_io_components["load_strategy"]
        assert not checkpoint_io.async_save

    def test_initializer_string_base_path(self, mocker):
        """Tests that a plain string can be used for the base path."""
        # Given
        flashpoint_base_path = "/test_base_path"

        # When
        checkpoint_io = MLFlashpointCheckpointIO(
            flashpoint_base_path=flashpoint_base_path,
            alt_checkpoint_io=mocker.MagicMock(),
            chkpt_obj_manager=mocker.MagicMock(),
            save_strategy=mocker.MagicMock(),
            load_strategy=mocker.MagicMock(),
            trainer=mocker.MagicMock(),
        )

        # Then
        assert checkpoint_io.flashpoint_base_dir == CheckpointContainerId(flashpoint_base_path)

    def test_initialize_base_dir_with_container_id_works(self, mocker):
        """Tests that a CheckpointContainerId can be used for the base path."""
        # Given
        flashpoint_base_path = CheckpointContainerId("/test_base_path")

        # When
        checkpoint_io = MLFlashpointCheckpointIO(
            flashpoint_base_path=flashpoint_base_path,
            alt_checkpoint_io=mocker.MagicMock(),
            chkpt_obj_manager=mocker.MagicMock(),
            save_strategy=mocker.MagicMock(),
            load_strategy=mocker.MagicMock(),
            trainer=mocker.MagicMock(),
        )

        # Then
        assert checkpoint_io.flashpoint_base_dir == flashpoint_base_path

    def test_save_checkpoint_fallback(self, checkpoint_io_components, tmp_path, mocker):
        """Tests that saving falls back to the alternative IO when the path is not an MLF path."""
        # Given
        checkpoint_io = checkpoint_io_components["checkpoint_io"]
        alt_checkpoint_io = checkpoint_io_components["alt_checkpoint_io"]
        checkpoint = {"model": torch.nn.Linear(2, 2)}
        ckpt_version_path = str(tmp_path / "diff_path")

        expected_return = mocker.MagicMock()
        alt_checkpoint_io.save_checkpoint.return_value = expected_return

        # When
        result = checkpoint_io.save_checkpoint(checkpoint, ckpt_version_path)

        # Then
        alt_checkpoint_io.save_checkpoint.assert_called_once_with(checkpoint, ckpt_version_path)
        assert result is expected_return

    def test_save_ml_flashpoint_checkpoint_writes_common_state_dict(self, checkpoint_io_components, mocker):
        """Tests that the common state dict is written to a file during an MLF save."""
        # Given
        mock_torch_distributed = mocker.patch("ml_flashpoint.adapter.megatron.save_utils.torch.distributed")
        mock_torch_distributed.get_node_local_rank.return_value = 0
        checkpoint_io = checkpoint_io_components["checkpoint_io"]
        save_strategy = checkpoint_io_components["save_strategy"]
        base_path = checkpoint_io_components["base_path"]
        ckpt_version_path = base_path + "/checkpoint_common_test"

        # Define a sample common_state_dict
        common_state_dict = {"common_key_1": "common_value_1", "common_key_2": 42}
        checkpoint = {"common_state_dict": common_state_dict}

        # Mock save_preprocess to return an empty sharded_state_dict and our common_state_dict
        mocker.patch(
            "ml_flashpoint.adapter.megatron.save_utils.mcore_state_dict_utils.save_preprocess",
            return_value=({}, common_state_dict),
        )

        # Mock async_save to prevent actual saving of sharded data, as we're only testing common_state_dict
        mock_async_request = mocker.MagicMock()
        save_strategy.async_save.return_value = mock_async_request

        expected_common_state_filename = COMMON_STATE_FNAME
        common_state_file_path = os.path.join(ckpt_version_path, expected_common_state_filename)

        # Mock _save_context to prevent side effects
        mocker.patch.object(checkpoint_io, "_save_context")

        # When
        checkpoint_io.save_checkpoint(checkpoint, ckpt_version_path)

        # Then
        # Verify the common.pt file exists and is not empty
        assert os.path.exists(common_state_file_path)
        assert os.path.getsize(common_state_file_path) > 0

        # Load the saved common_state_dict and verify its content
        loaded_common_state_dict = torch.load(common_state_file_path)
        assert loaded_common_state_dict == common_state_dict

    def test_save_ml_flashpoint_checkpoint_async_success(self, checkpoint_io_components, mocker):
        """Tests a successful asynchronous MLF save."""
        # Given
        mock_torch_distributed = mocker.patch("ml_flashpoint.adapter.megatron.save_utils.torch.distributed")
        mock_torch_distributed.get_node_local_rank.return_value = 0
        checkpoint_io = checkpoint_io_components["checkpoint_io"]
        save_strategy = checkpoint_io_components["save_strategy"]
        base_path = checkpoint_io_components["base_path"]
        ckpt_version_path = base_path + "/checkpoint1"

        my_test_data = {"config": "test_value", "step": 100}
        sharded_state_dict = {
            "model": ShardedObject(
                key="model",
                data=my_test_data,
                global_shape=(2, 2),
                global_offset=(0, 0),
                replica_id=0,
            )
        }
        common_state_dict = {"common": "data"}
        checkpoint = {
            "sharded_state_dict": sharded_state_dict,
            "common_state_dict": common_state_dict,
        }

        mock_save_preprocess = mocker.patch(
            "ml_flashpoint.adapter.megatron.save_utils.mcore_state_dict_utils.save_preprocess",
            return_value=(sharded_state_dict, common_state_dict),
        )
        mock_torch_save = mocker.patch("ml_flashpoint.adapter.megatron.save_utils.torch.save")
        expected_common_state_filename = COMMON_STATE_FNAME

        mock_async_request = mocker.MagicMock()
        save_strategy.async_save.return_value = mock_async_request

        # Mock _save_context to prevent side effects
        mocker.patch.object(checkpoint_io, "_save_context")

        # When
        result = checkpoint_io.save_checkpoint(checkpoint, ckpt_version_path)

        # Then
        mock_save_preprocess.assert_called_once_with(checkpoint)
        save_strategy.async_save.assert_called_once_with(
            sharded_state_dict=sharded_state_dict,
            checkpoint_dir=ckpt_version_path,
        )
        mock_torch_save.assert_called_once_with(
            common_state_dict,
            os.path.join(ckpt_version_path, expected_common_state_filename),
        )
        assert result is mock_async_request

    def test_save_ml_flashpoint_checkpoint_sync_success(self, tmp_path, mocker):
        """Tests a successful synchronous MLF save."""
        # Given
        mock_torch_distributed = mocker.patch("ml_flashpoint.adapter.megatron.save_utils.torch.distributed")
        mock_torch_distributed.get_node_local_rank.return_value = 0
        mock_alt_checkpoint_io = mocker.MagicMock()
        mock_chkpt_obj_manager = mocker.MagicMock()
        mock_save_strategy = mocker.MagicMock()
        mock_load_strategy = mocker.MagicMock()
        flashpoint_base_path = str(tmp_path / "test_base_path")

        # Instantiate with async_save=False
        checkpoint_io = MLFlashpointCheckpointIO(
            flashpoint_base_path=flashpoint_base_path,
            alt_checkpoint_io=mock_alt_checkpoint_io,
            chkpt_obj_manager=mock_chkpt_obj_manager,
            save_strategy=mock_save_strategy,
            load_strategy=mock_load_strategy,
            trainer=mocker.MagicMock(),
            async_save=False,
        )

        ckpt_version_path = flashpoint_base_path + "/checkpoint1"

        my_test_data = {"config": "test_value", "step": 100}
        sharded_state_dict = {
            "model": ShardedObject(
                key="model",
                data=my_test_data,
                global_shape=(2, 2),
                global_offset=(0, 0),
                replica_id=0,
            )
        }
        common_state_dict = {"common": "data"}
        checkpoint = {
            "sharded_state_dict": sharded_state_dict,
            "common_state_dict": common_state_dict,
        }

        mock_save_preprocess = mocker.patch(
            "ml_flashpoint.adapter.megatron.save_utils.mcore_state_dict_utils.save_preprocess",
            return_value=(sharded_state_dict, common_state_dict),
        )
        mock_torch_save = mocker.patch("ml_flashpoint.adapter.megatron.save_utils.torch.save")
        expected_common_state_filename = COMMON_STATE_FNAME

        # Mock _save_context to prevent side effects
        mocker.patch.object(checkpoint_io, "_save_context")

        # When
        result = checkpoint_io.save_checkpoint(checkpoint, ckpt_version_path)

        # Then
        mock_save_preprocess.assert_called_once_with(checkpoint)
        mock_save_strategy.save.assert_called_once_with(
            sharded_state_dict=sharded_state_dict,
            checkpoint_dir=ckpt_version_path,
        )
        mock_torch_save.assert_called_once_with(
            common_state_dict,
            os.path.join(ckpt_version_path, expected_common_state_filename),
        )
        assert result is None

    def test_save_ml_flashpoint_checkpoint_async_failure_returns_none(self, checkpoint_io_components, mocker):
        """Tests that an async MLF save failure returns None."""
        # Given
        mock_torch_distributed = mocker.patch("ml_flashpoint.adapter.megatron.save_utils.torch.distributed")
        mock_torch_distributed.get_node_local_rank.return_value = 0
        checkpoint_io = checkpoint_io_components["checkpoint_io"]
        save_strategy = checkpoint_io_components["save_strategy"]
        base_path = checkpoint_io_components["base_path"]
        ckpt_version_path = base_path + "/checkpoint1"

        my_test_data = {"config": "test_value", "step": 100}
        sharded_state_dict = {
            "model": ShardedObject(
                key="model",
                data=my_test_data,
                global_shape=(2, 2),
                global_offset=(0, 0),
                replica_id=0,
            )
        }
        common_state_dict = {"common": "data"}
        checkpoint = {
            "sharded_state_dict": sharded_state_dict,
            "common_state_dict": common_state_dict,
        }

        mocker.patch(
            "ml_flashpoint.adapter.megatron.save_utils.mcore_state_dict_utils.save_preprocess",
            return_value=(sharded_state_dict, common_state_dict),
        )
        mocker.patch("ml_flashpoint.adapter.megatron.save_utils.torch.save")

        # Mock save strategy to fail
        test_exception = ValueError("Test async save failure")
        save_strategy.async_save.side_effect = test_exception

        # Mock _save_context to prevent side effects
        mocker.patch.object(checkpoint_io, "_save_context")

        # When
        result = checkpoint_io.save_checkpoint(checkpoint, ckpt_version_path)

        # Then
        assert result is None
        save_strategy.async_save.assert_called_once()

    def test_save_ml_flashpoint_checkpoint_sync_failure_returns_none(self, tmp_path, mocker):
        """Tests that a sync MLF save failure returns None."""
        # Given
        mock_torch_distributed = mocker.patch("ml_flashpoint.adapter.megatron.save_utils.torch.distributed")
        mock_torch_distributed.get_node_local_rank.return_value = 0
        mock_alt_checkpoint_io = mocker.MagicMock()
        mock_chkpt_obj_manager = mocker.MagicMock()
        mock_save_strategy = mocker.MagicMock()
        mock_load_strategy = mocker.MagicMock()
        flashpoint_base_path = str(tmp_path / "test_base_path")

        checkpoint_io = MLFlashpointCheckpointIO(
            flashpoint_base_path=flashpoint_base_path,
            alt_checkpoint_io=mock_alt_checkpoint_io,
            chkpt_obj_manager=mock_chkpt_obj_manager,
            save_strategy=mock_save_strategy,
            load_strategy=mock_load_strategy,
            trainer=mocker.MagicMock(),
            async_save=False,
        )
        ckpt_version_path = flashpoint_base_path + "/checkpoint1"

        my_test_data = {"config": "test_value", "step": 100}
        sharded_state_dict = {
            "model": ShardedObject(
                key="model",
                data=my_test_data,
                global_shape=(2, 2),
                global_offset=(0, 0),
                replica_id=0,
            )
        }
        common_state_dict = {"common": "data"}
        checkpoint = {
            "sharded_state_dict": sharded_state_dict,
            "common_state_dict": common_state_dict,
        }

        mocker.patch(
            "ml_flashpoint.adapter.megatron.save_utils.mcore_state_dict_utils.save_preprocess",
            return_value=(sharded_state_dict, common_state_dict),
        )
        mocker.patch("ml_flashpoint.adapter.megatron.save_utils.torch.save")

        # Mock save strategy to fail
        test_exception = ValueError("Test sync save failure")
        mock_save_strategy.save.side_effect = test_exception

        # Mock _save_context to prevent side effects
        mocker.patch.object(checkpoint_io, "_save_context")

        # When
        result = checkpoint_io.save_checkpoint(checkpoint, ckpt_version_path)

        # Then
        assert result is None
        mock_save_strategy.save.assert_called_once()

    @pytest.mark.parametrize(
        "node_local_rank,should_save",
        [
            (0, True),
            (1, False),
            (5, False),
        ],
    )
    def test_save_ml_flashpoint_checkpoint_common_state_dict_rank_handling(
        self, checkpoint_io_components, mocker, node_local_rank, should_save
    ):
        """Tests that common_state_dict is saved only on rank 0."""
        # Given
        mock_torch_distributed = mocker.patch("ml_flashpoint.adapter.megatron.save_utils.torch.distributed")
        mock_torch_distributed.get_node_local_rank.return_value = node_local_rank
        mock_torch_distributed.get_rank.return_value = 0  # Ensure global rank 0 to avoid context propagation logic
        checkpoint_io = checkpoint_io_components["checkpoint_io"]
        save_strategy = checkpoint_io_components["save_strategy"]
        base_path = checkpoint_io_components["base_path"]
        ckpt_version_path = base_path + f"/checkpoint_common_test_rank{node_local_rank}"

        common_state_dict = {"common_key": "common_value"}
        checkpoint = {"common_state_dict": common_state_dict}

        mocker.patch(
            "ml_flashpoint.adapter.megatron.save_utils.mcore_state_dict_utils.save_preprocess",
            return_value=({}, common_state_dict),
        )
        mock_torch_save = mocker.patch("ml_flashpoint.adapter.megatron.save_utils.torch.save")
        mock_os_makedirs = mocker.patch("ml_flashpoint.adapter.nemo.checkpoint_io.os.makedirs")
        save_strategy.async_save.return_value = mocker.MagicMock()

        # Mock _save_context to prevent side effects
        mocker.patch.object(checkpoint_io, "_save_context")

        # When
        checkpoint_io.save_checkpoint(checkpoint, ckpt_version_path)

        # Then
        if should_save:
            mock_os_makedirs.assert_called_once_with(ckpt_version_path, exist_ok=True)
            mock_torch_save.assert_called_once_with(
                common_state_dict, os.path.join(ckpt_version_path, COMMON_STATE_FNAME)
            )
        else:
            mock_os_makedirs.assert_not_called()
            mock_torch_save.assert_not_called()

    def test_load_checkpoint_fallback(self, checkpoint_io_components, tmp_path):
        """Tests that loading falls back to the alternative IO when the path is not an MLF path."""
        # Given
        checkpoint_io = checkpoint_io_components["checkpoint_io"]
        alt_checkpoint_io = checkpoint_io_components["alt_checkpoint_io"]
        ckpt_version_path = str(tmp_path / "diff_path")
        expected_checkpoint = {"model": "fallback_data"}
        alt_checkpoint_io.load_checkpoint.return_value = expected_checkpoint
        sharded_state_dict = {"key": "value"}

        # When
        result = checkpoint_io.load_checkpoint(ckpt_version_path, sharded_state_dict=sharded_state_dict)

        # Then
        alt_checkpoint_io.load_checkpoint.assert_called_once_with(
            ckpt_version_path, sharded_state_dict=sharded_state_dict, map_location=None
        )
        assert result is expected_checkpoint

    def test_load_checkpoint_fallback_passes_sharded_state_dict(self, checkpoint_io_components, tmp_path):
        """Tests that loading falls back to the alternative IO and passes sharded_state_dict."""
        # Given
        checkpoint_io = checkpoint_io_components["checkpoint_io"]
        alt_checkpoint_io = checkpoint_io_components["alt_checkpoint_io"]
        ckpt_version_path = str(tmp_path / "diff_path")
        expected_checkpoint = {"model": "fallback_data"}
        alt_checkpoint_io.load_checkpoint.return_value = expected_checkpoint
        sharded_state_dict = {"key": "value"}

        # When
        result = checkpoint_io.load_checkpoint(ckpt_version_path, sharded_state_dict=sharded_state_dict)

        # Then
        alt_checkpoint_io.load_checkpoint.assert_called_once_with(
            ckpt_version_path, sharded_state_dict=sharded_state_dict, map_location=None
        )
        assert result is expected_checkpoint

    def test_load_checkpoint_fallback_with_arbitrary_kwargs(self, checkpoint_io_components, tmp_path, mocker):
        """Tests that loading falls back to the alternative IO and passes arbitrary kwargs."""
        # Given
        checkpoint_io = checkpoint_io_components["checkpoint_io"]
        alt_checkpoint_io = checkpoint_io_components["alt_checkpoint_io"]
        ckpt_version_path = str(tmp_path / "diff_path")
        expected_checkpoint = {"model": "fallback_data"}
        alt_checkpoint_io.load_checkpoint.return_value = expected_checkpoint
        # Explicitly pass map_location and another arbitrary kwarg
        map_location_arg = "cpu"
        arbitrary_kwargs = {"strict": True, "another_arg": 123}

        # When
        result = checkpoint_io.load_checkpoint(ckpt_version_path, map_location=map_location_arg, **arbitrary_kwargs)

        # Then
        alt_checkpoint_io.load_checkpoint.assert_called_once_with(
            ckpt_version_path,
            sharded_state_dict=None,
            map_location=map_location_arg,
            **arbitrary_kwargs,
        )
        assert result is expected_checkpoint

    def test_load_ml_flashpoint_checkpoint_success(self, checkpoint_io_components, mocker):
        """Tests a successful MLF load."""
        # Given
        checkpoint_io = checkpoint_io_components["checkpoint_io"]
        load_strategy = checkpoint_io_components["load_strategy"]
        base_path = checkpoint_io_components["base_path"]
        ckpt_version_path = base_path + "/checkpoint1"
        sharded_state_dict = {"key": "value"}

        expected_checkpoint = {"model": torch.nn.Linear(2, 2)}
        mock_mcore_load = mocker.patch(
            "ml_flashpoint.adapter.nemo.checkpoint_io.mcore_dist_checkpointing.load",
            return_value=expected_checkpoint,
        )

        # When
        result = checkpoint_io.load_checkpoint(ckpt_version_path, sharded_state_dict=sharded_state_dict)

        # Then
        assert result is expected_checkpoint
        mock_mcore_load.assert_called_once_with(
            sharded_state_dict=sharded_state_dict,
            checkpoint_dir=ckpt_version_path,
            sharded_strategy=load_strategy,
            common_strategy=mocker.ANY,
        )
        # Additionally, verify the type of common_strategy if it's passed
        actual_call_kwargs = mock_mcore_load.call_args.kwargs
        assert "common_strategy" in actual_call_kwargs
        assert isinstance(actual_call_kwargs["common_strategy"], TorchCommonLoadStrategy)

    def test_load_ml_flashpoint_checkpoint_failure_raise_exception(self, checkpoint_io_components, mocker):
        """Tests that an MLF load failure raises an exception."""
        # Given
        checkpoint_io = checkpoint_io_components["checkpoint_io"]
        load_strategy = checkpoint_io_components["load_strategy"]
        alt_checkpoint_io = checkpoint_io_components["alt_checkpoint_io"]
        base_path = checkpoint_io_components["base_path"]
        ckpt_version_path = base_path + "/checkpoint1"

        # Mock mcore_dist_checkpointing.load to fail
        test_exception = ValueError("Test load failure")
        mock_mcore_load = mocker.patch(
            "ml_flashpoint.adapter.nemo.checkpoint_io.mcore_dist_checkpointing.load",
            side_effect=test_exception,
        )

        # When & Then
        # Check that the specific exception is raised
        with pytest.raises(ValueError, match="Test load failure") as exc_info:
            checkpoint_io.load_checkpoint(ckpt_version_path, None)

        assert exc_info.value is test_exception

        # Check that the load was attempted
        mock_mcore_load.assert_called_once_with(
            sharded_state_dict=None,
            checkpoint_dir=ckpt_version_path,
            sharded_strategy=load_strategy,
            common_strategy=mocker.ANY,
        )
        # Additionally, verify the type of common_strategy if it's passed
        actual_call_kwargs = mock_mcore_load.call_args.kwargs
        assert "common_strategy" in actual_call_kwargs
        assert isinstance(actual_call_kwargs["common_strategy"], TorchCommonLoadStrategy)
        # Check that the fallback IO was *not* called
        alt_checkpoint_io.load_checkpoint.assert_not_called()

    @pytest.mark.parametrize("cuda_is_initialized", [True, False])
    def test_load_ml_flashpoint_checkpoint_moves_to_cuda_if_initialized(
        self, checkpoint_io_components, mocker, cuda_is_initialized
    ):
        """Tests that _fix_tensors_device is called conditionally based on CUDA initialization."""
        # Given
        checkpoint_io = checkpoint_io_components["checkpoint_io"]
        base_path = checkpoint_io_components["base_path"]
        ckpt_version_path = base_path + "/checkpoint1"
        sharded_state_dict = {"key": "value"}
        expected_checkpoint = {"model": torch.nn.Linear(2, 2)}

        mocker.patch("torch.cuda.is_initialized", return_value=cuda_is_initialized)
        mock_fix_tensors_device = mocker.patch("ml_flashpoint.adapter.nemo.checkpoint_io._fix_tensors_device")
        mocker.patch(
            "ml_flashpoint.adapter.nemo.checkpoint_io.mcore_dist_checkpointing.load",
            return_value=expected_checkpoint,
        )

        # When
        result = checkpoint_io.load_checkpoint(ckpt_version_path, sharded_state_dict=sharded_state_dict)

        # Then
        assert result is expected_checkpoint
        if cuda_is_initialized:
            mock_fix_tensors_device.assert_called_once_with(expected_checkpoint)
        else:
            mock_fix_tensors_device.assert_not_called()

    def test_remove_checkpoint_is_not_flashpoint(self, checkpoint_io_components, tmp_path):
        """Tests that removing a non-MLF checkpoint falls back to the alternative IO."""
        # Given
        checkpoint_io = checkpoint_io_components["checkpoint_io"]
        alt_checkpoint_io = checkpoint_io_components["alt_checkpoint_io"]
        ckpt_version_path = str(tmp_path / "different_path")

        # When
        checkpoint_io.remove_checkpoint(ckpt_version_path)

        # Then
        alt_checkpoint_io.remove_checkpoint.assert_called_once_with(ckpt_version_path)

    def test_remove_checkpoint_is_flashpoint(self, checkpoint_io_components, mocker):
        """Tests that removing an MLF checkpoint uses the CheckpointObjectManager."""
        # Given
        checkpoint_io = checkpoint_io_components["checkpoint_io"]
        # chkpt_obj_manager is a real object, so we mock its method
        chkpt_obj_manager = checkpoint_io_components["chkpt_obj_manager"]
        base_path = checkpoint_io_components["base_path"]
        ckpt_version_path = base_path + "/checkpoint1"

        # Mock the delete_container method on the real object
        mocker.patch.object(chkpt_obj_manager, "delete_container")

        # When
        checkpoint_io.remove_checkpoint(ckpt_version_path)

        # Then
        # Assert that the real object's method was called correctly
        expected_path = str(CheckpointContainerId(ckpt_version_path))
        chkpt_obj_manager.delete_container.assert_called_once_with(CheckpointContainerId(expected_path))

    def test_save_context_async_execution(self, checkpoint_io_components, mocker):
        """Tests that _save_context executes in a separate thread and writes real files."""
        # Given
        checkpoint_io = checkpoint_io_components["checkpoint_io"]
        checkpoint_io.always_save_context = True
        base_path = checkpoint_io_components["base_path"]
        ckpt_version_path = base_path + "/checkpoint_context_test"

        # Mock distributed environment
        mocker.patch("ml_flashpoint.adapter.megatron.save_utils.torch.distributed.get_node_local_rank", return_value=0)
        mocker.patch(
            "ml_flashpoint.adapter.megatron.save_utils.torch.distributed.get_rank", return_value=1
        )  # Not rank 0 global, but rank 0 local

        # Mock broadcast: simulating receiving data from rank 0
        context_data = {"test_file.txt": b"content"}

        def mock_broadcast_object_list(obj_list, src=0):
            obj_list[0] = context_data

        mocker.patch(
            "ml_flashpoint.adapter.megatron.save_utils.torch.distributed.broadcast_object_list",
            side_effect=mock_broadcast_object_list,
        )

        # Mock mcore_state_dict_utils to avoid real distributed calls
        mocker.patch(
            "ml_flashpoint.adapter.megatron.save_utils.mcore_state_dict_utils.save_preprocess", return_value=({}, {})
        )

        # When
        thread = checkpoint_io._save_context(ckpt_version_path)
        assert thread is not None
        thread.join(timeout=5)

        expected_file = os.path.join(ckpt_version_path, "context", "test_file.txt")
        assert os.path.exists(expected_file), "Context file was not created by the background thread"
        with open(expected_file, "rb") as f:
            assert f.read() == b"content"

    def test_save_context_sync_on_rank0(self, checkpoint_io_components, mocker):
        """Tests that rank 0 _save_context generates context synchronously, reads it back, and broadcasts it."""
        # Given
        checkpoint_io = checkpoint_io_components["checkpoint_io"]
        checkpoint_io.always_save_context = True
        base_path = checkpoint_io_components["base_path"]
        ckpt_version_path = base_path + "/checkpoint_context_sync_test"

        # Mock distributed environment for rank 0
        mocker.patch("ml_flashpoint.adapter.megatron.save_utils.torch.distributed.get_node_local_rank", return_value=0)
        mocker.patch("ml_flashpoint.adapter.megatron.save_utils.torch.distributed.get_rank", return_value=0)

        # Mock TrainerContext to validly 'dump' files (we simulate this by writing files in side_effect)
        mock_trainer_context_cls = mocker.patch("ml_flashpoint.adapter.nemo.checkpoint_io.TrainerContext")
        mock_trainer_context = mock_trainer_context_cls.from_trainer.return_value

        def simulate_io_dump(path, **kwargs):
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "test_file.txt"), "wb") as f:
                f.write(b"content_on_disk")

        mock_trainer_context.io_dump.side_effect = simulate_io_dump

        # Mock threading.Thread to ensure it's NOT called
        mock_thread_cls = mocker.patch("ml_flashpoint.adapter.nemo.checkpoint_io.threading.Thread")

        # Mock mcore_state_dict_utils/torch.save/broadcast
        mocker.patch(
            "ml_flashpoint.adapter.megatron.save_utils.mcore_state_dict_utils.save_preprocess", return_value=({}, {})
        )
        mocker.patch("ml_flashpoint.adapter.megatron.save_utils.torch.save")
        mock_broadcast = mocker.patch(
            "ml_flashpoint.adapter.megatron.save_utils.torch.distributed.broadcast_object_list"
        )

        # We do NOT mock os.makedirs, os.walk, open. They run for real.

        # When
        checkpoint_io._save_context(ckpt_version_path)

        # Then
        # Check that io_dump was called
        mock_trainer_context.io_dump.assert_called_once()

        # Check that files were read and broadcasted
        # broadcast_object_list is called with [context_data]
        mock_broadcast.assert_called_once()
        args, _ = mock_broadcast.call_args
        object_list = args[0]
        assert len(object_list) == 1
        context_data = object_list[0]

        # Verify the content read from disk matches what we wrote
        assert "test_file.txt" in context_data
        assert context_data["test_file.txt"] == b"content_on_disk"

        # Check that NO thread was spawned
        mock_thread_cls.assert_not_called()


class TestMLFlashpointAsyncFinalizableCheckpointIO:
    """Parent test class for MLFlashpointAsyncFinalizableCheckpointIO."""

    class TestInit:
        """Test the __init__ method."""

        @pytest.fixture(autouse=True)
        def setup_mocks(self, mocker):
            self.mock_async_calls_queue_cls = mocker.patch("ml_flashpoint.adapter.nemo.checkpoint_io.AsyncCallsQueue")

        def test_successful_initialization(self, mocker):
            """Tests that the class initializes correctly with a valid CheckpointIO."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            # Mock the thread count needed for buffer pool init
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_mlf_queue = mocker.MagicMock()
            mock_alt_queue = mocker.MagicMock()

            self.mock_async_calls_queue_cls.side_effect = [mock_mlf_queue, mock_alt_queue]

            # When
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)

            # Then
            assert isinstance(instance, MLFlashpointAsyncFinalizableCheckpointIO)
            assert instance.checkpoint_io == mock_checkpoint_io
            assert instance._mlf_async_calls_queue == mock_mlf_queue
            assert instance._alt_async_calls_queue == mock_alt_queue

        def test_incompatible_checkpoint_io_type_raises_error(self, mocker):
            """Tests that a ValueError is raised for an incompatible CheckpointIO type."""
            # Given
            mock_checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)

            # When/Then
            with pytest.raises(ValueError):
                MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)

    class TestSaveCheckpoint:
        """Test the save_checkpoint method."""

        @pytest.fixture(autouse=True)
        def setup_mocks(self, mocker):
            self.mock_async_calls_queue_cls = mocker.patch("ml_flashpoint.adapter.nemo.checkpoint_io.AsyncCallsQueue")

        def test_save_ml_flashpoint_checkpoint(self, mocker):
            """Tests that an MLF checkpoint is scheduled to the correct queue."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            mock_checkpoint_io.trainer.global_rank = 0
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_checkpoint_io.flashpoint_base_dir = "/mlf/checkpoints"
            mock_mlf_queue = MagicMock()
            mock_alt_queue = MagicMock()
            self.mock_async_calls_queue_cls.side_effect = [mock_mlf_queue, mock_alt_queue]
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)
            mock_async_request = MagicMock(spec=MegatronAsyncRequest)
            mock_checkpoint_io.save_checkpoint.return_value = mock_async_request
            path = "/mlf/checkpoints/step-100"
            checkpoint = {"some": "data"}

            # When
            instance.save_checkpoint(checkpoint, path)

            # Then
            mock_checkpoint_io.save_checkpoint.assert_called_once_with(checkpoint, path, None)
            # Assert schedule_async_request was called for the save request (and potentially init)
            assert mock_mlf_queue.schedule_async_request.call_count >= 1
            # Verify the LAST call was for the save request
            assert mock_mlf_queue.schedule_async_request.call_args_list[-1][0][0] == mock_async_request
            mock_alt_queue.schedule_async_request.assert_not_called()

        def test_save_alternative_checkpoint(self, mocker):
            """Tests that a non-MLF checkpoint is scheduled to the alternative queue."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            mock_checkpoint_io.trainer.global_rank = 0
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_checkpoint_io.flashpoint_base_dir = "/mlf/checkpoints"
            mock_mlf_queue = mocker.MagicMock()
            mock_alt_queue = mocker.MagicMock()
            self.mock_async_calls_queue_cls.side_effect = [mock_mlf_queue, mock_alt_queue]
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)
            mock_async_request = mocker.MagicMock(spec=MegatronAsyncRequest)
            mock_checkpoint_io.save_checkpoint.return_value = mock_async_request
            path = "/other/checkpoints/step-100"
            checkpoint = {"some": "data"}

            # When
            instance.save_checkpoint(checkpoint, path)

            # Then
            mock_checkpoint_io.save_checkpoint.assert_called_once_with(checkpoint, path, None)
            mock_alt_queue.schedule_async_request.assert_called_once_with(mock_async_request)
            mock_mlf_queue.schedule_async_request.assert_not_called()

        def test_buffer_pool_init_scheduled_on_first_save(self, mocker):
            """Tests that BufferPool initialization is scheduled on the first MLF save."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            mock_checkpoint_io.trainer.global_rank = 0
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_checkpoint_io.flashpoint_base_dir = "/mlf/checkpoints"
            mock_mlf_queue = mocker.MagicMock()
            mock_alt_queue = mocker.MagicMock()
            self.mock_async_calls_queue_cls.side_effect = [mock_mlf_queue, mock_alt_queue]
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)
            mock_async_request = mocker.MagicMock(spec=MegatronAsyncRequest)
            mock_checkpoint_io.save_checkpoint.return_value = mock_async_request
            path = "/mlf/checkpoints/step-100"
            checkpoint = {"some": "data"}

            # When
            instance.save_checkpoint(checkpoint, path)

            # Then
            # Verify initialization was scheduled
            # It should be the first call to schedule_async_request
            first_call_args = mock_mlf_queue.schedule_async_request.call_args_list[0]
            scheduled_request = first_call_args[0][0]
            assert scheduled_request.async_fn == _init_pool_in_worker
            # Expecting (path, rank, num_buffers, size)
            # num_buffers = 3 * 1 (thread_count) = 3
            assert scheduled_request.async_fn_args == (
                "/mlf/checkpoints/buffer_pool",
                0,
                3,
                16000000000,
            )

        def test_buffer_pool_init_not_scheduled_on_subsequent_save(self, mocker):
            """Tests that BufferPool initialization is NOT scheduled on subsequent MLF saves."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            mock_checkpoint_io.trainer.global_rank = 0
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_checkpoint_io.flashpoint_base_dir = "/mlf/checkpoints"
            mock_mlf_queue = mocker.MagicMock()
            mock_alt_queue = mocker.MagicMock()
            self.mock_async_calls_queue_cls.side_effect = [mock_mlf_queue, mock_alt_queue]
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)
            mock_async_request = mocker.MagicMock(spec=MegatronAsyncRequest)
            mock_checkpoint_io.save_checkpoint.return_value = mock_async_request
            path1 = "/mlf/checkpoints/step-100"
            path2 = "/mlf/checkpoints/step-200"
            checkpoint = {"some": "data"}

            # When
            instance.save_checkpoint(checkpoint, path1)
            # Reset mocks to clear the first save calls
            mock_mlf_queue.schedule_async_request.reset_mock()
            instance.save_checkpoint(checkpoint, path2)

            # Then
            # Verify initialization was NOT scheduled again
            # Only the save request should be scheduled
            mock_mlf_queue.schedule_async_request.assert_called_once_with(mock_async_request)

        def test_save_with_external_finalize_fn(self, mocker):
            """Tests that an external finalize function is added to the async request."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            mock_checkpoint_io.trainer.global_rank = 0
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_checkpoint_io.flashpoint_base_dir = "/mlf/checkpoints"
            self.mock_async_calls_queue_cls.side_effect = [mocker.MagicMock(), mocker.MagicMock()]
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)
            mock_async_request = mocker.MagicMock(spec=MegatronAsyncRequest)
            mock_checkpoint_io.save_checkpoint.return_value = mock_async_request
            path = "/mlf/checkpoints/step-100"
            checkpoint = {"some": "data"}
            finalize_fn = mocker.MagicMock()
            storage_options = {"finalize_fn": finalize_fn}

            # When
            instance.save_checkpoint(checkpoint, path, storage_options)

            # Then
            mock_async_request.add_finalize_fn.assert_called_once_with(finalize_fn)

    class TestMaybeFinalizeSaveCheckpoint:
        """Test the maybe_finalize_save_checkpoint method."""

        @pytest.fixture(autouse=True)
        def setup_mocks(self, mocker):
            self.mock_async_calls_queue_cls = mocker.patch("ml_flashpoint.adapter.nemo.checkpoint_io.AsyncCallsQueue")

        def test_no_unfinalized_calls(self, mocker):
            """Tests that the method returns False when there are no calls to finalize."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            mock_checkpoint_io.trainer.global_rank = 0
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_mlf_queue = mocker.MagicMock()
            mock_alt_queue = mocker.MagicMock()
            self.mock_async_calls_queue_cls.side_effect = [mock_mlf_queue, mock_alt_queue]
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)
            mock_mlf_queue.get_num_unfinalized_calls.return_value = 0
            mock_alt_queue.get_num_unfinalized_calls.return_value = 0

            # When
            result = instance.maybe_finalize_save_checkpoint()

            # Then
            assert not result
            mock_mlf_queue.maybe_finalize_async_calls.assert_not_called()
            mock_alt_queue.maybe_finalize_async_calls.assert_not_called()

        def test_finalize_mlf_calls_only(self, mocker):
            """Tests that only MLF calls are finalized when only the MLF queue has calls."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            mock_checkpoint_io.trainer.global_rank = 0
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_mlf_queue = mocker.MagicMock()
            mock_alt_queue = mocker.MagicMock()
            self.mock_async_calls_queue_cls.side_effect = [mock_mlf_queue, mock_alt_queue]
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)
            mock_mlf_queue.get_num_unfinalized_calls.return_value = 1
            mock_alt_queue.get_num_unfinalized_calls.return_value = 0
            mock_mlf_queue.maybe_finalize_async_calls.return_value = [1]
            mock_alt_queue.maybe_finalize_async_calls.return_value = []

            # When
            result = instance.maybe_finalize_save_checkpoint()

            # Then
            assert result
            mock_mlf_queue.maybe_finalize_async_calls.assert_called_once_with(False)
            mock_alt_queue.maybe_finalize_async_calls.assert_called_once_with(False)

        def test_finalize_alt_calls_only(self, mocker):
            """Tests that only alternative calls are finalized when only the alternative queue has calls."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            mock_checkpoint_io.trainer.global_rank = 0
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_mlf_queue = mocker.MagicMock()
            mock_alt_queue = mocker.MagicMock()
            self.mock_async_calls_queue_cls.side_effect = [mock_mlf_queue, mock_alt_queue]
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)
            mock_mlf_queue.get_num_unfinalized_calls.return_value = 0
            mock_alt_queue.get_num_unfinalized_calls.return_value = 1
            mock_mlf_queue.maybe_finalize_async_calls.return_value = []
            mock_alt_queue.maybe_finalize_async_calls.return_value = [1]

            # When
            result = instance.maybe_finalize_save_checkpoint()

            # Then
            assert result
            mock_mlf_queue.maybe_finalize_async_calls.assert_called_once_with(False)
            mock_alt_queue.maybe_finalize_async_calls.assert_called_once_with(False)

        def test_finalize_both_queues(self, mocker):
            """Tests that calls from both queues are finalized when both have calls."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            mock_checkpoint_io.trainer.global_rank = 0
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_mlf_queue = mocker.MagicMock()
            mock_alt_queue = mocker.MagicMock()
            self.mock_async_calls_queue_cls.side_effect = [mock_mlf_queue, mock_alt_queue]
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)
            mock_mlf_queue.get_num_unfinalized_calls.return_value = 1
            mock_alt_queue.get_num_unfinalized_calls.return_value = 1
            mock_mlf_queue.maybe_finalize_async_calls.return_value = [1]
            mock_alt_queue.maybe_finalize_async_calls.return_value = [1]

            # When
            result = instance.maybe_finalize_save_checkpoint(blocking=True)

            # Then
            assert result
            mock_mlf_queue.maybe_finalize_async_calls.assert_called_once_with(True)
            mock_alt_queue.maybe_finalize_async_calls.assert_called_once_with(True)

    class TestTeardown:
        """Test the teardown method."""

        @pytest.fixture(autouse=True)
        def setup_mocks(self, mocker):
            self.mock_async_calls_queue_cls = mocker.patch("ml_flashpoint.adapter.nemo.checkpoint_io.AsyncCallsQueue")
            self.mock_logger = mocker.patch("ml_flashpoint.adapter.nemo.checkpoint_io._LOGGER")

        def test_teardown_with_no_pending_saves(self, mocker):
            """Tests that no warning is logged when there are no pending saves."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            mock_checkpoint_io.trainer.global_rank = 0
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_mlf_queue = MagicMock()
            mock_alt_queue = MagicMock()
            self.mock_async_calls_queue_cls.side_effect = [mock_mlf_queue, mock_alt_queue]
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)
            mock_mlf_queue.get_num_unfinalized_calls.return_value = 0
            mock_alt_queue.get_num_unfinalized_calls.return_value = 0

            # When
            instance.teardown()

            # Then
            self.mock_logger.warning.assert_not_called()

        def test_teardown_with_pending_mlf_saves(self, mocker):
            """Tests that a warning is logged when there are pending MLF saves."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            mock_checkpoint_io.trainer.global_rank = 0
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_mlf_queue = mocker.MagicMock()
            mock_alt_queue = mocker.MagicMock()
            self.mock_async_calls_queue_cls.side_effect = [mock_mlf_queue, mock_alt_queue]
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)
            mock_mlf_queue.get_num_unfinalized_calls.return_value = 1
            mock_alt_queue.get_num_unfinalized_calls.return_value = 0

            # When
            instance.teardown()

            # Then
            self.mock_logger.warning.assert_called_once()

        def test_teardown_with_pending_alt_saves(self, mocker):
            """Tests that a warning is logged when there are pending alternative saves."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            mock_checkpoint_io.trainer.global_rank = 0
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_mlf_queue = mocker.MagicMock()
            mock_alt_queue = mocker.MagicMock()
            self.mock_async_calls_queue_cls.side_effect = [mock_mlf_queue, mock_alt_queue]
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)
            mock_mlf_queue.get_num_unfinalized_calls.return_value = 0
            mock_alt_queue.get_num_unfinalized_calls.return_value = 1

            # When
            instance.teardown()

            # Then
            self.mock_logger.warning.assert_called_once()

        def test_buffer_pool_teardown_scheduled(self, mocker):
            """Tests that BufferPool teardown is scheduled during teardown."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            mock_checkpoint_io.trainer.global_rank = 0
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_mlf_queue = mocker.MagicMock()
            mock_alt_queue = mocker.MagicMock()
            self.mock_async_calls_queue_cls.side_effect = [mock_mlf_queue, mock_alt_queue]
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)
            # simulate queue not closed
            mock_mlf_queue.__bool__.return_value = True
            mock_mlf_queue.get_num_unfinalized_calls.return_value = 0
            mock_alt_queue.get_num_unfinalized_calls.return_value = 0

            # When
            instance.teardown()

            # Then
            # Verify teardown was scheduled
            # We can't easily check for the exact AsyncRequest object because it's created inside teardown
            # But we can check if schedule_async_request was called with a request having the right function
            calls = mock_mlf_queue.schedule_async_request.call_args_list
            assert len(calls) > 0
            teardown_call = calls[0]
            scheduled_request = teardown_call[0][0]
            assert scheduled_request.async_fn == _teardown_pool_in_worker
            assert scheduled_request.async_fn_args == ()

    class TestIntegration:
        """Integration tests for MLFlashpointAsyncFinalizableCheckpointIO."""

        @pytest.fixture(autouse=True)
        def setup_mocks(self, mocker):
            self.mock_async_calls_queue_cls = mocker.patch("ml_flashpoint.adapter.nemo.checkpoint_io.AsyncCallsQueue")

        def test_full_lifecycle_mlf_checkpoint(self, mocker):
            """Tests the full lifecycle of saving and finalizing an MLF checkpoint."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            mock_checkpoint_io.trainer.global_rank = 0
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_checkpoint_io.flashpoint_base_dir = "/mlf/checkpoints"
            mock_mlf_queue = MagicMock()
            mock_alt_queue = MagicMock()
            self.mock_async_calls_queue_cls.side_effect = [mock_mlf_queue, mock_alt_queue]
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)
            mock_async_request = MagicMock(spec=MegatronAsyncRequest)
            mock_checkpoint_io.save_checkpoint.return_value = mock_async_request
            path = "/mlf/checkpoints/step-100"
            checkpoint = {"some": "data"}

            # When
            instance.save_checkpoint(checkpoint, path)
            mock_mlf_queue.maybe_finalize_async_calls.return_value = [0]
            instance.maybe_finalize_save_checkpoint()

            # Then
            # Assert schedule_async_request was called for the save request
            assert mock_mlf_queue.schedule_async_request.call_count >= 1
            # Verify the LAST call was for the save request
            assert mock_mlf_queue.schedule_async_request.call_args_list[-1][0][0] == mock_async_request
            mock_mlf_queue.maybe_finalize_async_calls.assert_called_once()

        def test_full_lifecycle_alt_checkpoint(self, mocker):
            """Tests the full lifecycle of saving and finalizing an alternative checkpoint."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            mock_checkpoint_io.trainer.global_rank = 0
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_checkpoint_io.flashpoint_base_dir = "/mlf/checkpoints"
            mock_mlf_queue = mocker.MagicMock()
            mock_alt_queue = mocker.MagicMock()
            self.mock_async_calls_queue_cls.side_effect = [mock_mlf_queue, mock_alt_queue]
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)
            mock_async_request = mocker.MagicMock(spec=MegatronAsyncRequest)
            mock_checkpoint_io.save_checkpoint.return_value = mock_async_request
            path = "/other/checkpoints/step-100"
            checkpoint = {"some": "data"}

            # When
            instance.save_checkpoint(checkpoint, path)
            mock_alt_queue.maybe_finalize_async_calls.return_value = [0]
            instance.maybe_finalize_save_checkpoint()

            # Then
            mock_alt_queue.schedule_async_request.assert_called_once_with(mock_async_request)
            mock_alt_queue.maybe_finalize_async_calls.assert_called_once()

        def test_interleaved_checkpoints_are_finalized_independently(self, mocker):
            """Tests that interleaved MLF and alternative checkpoints are finalized independently."""
            # Given
            mock_checkpoint_io = mocker.Mock(
                spec=MLFlashpointCheckpointIO,
                trainer=mocker.MagicMock(),
                save_strategy=mocker.MagicMock(),
                load_strategy=mocker.MagicMock(),
                chkpt_obj_manager=mocker.MagicMock(),
                fallback_checkpoint_io=mocker.MagicMock(),
                async_save=True,
                flashpoint_base_dir="/mlf/checkpoints",
            )
            mock_checkpoint_io.trainer.global_rank = 0
            mock_checkpoint_io.save_strategy._storage_writer._thread_count = 1
            mock_checkpoint_io.flashpoint_base_dir = "/mlf/checkpoints"
            mock_mlf_queue = mocker.MagicMock()
            mock_alt_queue = mocker.MagicMock()
            self.mock_async_calls_queue_cls.side_effect = [mock_mlf_queue, mock_alt_queue]
            instance = MLFlashpointAsyncFinalizableCheckpointIO(mock_checkpoint_io)

            mlf_request = mocker.MagicMock(spec=MegatronAsyncRequest)
            alt_request = mocker.MagicMock(spec=MegatronAsyncRequest)

            def save_side_effect(checkpoint, path, storage_options=None):
                if "mlf" in path:
                    return mlf_request
                return alt_request

            mock_checkpoint_io.save_checkpoint.side_effect = save_side_effect

            # When
            # Save one of each
            instance.save_checkpoint({}, "/mlf/checkpoints/1")
            instance.save_checkpoint({}, "/other/checkpoints/1")

            # Finalize only MLF
            mock_mlf_queue.get_num_unfinalized_calls.return_value = 1
            mock_alt_queue.get_num_unfinalized_calls.return_value = 1
            mock_mlf_queue.maybe_finalize_async_calls.return_value = [0]
            mock_alt_queue.maybe_finalize_async_calls.return_value = []
            instance.maybe_finalize_save_checkpoint()

            # Then
            # Assert schedule_async_request was called for mlf request
            calls = mock_mlf_queue.schedule_async_request.call_args_list
            assert any(call.args[0] == mlf_request for call in calls)
            mock_alt_queue.schedule_async_request.assert_called_once_with(alt_request)
            mock_mlf_queue.maybe_finalize_async_calls.assert_called_once()
            mock_alt_queue.maybe_finalize_async_calls.assert_called_once()
