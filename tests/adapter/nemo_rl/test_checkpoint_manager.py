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

pytest.importorskip("nemo_rl")

from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId


class MockPolicy:
    def __init__(self, mocker):
        self.save_checkpoint_called = False
        self.mcore_state = mocker.MagicMock()
        self.mcore_state.train_state.floating_point_operations_so_far = 123
        self.model = mocker.MagicMock(training=True)
        self.optimizer = mocker.MagicMock()
        self.scheduler = mocker.MagicMock()
        self.checkpointing_context = mocker.MagicMock()

    def save_checkpoint(self, weights_path, optimizer_path=None, tokenizer_path=None, **kwargs):
        self.save_checkpoint_called = True
        self.last_weights_path = weights_path


@pytest.fixture
def mock_base_checkpointer(mocker):
    checkpointer = mocker.MagicMock()
    checkpointer.checkpoint_dir = "/tmp/fake_dir"
    checkpointer.init_tmp_checkpoint.return_value = "/tmp/fake_dir/tmp_step_100"
    return checkpointer


@pytest.fixture
def mock_save_strategy(mocker):
    return mocker.MagicMock()


@pytest.fixture
def mock_checkpoint_loader(mocker):
    return mocker.MagicMock()


@pytest.fixture
def mlf_checkpoint_manager(mocker, mock_base_checkpointer, mock_save_strategy, mock_checkpoint_loader):
    from ml_flashpoint.adapter.nemo_rl.checkpoint_manager import MLFlashpointRLCheckpointManager

    manager = MLFlashpointRLCheckpointManager(
        base_checkpointer=mock_base_checkpointer,
        flashpoint_base_container="/test-mlf",
        standard_save_period=50,
        save_strategy=mock_save_strategy,
        checkpoint_loader=mock_checkpoint_loader,
    )
    return manager


def test_wrap_rl_components_with_mlflashpoint(mocker, mock_base_checkpointer, mock_save_strategy):
    """Test the wrap_rl_components_with_mlflashpoint utility."""
    from ml_flashpoint.adapter.nemo_rl.checkpoint_manager import MLFlashpointRLCheckpointManager
    from ml_flashpoint.adapter.nemo_rl.wrapper_util import wrap_rl_components_with_mlflashpoint

    # Given
    policy = MockPolicy(mocker)
    base_container = "/test-mlf"
    period = 50

    # When
    manager = wrap_rl_components_with_mlflashpoint(
        checkpointer=mock_base_checkpointer,
        policy=policy,
        flashpoint_base_container=base_container,
        standard_save_period=period,
        save_strategy=mock_save_strategy,
        checkpoint_loader=mock_checkpoint_loader,
    )

    # Then
    assert isinstance(manager, MLFlashpointRLCheckpointManager)
    assert manager.standard_save_period == period
    assert manager.flashpoint_base_container == CheckpointContainerId(base_container)
    assert manager.save_strategy == mock_save_strategy


def test_wrap_rl_components_raises_if_strategy_missing(mocker, mock_base_checkpointer):
    """Test that wrap_rl_components_with_mlflashpoint raises error if save_strategy is missing."""
    from ml_flashpoint.adapter.nemo_rl.wrapper_util import wrap_rl_components_with_mlflashpoint

    # Given
    policy = MockPolicy(mocker)

    # When/Then
    with pytest.raises(ValueError, match="save_strategy must be provided."):
        wrap_rl_components_with_mlflashpoint(
            checkpointer=mock_base_checkpointer,
            policy=policy,
            flashpoint_base_container="/tmp",
            standard_save_period=100,
            save_strategy=None,
            checkpoint_loader=mock_checkpoint_loader,
        )


def test_getattr_delegation_to_base_checkpointer(mlf_checkpoint_manager, mock_base_checkpointer):
    """Test that attributes not found on manager are delegated to base checkpointer."""
    # Given
    mock_base_checkpointer.some_custom_attr = "custom_value"
    mock_base_checkpointer.checkpoint_dir = "/base/dir"

    # When/Then
    assert mlf_checkpoint_manager.some_custom_attr == "custom_value"
    assert mlf_checkpoint_manager.checkpoint_dir == "/base/dir"


def test_getattr_raises_attribute_error_if_not_in_base(mlf_checkpoint_manager):
    """Test that getattr still raises AttributeError if not found in base."""
    # When/Then
    with pytest.raises(AttributeError):
        _ = mlf_checkpoint_manager.non_existent_attr


def test_standard_save_period_delegates_to_base_checkpointer(mocker, mlf_checkpoint_manager, mock_base_checkpointer):
    """Test that a standard save step (step % standard_save_period == 0) delegates to standard tools."""
    mock_save_local_aware = mocker.patch(
        "ml_flashpoint.adapter.nemo_rl.checkpoint_manager.save_local_aware_megatron_checkpoint"
    )
    # Given
    step = 200

    # When
    returned_path = mlf_checkpoint_manager.init_tmp_checkpoint(step, {})
    mlf_checkpoint_manager.finalize_checkpoint(returned_path)

    # Then
    mock_base_checkpointer.init_tmp_checkpoint.assert_called_once_with(step, {}, None)
    assert returned_path == "/tmp/fake_dir/tmp_step_100"
    mock_save_local_aware.assert_not_called()
    mock_base_checkpointer.finalize_checkpoint.assert_called_once_with(returned_path)


def test_mlf_save_period_invokes_mlflashpoint_save(mocker, mlf_checkpoint_manager, mock_base_checkpointer):
    """Test that an ML Flashpoint save step dynamically reroutes to ML Flashpoint logic."""
    mock_save_local_aware = mocker.patch(
        "ml_flashpoint.adapter.nemo_rl.checkpoint_manager.save_local_aware_megatron_checkpoint"
    )
    mock_makedirs = mocker.patch("os.makedirs")
    # Given
    step = 50

    # When
    returned_path = mlf_checkpoint_manager.init_tmp_checkpoint(step, {})
    mlf_checkpoint_manager.finalize_checkpoint(returned_path)

    # Then
    mock_base_checkpointer.init_tmp_checkpoint.assert_not_called()

    # Check that os.makedirs was called for the new path
    expected_path = os.path.join("/test-mlf", f"step-{step}_ckpt")
    mock_makedirs.assert_called_with(expected_path, exist_ok=True)
    assert returned_path == expected_path

    mock_save_local_aware.assert_not_called() # Interception is gone!
    mock_base_checkpointer.finalize_checkpoint.assert_not_called()




def test_get_best_checkpoint_path_delegates(mlf_checkpoint_manager, mock_base_checkpointer):
    """Test that get_best_checkpoint_path delegates to the base checkpointer."""
    # Given
    expected_path = "/path/to/best"
    mock_base_checkpointer.get_best_checkpoint_path.return_value = expected_path

    # When
    actual_path = mlf_checkpoint_manager.get_best_checkpoint_path()

    # Then
    assert actual_path == expected_path
    mock_base_checkpointer.get_best_checkpoint_path.assert_called_once()


def test_get_latest_checkpoint_path_returns_base_when_mlf_missing(
    mlf_checkpoint_manager, mock_base_checkpointer, mock_checkpoint_loader
):
    """Test that it returns base path if MLF loader finds nothing."""
    # Given
    expected_path = "/path/to/latest"
    mock_base_checkpointer.get_latest_checkpoint_path.return_value = expected_path
    mock_checkpoint_loader.get_latest_complete_checkpoint.return_value = None

    # When
    actual_path = mlf_checkpoint_manager.get_latest_checkpoint_path()

    # Then
    assert actual_path == expected_path


def test_get_latest_checkpoint_path_returns_mlf_when_base_missing(
    mlf_checkpoint_manager, mock_base_checkpointer, mock_checkpoint_loader
):
    """Test that it returns MLF path if base checkpointer finds nothing."""
    # Given
    mock_base_checkpointer.get_latest_checkpoint_path.return_value = None

    mock_mlf_container = MagicMock()
    mock_mlf_container.data = "/test-mlf/step-150_ckpt"
    mock_checkpoint_loader.get_latest_complete_checkpoint.return_value = mock_mlf_container

    # When
    actual_path = mlf_checkpoint_manager.get_latest_checkpoint_path()

    # Then
    assert actual_path == "/test-mlf/step-150_ckpt"


def test_get_latest_checkpoint_path_returns_mlf_if_fresher(
    mlf_checkpoint_manager, mock_base_checkpointer, mock_checkpoint_loader
):
    """Test that it returns MLF path if it has a higher step number."""
    # Given
    base_path = "/base/step_100"
    mock_base_checkpointer.get_latest_checkpoint_path.return_value = base_path
    mock_base_checkpointer.load_training_info.return_value = {"step": 100}

    mock_mlf_container = MagicMock()
    mock_mlf_container.data = "/test-mlf/step-150_ckpt"
    mock_checkpoint_loader.get_latest_complete_checkpoint.return_value = mock_mlf_container

    # When
    actual_path = mlf_checkpoint_manager.get_latest_checkpoint_path()

    # Then
    assert actual_path == "/test-mlf/step-150_ckpt"
    mock_base_checkpointer.load_training_info.assert_called_once_with(base_path)


def test_get_latest_checkpoint_path_returns_base_if_fresher(
    mlf_checkpoint_manager, mock_base_checkpointer, mock_checkpoint_loader
):
    """Test that it returns base path if it has a higher step number."""
    # Given
    base_path = "/base/step_200"
    mock_base_checkpointer.get_latest_checkpoint_path.return_value = base_path
    mock_base_checkpointer.load_training_info.return_value = {"step": 200}

    mock_mlf_container = MagicMock()
    mock_mlf_container.data = "/test-mlf/step-150_ckpt"
    mock_checkpoint_loader.get_latest_complete_checkpoint.return_value = mock_mlf_container

    # When
    actual_path = mlf_checkpoint_manager.get_latest_checkpoint_path()

    # Then
    assert actual_path == base_path


def test_load_training_info_delegates(mlf_checkpoint_manager, mock_base_checkpointer):
    """Test that load_training_info delegates to the base checkpointer."""
    # Given
    expected_info = {"step": 100}
    mock_base_checkpointer.load_training_info.return_value = expected_info
    checkpoint_path = "/some/path"

    # When
    actual_info = mlf_checkpoint_manager.load_training_info(checkpoint_path)

    # Then
    assert actual_info == expected_info
    mock_base_checkpointer.load_training_info.assert_called_once_with(checkpoint_path)


def test_remove_old_checkpoints_delegates(mlf_checkpoint_manager, mock_base_checkpointer):
    """Test that remove_old_checkpoints delegates to the base checkpointer."""
    # Given
    exclude_latest = False

    # When
    mlf_checkpoint_manager.remove_old_checkpoints(exclude_latest)

    # Then
    mock_base_checkpointer.remove_old_checkpoints.assert_called_once_with(exclude_latest)


def test_get_latest_checkpoint_path_handles_load_training_info_error(
    mocker, mlf_checkpoint_manager, mock_base_checkpointer, mock_checkpoint_loader
):
    """Test that it falls back to MLF path if load_training_info raises an exception."""
    # Given
    mock_base_checkpointer.get_latest_checkpoint_path.return_value = "/base/path"
    mock_base_checkpointer.load_training_info.side_effect = Exception("Corrupted meta")

    mock_mlf_container = mocker.MagicMock()
    mock_mlf_container.data = "/test-mlf/step-150_ckpt"
    mock_checkpoint_loader.get_latest_complete_checkpoint.return_value = mock_mlf_container

    # When
    actual_path = mlf_checkpoint_manager.get_latest_checkpoint_path()

    # Then
    assert actual_path == "/test-mlf/step-150_ckpt"


def test_get_latest_checkpoint_path_handles_empty_training_info(
    mocker, mlf_checkpoint_manager, mock_base_checkpointer, mock_checkpoint_loader
):
    """Test that it defaults step to -1 if load_training_info returns empty dict."""
    # Given
    mock_base_checkpointer.get_latest_checkpoint_path.return_value = "/base/path"
    mock_base_checkpointer.load_training_info.return_value = {} # Empty dict

    mock_mlf_container = mocker.MagicMock()
    mock_mlf_container.data = "/test-mlf/step-150_ckpt"
    mock_checkpoint_loader.get_latest_complete_checkpoint.return_value = mock_mlf_container

    # When
    actual_path = mlf_checkpoint_manager.get_latest_checkpoint_path()

    # Then
    assert actual_path == "/test-mlf/step-150_ckpt"


