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

import lightning.pytorch as pl
import pytest

import ml_flashpoint
from ml_flashpoint.adapter.nemo.checkpoint_callback import (
    ML_FLASHPOINT_OPTS_KEY,
    ML_FLASHPOINT_TYPE,
    MLFlashpointCheckpointCallback,
)
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId
from ml_flashpoint.core.mlf_logging import _TRAINING_STEP


@pytest.fixture(autouse=True)
def training_step_fixture():
    """Fixture to manage the training step value for tests."""
    initial_value = _TRAINING_STEP.value
    yield
    _TRAINING_STEP.value = initial_value


def test_is_subtype_of_pytorch_lightning_callback():
    # Given
    base_container = CheckpointContainerId("/test")
    callback = MLFlashpointCheckpointCallback(checkpoint_base_container=base_container, every_n_steps=1)

    # When/Then
    assert issubclass(MLFlashpointCheckpointCallback, pl.callbacks.Callback)
    assert isinstance(callback, pl.callbacks.Callback)


def test_init_with_string_base_container_works():
    # When
    callback = MLFlashpointCheckpointCallback(checkpoint_base_container="/test", every_n_steps=1)

    # Then
    assert callback.base_container == CheckpointContainerId("/test")


def test_init_with_container_id_base_container_works():
    # Given
    base_container = CheckpointContainerId("/test")

    # When
    callback = MLFlashpointCheckpointCallback(checkpoint_base_container=base_container, every_n_steps=1)

    # Then
    assert callback.base_container == base_container


@pytest.mark.parametrize(
    "base_container_str, test_step, expected_ckpt_id_str",
    [
        ("/test/base", 123, "/test/base/step-123_ckpt"),
        ("/test", 456, "/test/step-456_ckpt"),
    ],
)
def test_on_train_batch_end_base_container_variations(mocker, base_container_str, test_step, expected_ckpt_id_str):
    """Tests that checkpoints are saved with correct paths for different base containers."""
    # Given
    # Mock Trainer and LightningModule
    trainer = mocker.MagicMock(spec=pl.Trainer)
    pl_module = mocker.MagicMock(spec=pl.LightningModule)

    # Configure trainer.global_step
    trainer.global_step = test_step

    # Instantiate the callback
    base_container = CheckpointContainerId(base_container_str)
    # Using every_n_steps=1 as that is not the subject of this test case and we assume checkpointing is always on.
    callback = MLFlashpointCheckpointCallback(checkpoint_base_container=base_container, every_n_steps=1)

    # When
    callback.on_train_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,  # Not used by this callback
        batch=None,  # Not used by this callback
        batch_idx=0,  # Not used by this callback
    )

    # Then
    expected_ckpt_version_container = CheckpointContainerId(expected_ckpt_id_str)
    expected_storage_options = {
        ML_FLASHPOINT_OPTS_KEY: {
            "ckpt_type": ML_FLASHPOINT_TYPE,
            "step": test_step,
        }
    }

    trainer.save_checkpoint.assert_called_once_with(
        expected_ckpt_version_container.data, storage_options=expected_storage_options
    )


@pytest.mark.parametrize(
    "test_step, every_n_steps, should_save",
    [
        (123, 1, True),  # Save every step
        (10, 5, True),  # Step is a multiple
        (11, 5, False),  # Step is not a multiple
        (3, 5, False),  # Step is less than every_n_steps
        (5, 5, True),  # Step equals every_n_steps
        (0, 5, True),  # Step is 0, 0 % 5 == 0
        (1000000, 100, True),  # Large step number
    ],
)
def test_on_train_batch_end_every_n_steps(mocker, test_step, every_n_steps, should_save):
    """Tests the every_n_steps logic in on_train_batch_end."""
    # Given
    trainer = mocker.MagicMock(spec=pl.Trainer)
    pl_module = mocker.MagicMock(spec=pl.LightningModule)
    trainer.global_step = test_step

    base_container = CheckpointContainerId("/test/base")
    callback = MLFlashpointCheckpointCallback(checkpoint_base_container=base_container, every_n_steps=every_n_steps)

    # Mock mlf_logging.update_training_step
    mocker.patch("ml_flashpoint.core.mlf_logging.update_training_step")

    # When
    callback.on_train_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=None,
        batch_idx=0,
    )

    # Then
    ml_flashpoint.core.mlf_logging.update_training_step.assert_called_once_with(test_step)

    if should_save:
        expected_ckpt_id_str = f"/test/base/step-{test_step}_ckpt"
        expected_ckpt_version_container = CheckpointContainerId(expected_ckpt_id_str)
        expected_storage_options = {
            ML_FLASHPOINT_OPTS_KEY: {
                "ckpt_type": ML_FLASHPOINT_TYPE,
                "step": test_step,
            }
        }
        trainer.save_checkpoint.assert_called_once_with(
            expected_ckpt_version_container.data, storage_options=expected_storage_options
        )
    else:
        trainer.save_checkpoint.assert_not_called()


@pytest.mark.parametrize(
    "test_step, every_n_steps, skip_every_n_steps, should_save",
    [
        # Basic skipping
        (10, 5, 10, False),  # Step is a multiple of both, skip
        (20, 5, 10, False),  # Step is a multiple of both, skip
        (15, 5, 10, True),  # Step is a multiple of every_n_steps, but not skip
        # No skipping
        (10, 5, 0, True),  # skip_every_n_steps is 0, should not skip
        (10, 5, None, True),  # skip_every_n_steps is None, treated as 0, should not skip
        # Edge cases
        (0, 5, 10, False),  # Step is 0, multiple of both, skip
        (10, 10, 10, False),  # All three are equal
        (10, 1, 5, False),  # Skip is a multiple of every_n_steps
    ],
)
def test_on_train_batch_end_skip_every_n_steps(mocker, test_step, every_n_steps, skip_every_n_steps, should_save):
    """Tests the skip_every_n_steps logic in on_train_batch_end."""
    # Given
    trainer = mocker.MagicMock(spec=pl.Trainer)
    pl_module = mocker.MagicMock(spec=pl.LightningModule)
    trainer.global_step = test_step

    base_container = CheckpointContainerId("/test/base")
    callback = MLFlashpointCheckpointCallback(
        checkpoint_base_container=base_container,
        every_n_steps=every_n_steps,
        skip_every_n_steps=skip_every_n_steps,
    )

    # Mock mlf_logging.update_training_step
    mocker.patch("ml_flashpoint.core.mlf_logging.update_training_step")

    # When
    callback.on_train_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=None,
        batch_idx=0,
    )

    # Then
    ml_flashpoint.core.mlf_logging.update_training_step.assert_called_once_with(test_step)

    if should_save:
        expected_ckpt_id_str = f"/test/base/step-{test_step}_ckpt"
        expected_ckpt_version_container = CheckpointContainerId(expected_ckpt_id_str)
        expected_storage_options = {
            ML_FLASHPOINT_OPTS_KEY: {
                "ckpt_type": ML_FLASHPOINT_TYPE,
                "step": test_step,
            }
        }
        trainer.save_checkpoint.assert_called_once_with(
            expected_ckpt_version_container.data, storage_options=expected_storage_options
        )
    else:
        trainer.save_checkpoint.assert_not_called()


@pytest.mark.parametrize("invalid_every_n_steps", [0, -1, -10, 1.5, "test"])
def test_invalid_every_n_steps_init(invalid_every_n_steps):
    """Tests that ValueError is raised for invalid every_n_steps values."""
    with pytest.raises(ValueError):
        MLFlashpointCheckpointCallback(
            checkpoint_base_container=CheckpointContainerId("/test"),
            every_n_steps=invalid_every_n_steps,
        )


@pytest.mark.parametrize("invalid_skip_every_n_steps", [-1, -10, 1.5, "test"])
def test_invalid_skip_every_n_steps_init(invalid_skip_every_n_steps):
    """Tests that ValueError is raised for invalid skip_every_n_steps values."""
    with pytest.raises(
        ValueError,
        match=f"skip_every_n_steps must be a non-negative integer, got '{invalid_skip_every_n_steps}' instead.",
    ):
        MLFlashpointCheckpointCallback(
            checkpoint_base_container=CheckpointContainerId("/test"),
            every_n_steps=1,
            skip_every_n_steps=invalid_skip_every_n_steps,
        )


def test_init_defaults_enabled_to_true():
    # Given
    base_container = CheckpointContainerId("/test")

    # When
    callback = MLFlashpointCheckpointCallback(checkpoint_base_container=base_container, every_n_steps=1)

    # Then
    assert callback._enabled is True


def test_init_sets_enabled_correctly():
    # Given
    base_container = CheckpointContainerId("/test")

    # When
    callback_enabled = MLFlashpointCheckpointCallback(
        checkpoint_base_container=base_container, every_n_steps=1, enabled=True
    )
    callback_disabled = MLFlashpointCheckpointCallback(
        checkpoint_base_container=base_container, every_n_steps=1, enabled=False
    )

    # Then
    assert callback_enabled._enabled is True
    assert callback_disabled._enabled is False


def test_on_train_batch_end_when_disabled(mocker):
    """Tests that no checkpoint is saved when the callback is disabled."""
    # Given
    trainer = mocker.MagicMock(spec=pl.Trainer)
    pl_module = mocker.MagicMock(spec=pl.LightningModule)
    test_step = 10
    trainer.global_step = test_step

    base_container = CheckpointContainerId("/test/base")
    # Set every_n_steps to a value that would normally trigger a save (10 % 5 == 0)
    callback = MLFlashpointCheckpointCallback(checkpoint_base_container=base_container, every_n_steps=5, enabled=False)

    # Mock mlf_logging.update_training_step
    mocker.patch("ml_flashpoint.core.mlf_logging.update_training_step")

    # When
    callback.on_train_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=None,
        batch_idx=0,
    )

    # Then
    ml_flashpoint.core.mlf_logging.update_training_step.assert_called_once_with(test_step)
    trainer.save_checkpoint.assert_not_called()


def test_on_train_batch_end_when_enabled(mocker):
    """Tests that checkpoint is saved when the callback is enabled."""
    # Given
    trainer = mocker.MagicMock(spec=pl.Trainer)
    pl_module = mocker.MagicMock(spec=pl.LightningModule)
    test_step = 10
    trainer.global_step = test_step

    base_container = CheckpointContainerId("/test/base")
    callback = MLFlashpointCheckpointCallback(checkpoint_base_container=base_container, every_n_steps=5, enabled=True)

    mocker.patch("ml_flashpoint.core.mlf_logging.update_training_step")

    # When
    callback.on_train_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        outputs=None,
        batch=None,
        batch_idx=0,
    )

    # Then
    # Should save
    trainer.save_checkpoint.assert_called_once()
