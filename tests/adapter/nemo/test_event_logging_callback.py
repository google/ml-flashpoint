# Copyright 2026 Google LLC
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

from unittest.mock import MagicMock

import lightning.pytorch as pl
import pytest
import torch

from ml_flashpoint.adapter.nemo.event_logging_callback import EventLoggingCallback

# Exhaustive list of all hooks implemented in EventLoggingCallback.
# Format: (method_name, extra_kwargs, expected_event_string)
HOOKS_TO_TEST = [
    ("on_train_start", {}, "on_train_start"),
    ("on_train_end", {}, "on_train_end"),
    ("on_train_batch_start", {"batch": None, "batch_idx": 0}, "on_train_batch_start"),
    ("on_train_batch_end", {"outputs": None, "batch": None, "batch_idx": 0}, "on_train_batch_end"),
    ("on_validation_batch_start", {"batch": None, "batch_idx": 0, "dataloader_idx": 0}, "on_validation_batch_start"),
    (
        "on_validation_batch_end",
        {"outputs": None, "batch": None, "batch_idx": 0, "dataloader_idx": 0},
        "on_validation_batch_end",
    ),
    ("on_test_epoch_start", {}, "on_test_epoch_start"),
    ("on_test_epoch_end", {}, "on_test_epoch_end"),
    ("on_test_batch_start", {"batch": None, "batch_idx": 0, "dataloader_idx": 0}, "on_test_batch_start"),
    ("on_test_batch_end", {"outputs": None, "batch": None, "batch_idx": 0, "dataloader_idx": 0}, "on_test_batch_end"),
    ("load_state_dict", {"state_dict": {}}, "load_state_dict"),
    ("on_save_checkpoint", {"checkpoint": {}}, "on_save_checkpoint"),
    ("on_load_checkpoint", {"checkpoint": {}}, "on_load_checkpoint"),
    ("on_before_backward", {"loss": torch.tensor(0.0)}, "on_before_backward"),
    ("on_after_backward", {}, "on_after_backward"),
    ("on_before_optimizer_step", {"optimizer": MagicMock(spec=torch.optim.Optimizer)}, "on_before_optimizer_step"),
    ("on_before_zero_grad", {"optimizer": MagicMock(spec=torch.optim.Optimizer)}, "on_before_zero_grad"),
]


def test_is_subtype_of_pytorch_lightning_callback():
    """Verify inheritance to ensure compatibility with PyTorch Lightning."""
    assert issubclass(EventLoggingCallback, pl.callbacks.Callback)


@pytest.mark.parametrize("hook_name, kwargs, expected_event", HOOKS_TO_TEST)
def test_event_logging_hooks_log_correctly(mocker, hook_name, kwargs, expected_event):
    """
    Tests that every lifecycle hook in EventLoggingCallback logs the correct event.
    """

    # Given
    mock_logger = mocker.patch("ml_flashpoint.adapter.nemo.event_logging_callback._LOGGER")
    callback = EventLoggingCallback()

    # Mock Trainer and LightningModule as required by the PyTorch Lightning API.
    trainer = mocker.MagicMock(spec=pl.Trainer)
    pl_module = mocker.MagicMock(spec=pl.LightningModule)

    # When
    # Dynamically fetch the method to test.
    hook_method = getattr(callback, hook_name)

    # load_state_dict does not follow the (trainer, pl_module) signature.
    if hook_name == "load_state_dict":
        hook_method(**kwargs)
    else:
        hook_method(trainer=trainer, pl_module=pl_module, **kwargs)

    # Then
    # Verify the log content matches the implementation of _log_event.
    # LOGGER.info(f"event={hook_name}")
    expected_log_msg = f"event={expected_event}"
    mock_logger.info.assert_called_once_with(expected_log_msg)
