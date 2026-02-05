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

import datetime
from typing import Any, Dict

import lightning.pytorch as pl
import torch
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch.utilities.types import STEP_OUTPUT

# from lightning.pytorch.utilities import types as pl_util_types
from typing_extensions import override

from ml_flashpoint.core.mlf_logging import get_logger

_LOGGER = get_logger(__name__)


class StepTimerCallback(pl_callbacks.Callback):
    """
    A comprehensive logging callback to record timestamps for all key PyTorch Lightning
    lifecycle events to monitor execution flow.
    """

    def _log_event(self, trainer: "pl.Trainer", hook_name: str) -> None:
        _LOGGER.info(f"{hook_name: <30}")

    @override
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train begins."""
        self._log_event(trainer, "on_train_start")

    @override
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train ends."""
        self._log_event(trainer, "on_train_end")

    @override
    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch begins."""
        self._log_event(trainer, "on_train_batch_start")

    @override
    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch ends."""
        self._log_event(trainer, "on_train_batch_end")

    @override
    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the validation batch begins."""
        self._log_event(trainer, "on_validation_batch_start")

    @override
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the validation batch ends."""
        self._log_event(trainer, "on_validation_batch_end")

    @override
    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the test epoch begins."""
        self._log_event(trainer, "on_test_epoch_start")

    @override
    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the test epoch ends."""
        self._log_event(trainer, "on_test_epoch_end")

    @override
    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch begins."""
        self._log_event(trainer, "on_test_batch_start")

    @override
    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch ends."""
        self._log_event(trainer, "on_test_batch_end")

    @override
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload callback state."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        _LOGGER.info(f"[{timestamp}] Hook: load_state_dict")

    @override
    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        """Called when saving a checkpoint."""
        self._log_event(trainer, "on_save_checkpoint")

    @override
    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        """Called when loading a model checkpoint, use to reload state."""
        self._log_event(trainer, "on_load_checkpoint")

    @override
    def on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: torch.Tensor) -> None:
        """Called before loss.backward()."""
        self._log_event(trainer, "on_before_backward")

    @override
    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called after loss.backward() and before optimizers are stepped."""
        self._log_event(trainer, "on_after_backward")

    @override
    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: torch.optim.Optimizer
    ) -> None:
        """Called before optimizer.step()."""
        self._log_event(trainer, "on_before_optimizer_step")

    @override
    def on_before_zero_grad(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: torch.optim.Optimizer
    ) -> None:
        """Called before optimizer.zero_grad()."""
        self._log_event(trainer, "on_before_zero_grad")
