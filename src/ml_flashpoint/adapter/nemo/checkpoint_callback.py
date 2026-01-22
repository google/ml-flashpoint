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

import logging
from typing import Any, Union

import lightning.pytorch as pl
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch.utilities import types as pl_util_types
from typing_extensions import override

from ml_flashpoint.core import mlf_logging
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId
from ml_flashpoint.core.mlf_logging import get_logger
from ml_flashpoint.core.utils import log_execution_time

_LOGGER = get_logger(__name__)


class MLFlashpointMarker:
    """
    Marker class used as a key and/or value to signify ML Flashpoint checkpointing in storage_options.
    """


ML_FLASHPOINT_TYPE = MLFlashpointMarker()
"""Constant value used in storage_options, that should be compared against."""
ML_FLASHPOINT_OPTS_KEY = "ML_FLASHPOINT_OPTS"
"""String key used in the storage_options dictionary."""


class MLFlashpointCheckpointCallback(pl_callbacks.Callback):
    """
    A PyTorch Lightning callback for saving checkpoints using ML Flashpoint.

    This callback hooks into the training loop to save checkpoints at specified intervals
    using the ML Flashpoint library. It utilizes the storage_options to pass
    ML Flashpoint specific information to :meth:`pl.Trainer.save_checkpoint`.
    """

    def __init__(
        self,
        checkpoint_base_container: Union[str, CheckpointContainerId],
        every_n_steps: int,
        skip_every_n_steps: int = 0,
        enabled: bool = True,
    ):
        """
        Initializes and validates the callback.

        Args:
            checkpoint_base_container (CheckpointContainerId): The base container for all checkpoints in this particular
                job run (session).
                Checkpoint versions will be written to child containers within this base. This base container should be
                the same one that is used in :class:`MLFlashpointAutoResume`, and must be unique per job run.
            every_n_steps (int): The step frequency to checkpoint. 1 is a typical, recommended value.
                Must be a positive integer.
            skip_every_n_steps (int): The step frequency to skip checkpointing. This is suggested to be set to the
                interval used for long-term checkpointing by the alternative strategy. Defaults to 0 (no skipping).
            enabled (bool): Whether this callback should be enabled. Defaults to True.
        """
        self.base_container = CheckpointContainerId(checkpoint_base_container)
        self.every_n_steps = every_n_steps
        self.skip_every_n_steps = skip_every_n_steps if skip_every_n_steps is not None else 0
        self._enabled = enabled
        self._validate()

    def _validate(self):
        """Ensures this instance passes validity checks and expectations. Expected to be used by __init__.

        Raises:
            ValueError: If every_n_steps has an invalid value.
        """
        if not isinstance(self.every_n_steps, int) or self.every_n_steps < 1:
            raise ValueError(f"every_n_steps must be a positive integer, got '{self.every_n_steps}' instead.")
        if not isinstance(self.skip_every_n_steps, int) or self.skip_every_n_steps < 0:
            raise ValueError(
                f"skip_every_n_steps must be a non-negative integer, got '{self.skip_every_n_steps}' instead."
            )

    def _format_checkpoint_version_container_id(self, step: int) -> CheckpointContainerId:
        """Formats the checkpoint container ID for a specific step, as a child of the base container.

        Args:
            step (int): The current global step number.

        Returns:
            CheckpointContainerId: The formatted checkpoint container ID.
        """
        return CheckpointContainerId.create_child(
            self.base_container, CheckpointContainerId.format_version_container(step)
        )

    @override
    @log_execution_time(logger=_LOGGER, name="on_train_batch_end", level=logging.INFO)
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: pl_util_types.STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step

        # Set training step context for logs
        mlf_logging.update_training_step(step)

        if not self._enabled:
            _LOGGER.debug("'%s' is disabled, skipping save.", self.__class__.__name__)
            return

        if step % self.every_n_steps != 0:
            # Not time to checkpoint, so skip.
            _LOGGER.info(
                "%s skipping save this step: '%d' (every_n_steps is configured as: '%d')",
                self.__class__.__name__,
                step,
                self.every_n_steps,
            )
            return

        if self.skip_every_n_steps > 0 and step % self.skip_every_n_steps == 0:
            _LOGGER.info(
                "%s skipping save this step: '%d' (skip_every_n_steps is configured as: '%d')",
                self.__class__.__name__,
                step,
                self.skip_every_n_steps,
            )
            return

        ckpt_version_container = self._format_checkpoint_version_container_id(step)
        ckpt_options = dict(
            ckpt_type=ML_FLASHPOINT_TYPE,
            step=step,
        )
        _LOGGER.info(
            "%s saving checkpoint path: '%s', ckpt_options: %s",
            self.__class__.__name__,
            ckpt_version_container,
            ckpt_options,
        )
        trainer.save_checkpoint(ckpt_version_container.data, storage_options={ML_FLASHPOINT_OPTS_KEY: ckpt_options})
