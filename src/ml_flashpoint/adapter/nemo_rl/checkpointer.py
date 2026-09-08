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

"""ML Flashpoint checkpointing for NeMo RL's Megatron policy worker.

NeMo RL builds its Megatron training state with Megatron Bridge but drives
checkpointing itself: the policy worker calls
``megatron.bridge.training.checkpointing.save_checkpoint`` directly rather than
going through ``create_checkpoint_manager``, so ``CheckpointConfig.
custom_manager_class`` is never consulted. This module therefore reuses the
Bridge checkpoint manager from the worker side, feeding it a save/load context
built from the worker's own attributes.
"""

import dataclasses
import logging
from typing import Any, Optional

from ml_flashpoint.adapter.megatron_bridge.checkpoint_manager import (
    MLFlashpointBridgeCheckpointManager,
)
from ml_flashpoint.adapter.megatron_bridge.config import MLFlashpointBridgeConfig, get_config
from ml_flashpoint.core.mlf_logging import get_logger
from ml_flashpoint.core.utils import log_execution_time

_LOGGER = get_logger(__name__)


@dataclasses.dataclass
class _SaveContext:
    """Duck-typed stand-in for Megatron Bridge's ``CheckpointSaveContext``.

    The Bridge manager only reads attributes off the context, so NeMo RL does not
    need to construct Bridge's dataclass (whose field set varies across versions).
    """

    state: Any
    model: list
    optimizer: Any
    opt_param_scheduler: Any
    num_floating_point_operations_so_far: int
    train_data_iterator: Any = None
    non_persistent_ckpt: bool = True
    pg_collection: Any = None
    module_name: Optional[str] = None


@dataclasses.dataclass
class _LoadContext:
    """Duck-typed stand-in for Megatron Bridge's ``CheckpointLoadContext``."""

    state: Any
    model: list
    optimizer: Any
    opt_param_scheduler: Any
    strict: bool = True
    skip_load_to_model_and_opt: bool = False
    pg_collection: Any = None
    module_name: Optional[str] = None


class MLFlashpointNeMoRLCheckpointer:
    """Saves and restores a NeMo RL Megatron policy through ML Flashpoint.

    ML Flashpoint checkpoints are node-local and peer-replicated: they survive a
    process or node failure within a run, but not the loss of the whole cluster
    and not the end of the run. They are a fast recovery tier, not a replacement
    for NeMo RL's durable ``checkpointing.save_period`` checkpoints, which must
    stay enabled.

    Attributes:
        manager: The underlying Megatron Bridge checkpoint manager.
    """

    def __init__(
        self,
        checkpoint_config,
        mlf_config: Optional[MLFlashpointBridgeConfig] = None,
    ):
        """Initializes the checkpointer.

        Args:
            checkpoint_config: The Bridge ``CheckpointConfig`` the worker was
                configured with (``worker.mcore_state.cfg.checkpoint``).
            mlf_config: ML Flashpoint settings. Defaults to the registered or
                environment-derived config.
        """
        self._mlf_config = mlf_config if mlf_config is not None else get_config()
        self.manager = MLFlashpointBridgeCheckpointManager(
            checkpoint_config=checkpoint_config,
            mlf_config=self._mlf_config,
        )

    @property
    def enabled(self) -> bool:
        """Whether ML Flashpoint is active for this run."""
        return self.manager.enabled

    @log_execution_time(logger=_LOGGER, name="MLFlashpointNeMoRLCheckpointer.save", level=logging.INFO)
    def save(
        self,
        state,
        model,
        optimizer=None,
        opt_param_scheduler=None,
        num_floating_point_operations_so_far: Optional[int] = None,
    ) -> None:
        """Writes an ML Flashpoint checkpoint for the current step.

        The step comes from ``state.train_state.step``, which NeMo RL keeps in
        sync with the RL loop.

        Args:
            state: The worker's ``GlobalState`` (``worker.mcore_state``).
            model: The model module, or a list of module chunks.
            optimizer: The optimizer, if its state should be captured.
            opt_param_scheduler: The scheduler, if its state should be captured.
            num_floating_point_operations_so_far: Cumulative FLOPs. Defaults to
                the value already tracked on the train state.
        """
        if num_floating_point_operations_so_far is None:
            num_floating_point_operations_so_far = getattr(state.train_state, "floating_point_operations_so_far", 0)
        ctx = _SaveContext(
            state=state,
            model=_as_module_list(model),
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            num_floating_point_operations_so_far=int(num_floating_point_operations_so_far),
            non_persistent_ckpt=True,
        )
        self.manager.save(ctx, callback_manager=None)

    @log_execution_time(logger=_LOGGER, name="MLFlashpointNeMoRLCheckpointer.load", level=logging.INFO)
    def load(
        self,
        state,
        model,
        optimizer=None,
        opt_param_scheduler=None,
        strict: bool = True,
    ) -> Optional[tuple[int, int]]:
        """Restores from the newest ML Flashpoint checkpoint, if one exists.

        Unlike the Megatron Bridge manager's ``load``, this never falls back to
        Bridge's durable load path: NeMo RL owns that decision and performs it
        during worker setup.

        Args:
            state: The worker's ``GlobalState``.
            model: The model module, or a list of module chunks.
            optimizer: The optimizer to restore into.
            opt_param_scheduler: The scheduler to restore into.
            strict: Whether to enforce strict key matching on the model load.

        Returns:
            A tuple of (step, cumulative FLOPs), or None when nothing was
            recovered.
        """
        container = self.manager._find_ml_flashpoint_checkpoint()
        if container is None:
            _LOGGER.info("No recoverable ML Flashpoint checkpoint found.")
            return None
        ctx = _LoadContext(
            state=state,
            model=_as_module_list(model),
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            strict=strict,
        )
        return self.manager._load_ml_flashpoint(ctx, container)

    def finalize(self, blocking: bool = True) -> None:
        """Finalizes pending ML Flashpoint saves.

        Args:
            blocking: If True, waits for every pending save to complete.
        """
        runtime = self.manager.runtime
        if runtime is None:
            return
        runtime.maybe_finalize(blocking=blocking)

    def shutdown(self) -> None:
        """Releases every ML Flashpoint resource held by this rank."""
        self.manager.shutdown()


def _as_module_list(model) -> list:
    """Normalizes a model argument to the list of chunks Bridge expects.

    Args:
        model: A single module or a list of module chunks.

    Returns:
        A list of modules.
    """
    if isinstance(model, (list, tuple)):
        return list(model)
    return [model]
