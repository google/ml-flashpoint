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

"""Attaching ML Flashpoint to NeMo RL's Megatron policy worker.

NeMo RL calls Megatron Bridge's ``save_checkpoint`` from
``MegatronPolicyWorker.save_checkpoint`` and never consults
``CheckpointConfig.custom_manager_class``, so there is no configuration-only way
to reach the Bridge checkpoint manager. :func:`install_into_worker` wraps that
one method on a worker instance instead, which keeps the change confined to a
single, explicitly named entry point.

Two modes are supported:

``augment``
    Every NeMo RL checkpoint is still written durably, and an ML Flashpoint
    checkpoint is written alongside it. Use this when the goal is faster
    recovery without weakening durability.

``replace``
    Only every ``durable_every_n_saves``-th checkpoint is written durably; the
    rest go to ML Flashpoint alone. This is what removes checkpoint stalls from
    the RL loop, and what a checkpoint-time A/B measures.
"""

import functools
import logging
from typing import Any, Optional

from ml_flashpoint.adapter.megatron_bridge.config import MLFlashpointBridgeConfig, get_config
from ml_flashpoint.adapter.nemo_rl.checkpointer import MLFlashpointNeMoRLCheckpointer
from ml_flashpoint.core import utils
from ml_flashpoint.core.mlf_logging import get_logger
from ml_flashpoint.core.utils import log_execution_time

_LOGGER = get_logger(__name__)

MODE_AUGMENT = "augment"
MODE_REPLACE = "replace"
_VALID_MODES = (MODE_AUGMENT, MODE_REPLACE)

_INSTALLED_ATTR = "_ml_flashpoint_checkpointer"


class MLFlashpointWorkerHooks:
    """Bookkeeping for one worker's ML Flashpoint installation.

    Attributes:
        checkpointer: The ML Flashpoint checkpointer bound to the worker.
        mode: Either ``augment`` or ``replace``.
        durable_every_n_saves: In ``replace`` mode, how often a durable
            checkpoint is still written. ``1`` means every save stays durable,
            which makes ``replace`` equivalent to ``augment``.
    """

    def __init__(
        self,
        checkpointer: MLFlashpointNeMoRLCheckpointer,
        mode: str,
        durable_every_n_saves: int,
    ):
        self.checkpointer = checkpointer
        self.mode = mode
        self.durable_every_n_saves = durable_every_n_saves
        self._save_count = 0

    def should_save_durable(self) -> bool:
        """Decides whether the current save must also go to durable storage.

        Returns:
            True when the durable write should run.
        """
        self._save_count += 1
        if self.mode == MODE_AUGMENT:
            return True
        return self._save_count % self.durable_every_n_saves == 0


def install_into_worker(
    worker,
    mode: str = MODE_AUGMENT,
    durable_every_n_saves: int = 1,
    mlf_config: Optional[MLFlashpointBridgeConfig] = None,
) -> MLFlashpointNeMoRLCheckpointer:
    """Routes a NeMo RL Megatron policy worker's saves through ML Flashpoint.

    Call this after the worker has finished initializing, i.e. once
    ``worker.mcore_state``, ``worker.model`` and ``torch.distributed`` are all
    live. Installing twice on the same worker is a no-op.

    Args:
        worker: A NeMo RL ``MegatronPolicyWorker`` instance.
        mode: ``augment`` to add an ML Flashpoint checkpoint next to every
            durable one, or ``replace`` to skip most durable writes.
        durable_every_n_saves: In ``replace`` mode, how often to still write
            durably. Must be positive.
        mlf_config: ML Flashpoint settings. Defaults to the registered or
            environment-derived config.

    Returns:
        The checkpointer bound to the worker.

    Raises:
        ValueError: If ``mode`` or ``durable_every_n_saves`` is invalid.
        AttributeError: If the worker does not expose the attributes the
            adapter needs.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got '{mode}'.")
    if durable_every_n_saves < 1:
        raise ValueError(f"durable_every_n_saves must be a positive integer, got {durable_every_n_saves}.")

    existing = getattr(worker, _INSTALLED_ATTR, None)
    if existing is not None:
        _LOGGER.info("ML Flashpoint is already installed on this worker; skipping.")
        return existing.checkpointer

    for attr in ("mcore_state", "model", "save_checkpoint"):
        if not hasattr(worker, attr):
            raise AttributeError(
                f"Worker of type '{type(worker).__name__}' has no '{attr}'. This does not look like a NeMo RL "
                "Megatron policy worker; ML Flashpoint cannot be installed on it."
            )

    checkpointer = MLFlashpointNeMoRLCheckpointer(
        checkpoint_config=worker.mcore_state.cfg.checkpoint,
        mlf_config=mlf_config,
    )
    hooks = MLFlashpointWorkerHooks(checkpointer, mode=mode, durable_every_n_saves=durable_every_n_saves)
    setattr(worker, _INSTALLED_ATTR, hooks)

    original_save = worker.save_checkpoint

    @functools.wraps(original_save)
    @log_execution_time(logger=_LOGGER, name="nemo_rl.save_checkpoint", level=logging.INFO)
    def save_checkpoint(weights_path: str, optimizer_path: Optional[str] = None, **kwargs):
        save_durable = hooks.should_save_durable()
        _save_ml_flashpoint(worker, hooks, optimizer_path is not None)
        if not save_durable:
            _LOGGER.info(
                "Skipping the durable checkpoint at '%s': ML Flashpoint holds this step (mode='%s', "
                "durable_every_n_saves=%d).",
                weights_path,
                hooks.mode,
                hooks.durable_every_n_saves,
            )
            return None
        return original_save(weights_path, optimizer_path=optimizer_path, **kwargs)

    worker.save_checkpoint = save_checkpoint
    _LOGGER.info(
        "Installed ML Flashpoint on the NeMo RL Megatron policy worker (mode='%s', durable_every_n_saves=%d).",
        mode,
        durable_every_n_saves,
    )
    return checkpointer


def _save_ml_flashpoint(worker, hooks: MLFlashpointWorkerHooks, include_optimizer: bool) -> None:
    """Writes one ML Flashpoint checkpoint for a worker, best effort.

    Args:
        worker: The NeMo RL Megatron policy worker.
        hooks: The worker's installation record.
        include_optimizer: Whether optimizer and scheduler state should be saved.
    """
    try:
        hooks.checkpointer.save(
            state=worker.mcore_state,
            model=worker.model,
            optimizer=getattr(worker, "optimizer", None) if include_optimizer else None,
            opt_param_scheduler=getattr(worker, "scheduler", None) if include_optimizer else None,
        )
    except Exception:
        _LOGGER.exception("ML Flashpoint save failed. Continuing; the durable checkpoint path is unaffected.")


def uninstall_from_worker(worker) -> None:
    """Removes the ML Flashpoint wrapper and releases its resources.

    Args:
        worker: A worker previously passed to :func:`install_into_worker`.
    """
    hooks = getattr(worker, _INSTALLED_ATTR, None)
    if hooks is None:
        return
    try:
        hooks.checkpointer.shutdown()
    finally:
        # functools.wraps keeps __wrapped__ pointing at the bound original.
        original = getattr(worker.save_checkpoint, "__wrapped__", None)
        if original is not None:
            worker.save_checkpoint = original
        else:
            # Fall back to the class method by dropping the instance attribute.
            worker.__dict__.pop("save_checkpoint", None)
        delattr(worker, _INSTALLED_ATTR)
    _LOGGER.info("Removed ML Flashpoint from the NeMo RL Megatron policy worker.")


def install_from_env(worker) -> Optional[MLFlashpointNeMoRLCheckpointer]:
    """Installs ML Flashpoint only when the environment asks for it.

    Reads ``MLFLASHPOINT_NEMO_RL_ENABLED`` (default false),
    ``MLFLASHPOINT_NEMO_RL_MODE`` (default ``augment``) and
    ``MLFLASHPOINT_NEMO_RL_DURABLE_EVERY_N_SAVES`` (default 1). This is what lets
    a single launch command run both arms of an A/B experiment.

    Args:
        worker: A NeMo RL ``MegatronPolicyWorker`` instance.

    Returns:
        The checkpointer, or None when ML Flashpoint is not enabled.
    """
    if not utils.get_env_val_bool("NEMO_RL_ENABLED", False):
        _LOGGER.info("MLFLASHPOINT_NEMO_RL_ENABLED is not set; running without ML Flashpoint.")
        return None
    if not get_config().enabled:
        _LOGGER.info("ML Flashpoint is disabled in its own config; running without it.")
        return None
    return install_into_worker(
        worker,
        mode=utils.get_env_val_str("NEMO_RL_MODE", MODE_AUGMENT),
        durable_every_n_saves=utils.get_env_val_int("NEMO_RL_DURABLE_EVERY_N_SAVES", 1),
    )


def get_checkpointer(worker) -> Optional[MLFlashpointNeMoRLCheckpointer]:
    """Returns the checkpointer installed on a worker, if any.

    Args:
        worker: A NeMo RL ``MegatronPolicyWorker`` instance.

    Returns:
        The checkpointer, or None.
    """
    hooks: Optional[Any] = getattr(worker, _INSTALLED_ATTR, None)
    return hooks.checkpointer if hooks is not None else None
