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

"""A Megatron Bridge ``CheckpointManager`` backed by ML Flashpoint.

Megatron Bridge lets a run swap in its own checkpoint manager through
``CheckpointConfig.custom_manager_class``. This module provides one that keeps
Bridge's behavior for durable checkpoints and takes over the frequent,
non-persistent ones, which is where ML Flashpoint's memory-first saves pay off.
"""

import logging
from typing import Any, Optional

import torch
import torch.distributed as dist
from megatron.bridge.training.checkpointing import (
    init_checkpointing_context,
    load_checkpoint,
    maybe_finalize_async_save,
    save_checkpoint,
)
from megatron.core import dist_checkpointing as mcore_dist_checkpointing
from megatron.core.dist_checkpointing.strategies.common import TorchCommonLoadStrategy

from ml_flashpoint.adapter.megatron.save_utils import save_local_aware_megatron_checkpoint
from ml_flashpoint.adapter.megatron_bridge import bridge_state
from ml_flashpoint.adapter.megatron_bridge.config import MLFlashpointBridgeConfig, get_config
from ml_flashpoint.adapter.megatron_bridge.local_checkpoint_index import (
    MLFlashpointLocalCheckpointIndex,
)
from ml_flashpoint.adapter.megatron_bridge.runtime import MLFlashpointBridgeRuntime, get_runtime
from ml_flashpoint.core import mlf_logging
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId
from ml_flashpoint.core.mlf_logging import get_logger
from ml_flashpoint.core.utils import log_execution_time

_LOGGER = get_logger(__name__)

SUPPORTED_CKPT_FORMAT = "torch_dist"
"""The only Megatron checkpoint format the ML Flashpoint strategies handle."""

LOCAL_NON_PERSISTENT_CKPT_TYPE = "local"
"""``non_persistent_ckpt_type`` value that hands non-persistent saves to this adapter."""


class MLFlashpointBridgeCheckpointManager:
    """Routes non-persistent checkpoints to ML Flashpoint, the rest to Megatron Bridge.

    Wire it up with::

        checkpoint = CheckpointConfig(
            save="/gcs/my-run/checkpoints",
            save_interval=500,
            non_persistent_save_interval=20,
            non_persistent_ckpt_type="local",
            custom_manager_class=(
                "ml_flashpoint.adapter.megatron_bridge."
                "MLFlashpointBridgeCheckpointManager"
            ),
        )

    Megatron Bridge validates ``custom_manager_class`` against an import
    allowlist, so ``ml_flashpoint`` has to be registered first. Calling
    :func:`ml_flashpoint.adapter.megatron_bridge.register_with_megatron_bridge`
    (or importing this package's ``enable`` helper) does that.

    Everything ML Flashpoint needs beyond ``CheckpointConfig`` comes from
    :func:`ml_flashpoint.adapter.megatron_bridge.config.get_config`, because
    Bridge's factory only passes the checkpoint config.

    Attributes:
        checkpoint_config: The Bridge ``CheckpointConfig`` this manager was built with.
    """

    def __init__(
        self,
        checkpoint_config,
        mlf_config: Optional[MLFlashpointBridgeConfig] = None,
        runtime: Optional[MLFlashpointBridgeRuntime] = None,
    ):
        """Initializes the manager.

        The ML Flashpoint runtime is built lazily on first use, because Bridge
        constructs the manager before ``torch.distributed`` initialization is
        guaranteed to have happened on every code path.

        Args:
            checkpoint_config: The Bridge ``CheckpointConfig``.
            mlf_config: ML Flashpoint settings. Defaults to the registered or
                environment-derived config.
            runtime: An already-built runtime. Mainly for tests; production code
                should let the manager build (and share) one.
        """
        self.checkpoint_config = checkpoint_config
        self._mlf_config = mlf_config if mlf_config is not None else get_config()
        self._runtime = runtime
        self._local_index: Optional[MLFlashpointLocalCheckpointIndex] = None
        self._context: dict[str, Any] = init_checkpointing_context_safely(checkpoint_config)
        self._enabled = self._resolve_enabled()
        if self._enabled and runtime is not None:
            self._install_local_index()

    def _resolve_enabled(self) -> bool:
        """Determines whether ML Flashpoint should handle non-persistent saves.

        Returns:
            True when ML Flashpoint is enabled and the run's checkpoint format is
            one the ML Flashpoint strategies support.
        """
        if not self._mlf_config.enabled:
            _LOGGER.info("ML Flashpoint is disabled; delegating every checkpoint operation to Megatron Bridge.")
            return False
        ckpt_format = getattr(self.checkpoint_config, "ckpt_format", SUPPORTED_CKPT_FORMAT)
        if ckpt_format != SUPPORTED_CKPT_FORMAT:
            _LOGGER.warning(
                "ML Flashpoint supports ckpt_format='%s' only, but this run uses '%s'. "
                "Delegating every checkpoint operation to Megatron Bridge.",
                SUPPORTED_CKPT_FORMAT,
                ckpt_format,
            )
            return False
        return True

    @property
    def enabled(self) -> bool:
        """Whether ML Flashpoint handles this run's non-persistent checkpoints."""
        return self._enabled

    @property
    def checkpointing_context(self) -> dict[str, Any]:
        """The context Megatron Bridge caches strategies and resume hints in.

        Bridge reads ``local_checkpoint_manager`` out of this to decide whether a
        resume should be attempted; the adapter installs its own index under that
        key so ML Flashpoint checkpoints are discoverable.

        Reading this builds the ML Flashpoint runtime if it does not exist yet.
        Megatron Bridge reads it during setup, before any save has happened, and
        without the runtime there would be no index to advertise -- a restarted
        job whose only checkpoint is an ML Flashpoint one would silently start
        from scratch.
        """
        if self._enabled and self._local_index is None:
            self._ensure_runtime()
        return self._context

    @property
    def runtime(self) -> Optional[MLFlashpointBridgeRuntime]:
        """The ML Flashpoint runtime, once it has been built."""
        return self._runtime

    def _ensure_runtime(self) -> Optional[MLFlashpointBridgeRuntime]:
        """Builds the shared runtime on first use.

        Returns:
            The runtime, or None if it could not be built (ML Flashpoint is then
            disabled for the rest of the run).
        """
        if not self._enabled:
            return None
        if self._runtime is not None:
            return self._runtime
        try:
            self._runtime = get_runtime(self._mlf_config)
            self._install_local_index()
        except Exception:
            _LOGGER.exception(
                "Failed to initialize the ML Flashpoint runtime. Falling back to Megatron Bridge checkpointing "
                "for the remainder of this run."
            )
            self._enabled = False
            self._runtime = None
        return self._runtime

    def _install_local_index(self) -> None:
        """Publishes the resume index into the checkpointing context."""
        if self._runtime is None or self._local_index is not None:
            return
        self._local_index = MLFlashpointLocalCheckpointIndex(
            base_container=self._runtime.base_container,
            checkpoint_loader=self._runtime.checkpoint_loader,
        )
        self._context["local_checkpoint_manager"] = self._local_index

    def _version_container(self, step: int) -> CheckpointContainerId:
        """Returns the container for a given step.

        Args:
            step: The training step.

        Returns:
            The child container ID for that step.
        """
        return CheckpointContainerId.create_child(
            self._runtime.base_container,
            CheckpointContainerId.format_version_container(step),
        )

    @log_execution_time(logger=_LOGGER, name="MLFlashpointBridgeCheckpointManager.save", level=logging.INFO)
    def save(self, ctx, callback_manager=None) -> None:
        """Saves a checkpoint.

        Non-persistent checkpoints go to ML Flashpoint; everything else is
        delegated to Megatron Bridge unchanged.

        A failure on the ML Flashpoint path is logged and swallowed: a
        non-persistent checkpoint is an optimization for crash recovery, and
        losing one must not take the training job down.

        Args:
            ctx: The Bridge ``CheckpointSaveContext``.
            callback_manager: The Bridge callback manager, if any.
        """
        step = ctx.state.train_state.step
        mlf_logging.update_training_step(step)

        if not ctx.non_persistent_ckpt or self._ensure_runtime() is None:
            self._delegate_save(ctx, callback_manager)
            return

        try:
            self._save_ml_flashpoint(ctx, step)
        except Exception:
            _LOGGER.exception(
                "ML Flashpoint save failed at step %d. Skipping this non-persistent checkpoint and continuing.",
                step,
            )

    def _delegate_save(self, ctx, callback_manager) -> None:
        """Runs Megatron Bridge's own save.

        Args:
            ctx: The Bridge ``CheckpointSaveContext``.
            callback_manager: The Bridge callback manager, if any.

        Raises:
            RuntimeError: If Bridge is asked for a local non-persistent save that
                ML Flashpoint was supposed to own.
        """
        if ctx.non_persistent_ckpt and self._is_local_non_persistent():
            raise RuntimeError(
                "Megatron Bridge was asked to write a local non-persistent checkpoint, but "
                "non_persistent_ckpt_type='local' is what routes those to ML Flashpoint and "
                "ML Flashpoint is unavailable. Set non_persistent_ckpt_type='global' (or drop "
                "non_persistent_save_interval) to run without ML Flashpoint."
            )
        save_checkpoint(
            state=ctx.state,
            model=ctx.model,
            optimizer=ctx.optimizer,
            opt_param_scheduler=ctx.opt_param_scheduler,
            num_floating_point_operations_so_far=ctx.num_floating_point_operations_so_far,
            checkpointing_context=self._context,
            non_persistent_ckpt=ctx.non_persistent_ckpt,
            train_data_iterator=ctx.train_data_iterator,
            pg_collection=getattr(ctx, "pg_collection", None),
            callback_manager=callback_manager,
            module_name=getattr(ctx, "module_name", None),
        )

    def _is_local_non_persistent(self) -> bool:
        """Whether this run routes non-persistent checkpoints to the local path."""
        return getattr(self.checkpoint_config, "non_persistent_ckpt_type", None) == LOCAL_NON_PERSISTENT_CKPT_TYPE

    @log_execution_time(logger=_LOGGER, name="MLFlashpointBridgeCheckpointManager.mlf_save", level=logging.INFO)
    def _save_ml_flashpoint(self, ctx, step: int) -> None:
        """Stages and schedules an ML Flashpoint save for the current step.

        Args:
            ctx: The Bridge ``CheckpointSaveContext``.
            step: The training step.
        """
        container = self._version_container(step)
        state_dict = bridge_state.build_save_state_dict(
            ctx,
            step=step,
            num_floating_point_operations_so_far=ctx.num_floating_point_operations_so_far,
        )

        async_request = save_local_aware_megatron_checkpoint(
            checkpoint=state_dict,
            checkpoint_dir=str(container),
            save_strategy=self._runtime.save_strategy,
            async_save=self._mlf_config.async_save,
        )
        if async_request is not None:
            self._runtime.schedule(async_request)
        # A new container exists, so any cached resume decision is stale.
        if self._local_index is not None:
            self._local_index.invalidate()
        _LOGGER.info("Scheduled ML Flashpoint checkpoint for step %d at '%s'", step, container)

    @log_execution_time(logger=_LOGGER, name="MLFlashpointBridgeCheckpointManager.load", level=logging.INFO)
    def load(self, ctx) -> tuple[int, int]:
        """Loads a checkpoint, preferring the newest ML Flashpoint container.

        Args:
            ctx: The Bridge ``CheckpointLoadContext``.

        Returns:
            A tuple of (step, cumulative floating point operations). ``(0, 0)``
            when nothing was loaded.
        """
        container = self._find_ml_flashpoint_checkpoint()
        if container is not None:
            loaded = self._load_ml_flashpoint(ctx, container)
            if loaded is not None:
                return loaded

        # Bridge's own local-checkpoint path cannot read an ML Flashpoint
        # container, so make sure it is not offered one.
        if self._local_index is not None:
            self._local_index.disable()
        return load_checkpoint(
            state=ctx.state,
            model=ctx.model,
            optimizer=ctx.optimizer,
            opt_param_scheduler=ctx.opt_param_scheduler,
            strict=ctx.strict,
            checkpointing_context=self._context,
            skip_load_to_model_and_opt=ctx.skip_load_to_model_and_opt,
            pg_collection=getattr(ctx, "pg_collection", None),
            module_name=getattr(ctx, "module_name", None),
        )

    def _find_ml_flashpoint_checkpoint(self) -> Optional[CheckpointContainerId]:
        """Finds the latest recoverable ML Flashpoint container, if any.

        Returns:
            The container, or None.
        """
        if self._ensure_runtime() is None:
            return None
        self._install_local_index()
        return self._local_index.resolve_latest_container()

    @log_execution_time(logger=_LOGGER, name="MLFlashpointBridgeCheckpointManager.mlf_load", level=logging.INFO)
    def _load_ml_flashpoint(self, ctx, container: CheckpointContainerId) -> Optional[tuple[int, int]]:
        """Loads from an ML Flashpoint container.

        The read itself is allowed to fail (the caller then falls back to
        Megatron Bridge), but once state has been applied to the model a failure
        is fatal, because there is no consistent state to fall back from.

        Args:
            ctx: The Bridge ``CheckpointLoadContext``.
            container: The container to read.

        Returns:
            A tuple of (step, cumulative FLOPs), or None when the read failed and
            the caller should fall back.
        """
        try:
            sharded_state_dict = bridge_state.build_load_state_dict(ctx)
            state_dict = mcore_dist_checkpointing.load(
                sharded_state_dict=sharded_state_dict,
                checkpoint_dir=str(container),
                sharded_strategy=self._runtime.load_strategy,
                common_strategy=TorchCommonLoadStrategy(),
            )
        except Exception:
            _LOGGER.exception(
                "Failed to read the ML Flashpoint checkpoint at '%s'. Falling back to Megatron Bridge.", container
            )
            return None

        step, flops = bridge_state.apply_loaded_state(ctx, state_dict)
        _LOGGER.info("Recovered from ML Flashpoint checkpoint '%s' at step %d", container, step)
        return step, flops

    @log_execution_time(
        logger=_LOGGER, name="MLFlashpointBridgeCheckpointManager.finalize_async_saves", level=logging.DEBUG
    )
    def finalize_async_saves(self, state, blocking: bool = False, terminate: bool = False) -> None:
        """Finalizes pending saves on both the ML Flashpoint and Bridge queues.

        The queues are finalized independently: ML Flashpoint saves complete far
        sooner than durable ones, and making them wait behind a durable save
        would keep their buffers pinned long enough to exhaust the pool.

        Args:
            state: The Bridge ``GlobalState``.
            blocking: If True, waits for every pending save.
            terminate: If True, tears down the queues afterwards.
        """
        if self._runtime is not None:
            try:
                self._runtime.maybe_finalize(blocking=blocking)
            except Exception:
                _LOGGER.exception("Failed to finalize ML Flashpoint async saves.")

        maybe_finalize_async_save(
            global_state=state,
            ckpt_cfg=self.checkpoint_config,
            blocking=blocking,
            terminate=terminate,
        )

        if terminate:
            self.shutdown()

    def shutdown(self) -> None:
        """Releases ML Flashpoint resources held by this rank.

        Waits for in-flight saves and synchronizes across ranks before deleting
        anything, so a peer is never mid-replication into a container that is
        about to disappear.
        """
        if self._runtime is None:
            return
        try:
            self._runtime.maybe_finalize(blocking=True)
        except Exception:
            _LOGGER.exception("Failed to drain ML Flashpoint async saves during shutdown.")

        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                _LOGGER.exception("Barrier before ML Flashpoint teardown failed. Continuing.")

        from ml_flashpoint.adapter.megatron_bridge.runtime import shutdown_runtime

        shutdown_runtime(remove_checkpoints=not self._mlf_config.keep_checkpoints_on_finalize)
        self._runtime = None
        self._local_index = None
        self._context.pop("local_checkpoint_manager", None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def init_checkpointing_context_safely(checkpoint_config) -> dict[str, Any]:
    """Builds Bridge's checkpointing context without requiring NVRx.

    ``init_checkpointing_context`` insists on ``nvidia_resiliency_ext`` when
    ``non_persistent_ckpt_type='local'``, but with this adapter that setting means
    "ML Flashpoint owns local checkpoints", so NVRx is not needed.

    Args:
        checkpoint_config: The Bridge ``CheckpointConfig``.

    Returns:
        The checkpointing context dictionary.
    """
    try:
        return init_checkpointing_context(checkpoint_config)
    except RuntimeError:
        _LOGGER.info(
            "Megatron Bridge could not build a local checkpointing context (nvidia_resiliency_ext is not "
            "installed). ML Flashpoint provides local checkpointing instead; continuing with an empty context."
        )
        return {}
