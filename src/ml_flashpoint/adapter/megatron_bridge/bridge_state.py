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

"""Translation between Megatron Bridge contexts and Megatron sharded state dicts.

Megatron Bridge builds its state dict inside ``save_checkpoint`` /
``load_checkpoint``, which also decide where the checkpoint goes. ML Flashpoint
needs the state dict but not the destination, so the pieces are rebuilt here
using Bridge's own helpers. Anything imported from Bridge that is private is
resolved defensively so that a Bridge upgrade degrades to a clear runtime error
rather than an import failure at module load.
"""

import importlib
import random
from typing import Any, Optional

import numpy as np
import torch
from megatron.bridge.training.checkpointing import (
    generate_state_dict,
    get_rng_state,
    set_checkpoint_version,
)
from megatron.bridge.training.state import TrainState
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.core import tensor_parallel
from megatron.core.num_microbatches_calculator import update_num_microbatches
from megatron.core.rerun_state_machine import get_rerun_state_machine

from ml_flashpoint.core.mlf_logging import get_logger

_LOGGER = get_logger(__name__)

TRAIN_STATE_KEY = "train_state_metadata"
"""Key under which the Bridge ``TrainState`` is embedded in the state dict.

Mirrors what Bridge itself does for local (non-persistent) checkpoints: local
checkpoints have no ``latest_train_state.pt`` tracker next to them, so the
counters have to travel inside the checkpoint.
"""

CONTENT_METADATA_KEY = "content_metadata"
"""Key under which the sharded-state-dict metadata is embedded."""

FLOPS_KEY = "num_floating_point_operations_so_far"
"""Key under which cumulative FLOPs are embedded."""

DP_CP_GROUP_KEY = "dp_cp_group"
"""Metadata key carrying the data/context-parallel process group.

Megatron's ``sharded_state_dict`` methods need it, but a process group cannot be
pickled, so it is stripped before the metadata is persisted.
"""


def unwrap_model(model):
    """Strips the DDP / Float16Module wrappers off each model chunk.

    ``unwrap_model`` moved into ``megatron.core.utils`` only in the Megatron Core
    releases that Megatron Bridge 0.6 requires, so it is resolved at call time and
    the older ``megatron.training.utils`` location is accepted as well.

    Args:
        model: A module or list of module chunks.

    Returns:
        The unwrapped chunks, always as a list.

    Raises:
        RuntimeError: If neither location provides the helper.
    """
    for module_path in ("megatron.core.utils", "megatron.training.utils"):
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        impl = getattr(module, "unwrap_model", None)
        if impl is not None:
            unwrapped = impl(model)
            return unwrapped if isinstance(unwrapped, list) else [unwrapped]
    raise RuntimeError(
        "Could not resolve unwrap_model from megatron.core.utils or megatron.training.utils. "
        "Install a Megatron Core version compatible with the Megatron Bridge release in use."
    )


def _import_optional(name: str):
    """Imports a Bridge symbol that is private and therefore may move.

    Args:
        name: Attribute name on ``megatron.bridge.training.checkpointing``.

    Returns:
        The attribute, or None when this Bridge version does not expose it.
    """
    import megatron.bridge.training.checkpointing as bridge_checkpointing

    return getattr(bridge_checkpointing, name, None)


def build_sharded_state_dict_metadata(use_distributed_optimizer: bool, ckpt_cfg) -> dict[str, Any]:
    """Builds the metadata Bridge passes to ``sharded_state_dict`` methods.

    Args:
        use_distributed_optimizer: Whether the run uses the distributed optimizer.
        ckpt_cfg: The Bridge ``CheckpointConfig``.

    Returns:
        The metadata dictionary.

    Raises:
        RuntimeError: If the installed Megatron Bridge does not expose the helper.
    """
    builder = _import_optional("_build_sharded_state_dict_metadata")
    if builder is None:
        raise RuntimeError(
            "megatron.bridge.training.checkpointing._build_sharded_state_dict_metadata is not available. "
            "This Megatron Bridge version is not supported by the ML Flashpoint adapter."
        )
    return builder(use_distributed_optimizer, ckpt_cfg)


def clean_metadata_for_serialization(metadata: dict[str, Any]) -> dict[str, Any]:
    """Strips non-serializable entries (e.g. process groups) from metadata.

    Args:
        metadata: The sharded-state-dict metadata.

    Returns:
        A copy safe to persist, or the input unchanged when the Megatron helper
        is unavailable.
    """
    try:
        from megatron.core.dist_checkpointing.utils import _clean_metadata_for_serialization

        return _clean_metadata_for_serialization(metadata)
    except ImportError:
        _LOGGER.warning("Could not import _clean_metadata_for_serialization; dropping process groups manually.")
        return {k: v for k, v in metadata.items() if not isinstance(v, torch.distributed.ProcessGroup)}


def resolve_pg_collection(model: list, pg_collection=None):
    """Returns the process group collection to use.

    Args:
        model: The (unwrapped) model modules.
        pg_collection: An explicit collection from the Bridge context, if any.

    Returns:
        The resolved ``ProcessGroupCollection``.
    """
    return pg_collection if pg_collection is not None else get_pg_collection(model)


def build_save_state_dict(
    ctx,
    step: int,
    num_floating_point_operations_so_far: int,
) -> dict[str, Any]:
    """Builds the sharded state dict for an ML Flashpoint save.

    Reproduces the portion of Bridge's ``save_checkpoint`` that assembles state,
    and additionally embeds the train state, cumulative FLOPs and content
    metadata, none of which have a home on disk for an ML Flashpoint container.

    Args:
        ctx: The Bridge ``CheckpointSaveContext``.
        step: The training step this checkpoint represents.
        num_floating_point_operations_so_far: Cumulative FLOPs to embed.

    Returns:
        The state dict to hand to the ML Flashpoint save strategy.
    """
    cfg = ctx.state.cfg
    ckpt_cfg = cfg.checkpoint
    model = unwrap_model(ctx.model)
    pg_collection = resolve_pg_collection(model, getattr(ctx, "pg_collection", None))

    rng_state = None
    if ckpt_cfg.save_rng:
        rng_state = get_rng_state(
            data_parallel_random_init=cfg.rng.data_parallel_random_init,
            ckpt_format=ckpt_cfg.ckpt_format,
            pg_collection=pg_collection,
            module_name=getattr(ctx, "module_name", None),
        )

    rerun_state = get_rerun_state_machine().state_dict(
        data_iterator=ctx.train_data_iterator,
        ckpt_format=ckpt_cfg.ckpt_format,
    )

    sharded_sd_metadata = build_sharded_state_dict_metadata(cfg.optimizer.use_distributed_optimizer, ckpt_cfg)
    # The process group is needed by sharded_state_dict() but must not be persisted.
    sharded_sd_metadata[DP_CP_GROUP_KEY] = pg_collection.dp_cp

    state_dict = generate_state_dict(
        ckpt_cfg,
        model,
        ctx.optimizer,
        ctx.opt_param_scheduler,
        rng_state,
        iteration=step,
        optim_sd_kwargs=dict(metadata=sharded_sd_metadata),
        model_sd_kwargs=dict(metadata=sharded_sd_metadata),
        rerun_state=rerun_state,
        pg_collection=pg_collection,
    )

    state_dict[TRAIN_STATE_KEY] = ctx.state.train_state.state_dict()
    state_dict[FLOPS_KEY] = int(num_floating_point_operations_so_far)
    # Drop the process group explicitly rather than trusting the Megatron cleaner
    # to recognize it: it is the one entry this adapter adds, and it is the one
    # entry that cannot be pickled into common.pt.
    persistable_metadata = {k: v for k, v in sharded_sd_metadata.items() if k != DP_CP_GROUP_KEY}
    state_dict[CONTENT_METADATA_KEY] = clean_metadata_for_serialization(persistable_metadata)
    return state_dict


def build_load_state_dict(ctx) -> dict[str, Any]:
    """Builds the sharded state dict skeleton used to load an ML Flashpoint save.

    The skeleton mirrors Bridge's local-checkpoint load path: TP/PP are assumed
    unchanged (an ML Flashpoint checkpoint is only ever recovered by the same
    job with the same parallelism), so no run-config comparison is performed.

    Args:
        ctx: The Bridge ``CheckpointLoadContext``.

    Returns:
        The sharded state dict to pass to ``dist_checkpointing.load``.
    """
    cfg = ctx.state.cfg
    ckpt_cfg = cfg.checkpoint
    model = unwrap_model(ctx.model)
    pg_collection = resolve_pg_collection(model, getattr(ctx, "pg_collection", None))

    rng_state = None
    if ckpt_cfg.load_rng:
        rng_state = get_rng_state(
            data_parallel_random_init=cfg.rng.data_parallel_random_init,
            ckpt_format=ckpt_cfg.ckpt_format,
            pg_collection=pg_collection,
            module_name=getattr(ctx, "module_name", None),
        )

    rerun_state = get_rerun_state_machine().state_dict(
        data_iterator=None,
        ckpt_format=ckpt_cfg.ckpt_format,
        force=True,
    )

    sharded_sd_metadata = build_sharded_state_dict_metadata(cfg.optimizer.use_distributed_optimizer, ckpt_cfg)
    sharded_sd_metadata[DP_CP_GROUP_KEY] = pg_collection.dp_cp

    load_optimizer = ckpt_cfg.load_optim and not ckpt_cfg.finetune
    return generate_state_dict(
        ckpt_cfg,
        model,
        ctx.optimizer if load_optimizer else None,
        ctx.opt_param_scheduler if load_optimizer else None,
        rng_state,
        optim_sd_kwargs=dict(metadata=sharded_sd_metadata, is_loading=True),
        model_sd_kwargs=dict(metadata=sharded_sd_metadata),
        rerun_state=rerun_state,
        pg_collection=pg_collection,
    )


def restore_train_state(state, state_dict: dict[str, Any]) -> None:
    """Restores the Bridge ``TrainState`` from a loaded ML Flashpoint state dict.

    Args:
        state: The Bridge ``GlobalState`` to mutate.
        state_dict: The loaded state dict.
    """
    if TRAIN_STATE_KEY in state_dict:
        state.train_state = TrainState()
        state.train_state.load_state_dict(state_dict[TRAIN_STATE_KEY])
    else:
        _LOGGER.warning("'%s' missing from the checkpoint; training counters reset.", TRAIN_STATE_KEY)
        state.train_state = TrainState()
        state.train_state.step = state_dict.get("iteration", 0)

    if FLOPS_KEY in state_dict:
        state.train_state.floating_point_operations_so_far = state_dict[FLOPS_KEY]


def restore_model(model: list, state_dict: dict[str, Any], strict: bool) -> None:
    """Loads model weights from a loaded state dict.

    Args:
        model: The unwrapped model modules.
        state_dict: The loaded state dict.
        strict: Whether to enforce strict key matching.

    Raises:
        RuntimeError: If the installed Megatron Bridge does not expose the helper.
    """
    loader = _import_optional("_load_model_state_dict")
    if loader is None:
        raise RuntimeError(
            "megatron.bridge.training.checkpointing._load_model_state_dict is not available. "
            "This Megatron Bridge version is not supported by the ML Flashpoint adapter."
        )
    if len(model) == 1:
        loader(model[0], state_dict["model"], strict)
        return
    for i in range(len(model)):
        model_key = "model%d" % i
        if model_key not in state_dict:
            # Empty pipeline stage.
            continue
        loader(model[i], state_dict[model_key], strict)


def restore_optimizer(ctx, state_dict: dict[str, Any]) -> None:
    """Loads optimizer and scheduler state from a loaded state dict.

    Args:
        ctx: The Bridge ``CheckpointLoadContext``.
        state_dict: The loaded state dict.
    """
    ckpt_cfg = ctx.state.cfg.checkpoint
    if ckpt_cfg.finetune or not ckpt_cfg.load_optim:
        return

    optimizer = ctx.optimizer
    if (
        not ctx.skip_load_to_model_and_opt
        and optimizer is not None
        and not getattr(optimizer, "is_stub_optimizer", False)
        and "optimizer" in state_dict
    ):
        # no_grad is required because DistributedOptimizer copies the loaded
        # tensors into main params with .copy_(), which rejects leaf Variables
        # that require grad.
        with torch.no_grad():
            optimizer.load_state_dict(state_dict["optimizer"])

    if ctx.opt_param_scheduler is not None:
        scheduler_state = state_dict.get("lr_scheduler", state_dict.get("opt_param_scheduler"))
        if scheduler_state is not None:
            ctx.opt_param_scheduler.load_state_dict(scheduler_state)


def restore_rerun_state(state_dict: dict[str, Any]) -> None:
    """Restores the rerun state machine, logging and continuing on failure.

    Args:
        state_dict: The loaded state dict.
    """
    if "rerun_state_machine" not in state_dict:
        return
    try:
        get_rerun_state_machine().load_state_dict(state_dict["rerun_state_machine"])
    except Exception:
        _LOGGER.exception("Unable to restore the rerun state machine. Skipping.")


def restore_rng_state(ctx, state_dict: dict[str, Any], pg_collection) -> None:
    """Restores RNG state, logging and continuing on failure.

    Args:
        ctx: The Bridge ``CheckpointLoadContext``.
        state_dict: The loaded state dict.
        pg_collection: The process group collection.
    """
    cfg = ctx.state.cfg
    if cfg.checkpoint.finetune or not cfg.checkpoint.load_rng or "rng_state" not in state_dict:
        return
    try:
        rng_states = state_dict["rng_state"]
        rng_state = rng_states[pg_collection.dp.rank()] if cfg.rng.data_parallel_random_init else rng_states[0]
        random.setstate(rng_state["random_rng_state"])
        np.random.set_state(rng_state["np_rng_state"])
        torch.set_rng_state(rng_state["torch_rng_state"])
        torch.cuda.set_rng_state(rng_state["cuda_rng_state"])
        tracker_states = rng_state["rng_tracker_states"]
        if not tracker_states:
            raise KeyError("rng_tracker_states is empty")
        cuda_rng_tracker = tensor_parallel.get_cuda_rng_tracker()
        # Graph-safe RNG conversion only exists on the Megatron Core releases that
        # ship CUDA-graph-capturable trackers; older builds store states directly.
        is_graph_safe = getattr(tensor_parallel, "is_graph_safe_cuda_rng_tracker", None)
        convert = getattr(tensor_parallel, "convert_cuda_rng_state", None)
        if is_graph_safe is not None and convert is not None:
            graph_safe_rng = is_graph_safe(cuda_rng_tracker)
            tracker_states = {k: convert(v, to_graphable=graph_safe_rng) for k, v in tracker_states.items()}
        cuda_rng_tracker.set_states(tracker_states)
    except Exception:
        _LOGGER.exception("Unable to restore RNG state from the ML Flashpoint checkpoint. Continuing without it.")


def apply_loaded_state(ctx, state_dict: dict[str, Any]) -> tuple[int, int]:
    """Applies a loaded ML Flashpoint state dict to the training state.

    Args:
        ctx: The Bridge ``CheckpointLoadContext``.
        state_dict: The state dict returned by ``dist_checkpointing.load``.

    Returns:
        A tuple of (step, cumulative floating point operations).
    """
    state = ctx.state
    model = unwrap_model(ctx.model)
    pg_collection = resolve_pg_collection(model, getattr(ctx, "pg_collection", None))

    set_checkpoint_version(state_dict.get("checkpoint_version", 0))
    restore_train_state(state, state_dict)
    update_num_microbatches(consumed_samples=state.train_state.consumed_train_samples, verbose=True)

    if not ctx.skip_load_to_model_and_opt:
        restore_model(model, state_dict, ctx.strict)
    restore_optimizer(ctx, state_dict)
    restore_rerun_state(state_dict)
    restore_rng_state(ctx, state_dict, pg_collection)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return state.train_state.step, state.train_state.floating_point_operations_so_far


def get_content_metadata(state_dict: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Returns the embedded content metadata, if present.

    Args:
        state_dict: A loaded state dict.

    Returns:
        The content metadata, or None.
    """
    return state_dict.get(CONTENT_METADATA_KEY)
