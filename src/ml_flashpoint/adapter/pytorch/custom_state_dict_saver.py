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

"""Provides customized state dict saving functionalities.

NOTE: This code is largely copied and tweaked from PyTorch:
https://sourcegraph.com/github.com/pytorch/pytorch@v2.8.0/-/blob/torch/distributed/checkpoint/state_dict_saver.py
"""

import logging

import torch.cuda
from torch import distributed as torchdist
from torch.distributed.checkpoint import state_dict_saver as torchdistsaver
from torch.distributed.checkpoint.logger import _dcp_method_logger
from torch.distributed.checkpoint.planner import SavePlan
from torch.distributed.checkpoint.storage import WriteResult
from torch.distributed.checkpoint.utils import _DistWrapper

from ml_flashpoint.adapter.pytorch.memory_storage_writer import MemoryStorageWriter
from ml_flashpoint.core import utils
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId
from ml_flashpoint.core.checkpoint_saver import ObjectWriteBucket
from ml_flashpoint.core.mlf_logging import get_logger
from ml_flashpoint.core.utils import log_execution_time

_LOGGER = get_logger(__name__)


@log_execution_time(logger=_LOGGER, name="generate_plan", level=logging.INFO)
def generate_plan(
    checkpoint_id: CheckpointContainerId,
    state_dict: torchdistsaver.STATE_DICT_TYPE,
    storage_writer: MemoryStorageWriter,
    planner: torchdistsaver.SavePlanner,
    world_dist_wrapper: _DistWrapper,
    cached_ckpt_structure: tuple[SavePlan, SavePlan, bool] | None = None,
    loaded_all_plans: list[SavePlan] | None = None,
) -> tuple[
    tuple[SavePlan, list[ObjectWriteBucket], torchdistsaver.Metadata | None],
    SavePlan,
    SavePlan,
    bool,
]:
    """Performs the planning phase of checkpointing.

    This function is similar to PyTorch's `state_dict_saver.save` but only
    handles the planning stage. It generates the save plan for the given state dict
    without actually writing data.
    Also see Megatron's equivalent implementation:
    :func:`megatron.core.dist_checkpointing.startegies.state_dict_saver.save_state_dict_async_plan`,

    Args:
        checkpoint_id: The CheckpointContainerId representing the checkpoint version to plan for.
        state_dict: The state dict to be saved. Need not be staged yet.
        storage_writer: The StorageWriter to use for the save.
        planner: The SavePlanner to use for the save.
        world_dist_wrapper: The distributed wrapper for world (all ranks) communication.
            Typically created as `_DistWrapper(process_group, not no_dist, coordinator_rank)`.
        cached_ckpt_structure: Tuple of (cached_central_plan, cached_local_plan, validated_cache_reuse).
        loaded_all_plans: List of all local plans from the previous checkpoint (for validation).

    Returns:
        A tuple containing:
            - (final_local_plan, write_buckets, global_metadata)
            - final_local_plan (for caching)
            - local_plan (for caching)
            - validated_cache_reuse (bool)
    """
    cached_central_plan, cached_local_plan, validated_cache_reuse = (None, None, False)
    if cached_ckpt_structure:
        cached_central_plan, cached_local_plan, validated_cache_reuse = cached_ckpt_structure

    global_metadata: torchdistsaver.Metadata | None = None

    ckpt_kwargs = {"checkpoint_id": storage_writer.current_checkpoint_id, "process_group": world_dist_wrapper.group}
    local_plan = cached_local_plan

    @_dcp_method_logger(**ckpt_kwargs)
    def local_step() -> SavePlan:
        nonlocal local_plan
        storage_meta = storage_writer.storage_meta()
        planner.set_up_planner(
            state_dict=state_dict,
            storage_meta=storage_meta,
            is_coordinator=world_dist_wrapper.is_coordinator,
        )
        storage_writer.set_up_storage_writer(world_dist_wrapper.is_coordinator)

        if not validated_cache_reuse:
            local_plan = planner.create_local_plan()

        local_plan = storage_writer.prepare_local_plan(local_plan)
        return local_plan

    @_dcp_method_logger(**ckpt_kwargs)
    def global_step(all_local_plans: list[SavePlan]) -> list[SavePlan]:
        nonlocal global_metadata

        all_local_plans, global_metadata = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        return all_local_plans

    central_plan = None
    if validated_cache_reuse and cached_central_plan:
        _LOGGER.debug("Passed cache reusable")
        local_step()
        central_plan = cached_central_plan
    else:
        with log_execution_time(logger=_LOGGER, name="generate_plan__reduce_scatter_plan"):
            _LOGGER.debug("Executing plan reduce_scatter to get central_plan...")
            central_plan = world_dist_wrapper.reduce_scatter("plan", local_step, global_step)

        with log_execution_time(logger=_LOGGER, name="generate_plan__broadcast_metadata"):
            _LOGGER.debug("Executing global_metadata broadcast...")
            global_metadata = world_dist_wrapper.broadcast_object(global_metadata)

    final_local_plan = planner.finish_plan(central_plan)
    write_buckets = storage_writer.prepare_write_data_buckets(checkpoint_id, final_local_plan, planner)

    return (
        (final_local_plan, write_buckets, global_metadata),
        central_plan,  # cached_central_plan
        local_plan,  # cached_local_plan
        cached_central_plan == central_plan,  # validated_cache_reuse
    )


@log_execution_time(logger=_LOGGER, name="write_data", level=logging.INFO)
def write_data(
    checkpoint_id: CheckpointContainerId,
    storage_writer: MemoryStorageWriter,
    staged_write_buckets: list[ObjectWriteBucket],
    replicate_after_write: bool,
) -> list[WriteResult]:
    """Writes the data to storage based on the provided plan.

    This function takes the central plan and uses the storage writer to write
    the data.

    Args:
        checkpoint_id: The CheckpointContainerId representing the checkpoint version to write for.
        storage_writer: The StorageWriter to use for writing data.
        staged_write_buckets: A list of ObjectWriteBucket objects to be written.
        replicate_after_write: Whether to replicate the written objects after writing is complete.

    Returns:
        A list of WriteResult objects.
    """
    all_writes_future = storage_writer.write_staged_data_buckets(
        checkpoint_id, staged_write_buckets, replicate_after_write
    )

    return all_writes_future.wait()


@log_execution_time(logger=_LOGGER, name="finish_write", level=logging.INFO)
def finish_write(
    checkpoint_id: CheckpointContainerId,
    storage_writer: MemoryStorageWriter,
    global_metadata: torchdistsaver.Metadata,
    world_dist_wrapper: _DistWrapper,
) -> None:
    """Finishes the writing process and writes the global metadata (once on each node).

    This includes gathering results, broadcasting metadata, and calling the
    StorageWriter's finish method once on each node.

    Should only be invoked AFTER the StorageWriter's write_data is confirmed complete.

    Args:
        checkpoint_id: The CheckpointContainerId representing the checkpoint version to finish the write for.
        storage_writer: The StorageWriter used for writing.
        global_metadata: The global metadata for the checkpoint.
        world_dist_wrapper: The distributed wrapper for world (all ranks) communication.
    """
    # all_results_lists is collected on the global coordinator rank (global rank 0),
    # and consists of each rank's list[WriteResult].
    all_results_lists = world_dist_wrapper.gather_object(storage_writer.get_write_results(checkpoint_id))
    # TODO(perf): consider optimizing comms across local rank 0s, but currently seems to be causing issues
    # local_rank0_dist_wrapper = _DistWrapper(
    #     torchdist.new_group(ranks=_get_local_rank_0_global_ranks()),
    #     use_dist=not disable_dist,
    #     coordinator_rank=0,
    # )

    # global_metadata is broadcast from the global coordinator rank, and received on all local rank 0s
    with log_execution_time(logger=_LOGGER, name="finish_write__broadcast_results"):
        _LOGGER.debug("Broadcasting write results across all local rank 0s...")
        all_results_lists = world_dist_wrapper.broadcast_object(all_results_lists)

    if torchdist.get_node_local_rank() == 0:
        _LOGGER.info("Calling finish_checkpoint to write global metadata...")
        storage_writer.finish_checkpoint(
            checkpoint_id=checkpoint_id, metadata=global_metadata, results=all_results_lists
        )


def _get_local_rank_0_global_ranks() -> list[int]:
    """Determines which global ranks are the 0th rank on each node.

    Relies on the number of GPUs per node to compute the global ranks.

    Returns:
        A list of global ranks that are local rank 0 on their respective nodes.
    """
    num_gpus_per_node = torch.cuda.device_count()
    return [i * num_gpus_per_node for i in range(utils.get_num_of_nodes())]
