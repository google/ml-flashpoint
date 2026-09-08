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

"""Per-rank ML Flashpoint runtime shared by the Megatron Bridge integrations.

This owns the objects that must exist exactly once per rank (buffer pool,
replication service, save/load strategies, async queue) and keeps their
construction out of the checkpoint manager itself so that NeMo RL, which does
not go through Megatron Bridge's checkpoint-manager factory, can reuse them.
"""

import concurrent.futures
import os
import threading
from typing import Optional

import torch.distributed as dist
from megatron.core.dist_checkpointing.strategies.async_utils import (
    AsyncCallsQueue,
    AsyncRequest,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from torch import multiprocessing as torch_mp

from ml_flashpoint.adapter.megatron.load_strategies import MLFlashpointMegatronLoadStrategy
from ml_flashpoint.adapter.megatron.save_strategies import MLFlashpointMegatronAsyncSaveStrategy
from ml_flashpoint.adapter.megatron_bridge.config import MLFlashpointBridgeConfig
from ml_flashpoint.adapter.pytorch.memory_storage_writer import MemoryStorageWriter
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.core.buffer_pool import BufferPoolConfig
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId
from ml_flashpoint.core.checkpoint_loader import DefaultMLFlashpointCheckpointLoader
from ml_flashpoint.core.checkpoint_saver import DefaultMLFlashpointCheckpointSaver
from ml_flashpoint.core.mlf_logging import get_logger
from ml_flashpoint.replication.replication_manager import ReplicationManager

_LOGGER = get_logger(__name__)

NUM_OF_BUFFERS_PER_OBJECT = 2
"""Buffers reserved per writer thread, matching the NeMo adapter's pool sizing."""


class MLFlashpointBridgeRuntime:
    """Owns the per-rank ML Flashpoint objects used by the Bridge adapter.

    Instances are expensive (they start a buffer pool, a transfer service and a
    persistent async worker), so a single instance is shared for the lifetime of
    a training process. Use :func:`get_runtime` rather than constructing one
    directly unless a test needs isolation.
    """

    def __init__(self, config: MLFlashpointBridgeConfig):
        """Builds the runtime and initializes replication.

        Requires ``torch.distributed`` to already be initialized, because
        replication address exchange is a collective.

        Args:
            config: The ML Flashpoint settings to apply.

        Raises:
            RuntimeError: If ``torch.distributed`` is not initialized.
        """
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "ML Flashpoint requires an initialized torch.distributed process group. "
                "Build the runtime after Megatron initialization."
            )

        self._config = config
        self._base_container = CheckpointContainerId(config.base_container)
        self._closed = False

        pool_config = BufferPoolConfig(
            pool_dir_path=os.path.join(str(self._base_container), "buffer_pool"),
            rank=dist.get_rank(),
            num_buffers=config.write_thread_count * NUM_OF_BUFFERS_PER_OBJECT,
            buffer_size=config.initial_write_buffer_size_bytes,
        )
        self._checkpoint_object_manager = CheckpointObjectManager(pool_config=pool_config)

        self._replication_manager = ReplicationManager()
        self._replication_manager.initialize(checkpoint_object_manager=self._checkpoint_object_manager)

        # 'spawn' avoids inheriting the parent's CUDA context in the SyncManager
        # process. A forked manager that outlives a SIGKILLed trainer keeps GPU
        # memory locked and makes the in-job restart OOM.
        ctx = torch_mp.get_context("spawn")
        self._mp_manager_future: concurrent.futures.Future = concurrent.futures.Future()

        def start_manager():
            self._mp_manager_future.set_result(ctx.Manager())

        threading.Thread(target=start_manager, daemon=True).start()

        self._checkpoint_saver = DefaultMLFlashpointCheckpointSaver(
            global_rank_getter=dist.get_rank,
            local_rank_getter=dist.get_node_local_rank,
            global_barrier_func=dist.barrier,
            ckpt_obj_manager=self._checkpoint_object_manager,
            replication_manager=self._replication_manager,
            initial_buffer_size_bytes=config.initial_write_buffer_size_bytes,
            use_optimized_save=config.use_optimized_save,
        )
        self._checkpoint_loader = DefaultMLFlashpointCheckpointLoader(
            self._checkpoint_object_manager,
            self._replication_manager,
            global_rank_getter=dist.get_rank,
            local_rank_getter=dist.get_node_local_rank,
            broadcast_object_list_func=dist.broadcast_object_list,
            all_gather_object_func=dist.all_gather_object,
            world_size_getter=dist.get_world_size,
        )

        save_strategy = MLFlashpointMegatronAsyncSaveStrategy(
            storage_writer=MemoryStorageWriter(
                checkpoint_saver=self._checkpoint_saver,
                mp_manager_future=self._mp_manager_future,
                thread_count=config.write_thread_count,
            ),
            use_cached_ckpt_structure=config.use_cached_ckpt_structure,
        )
        load_strategy = MLFlashpointMegatronLoadStrategy(
            replication_manager=self._replication_manager,
            checkpoint_loader=self._checkpoint_loader,
        )
        if config.use_fully_parallel_wrapper:
            # No parallelization group is passed, matching the NeMo adapter: the
            # wrapper then distributes across the default (world) group.
            save_strategy = FullyParallelSaveStrategyWrapper(save_strategy)
            load_strategy = FullyParallelLoadStrategyWrapper(load_strategy)
        self._save_strategy = save_strategy
        self._load_strategy = load_strategy

        # Persistent so the worker process (and its buffer pool handles) is
        # reused across steps instead of respawned per checkpoint.
        self._async_calls_queue = AsyncCallsQueue(persistent=True)

    @property
    def config(self) -> MLFlashpointBridgeConfig:
        """The configuration this runtime was built with."""
        return self._config

    @property
    def base_container(self) -> CheckpointContainerId:
        """The base container holding one child container per checkpoint version."""
        return self._base_container

    @property
    def save_strategy(self):
        """The Megatron sharded save strategy backed by ML Flashpoint."""
        return self._save_strategy

    @property
    def load_strategy(self):
        """The Megatron sharded load strategy backed by ML Flashpoint."""
        return self._load_strategy

    @property
    def checkpoint_loader(self) -> DefaultMLFlashpointCheckpointLoader:
        """The loader used to discover and retrieve recoverable checkpoints."""
        return self._checkpoint_loader

    @property
    def checkpoint_object_manager(self) -> CheckpointObjectManager:
        """The object manager owning this rank's buffer pool."""
        return self._checkpoint_object_manager

    @property
    def replication_manager(self) -> ReplicationManager:
        """The replication manager used to mirror objects to peer nodes."""
        return self._replication_manager

    def schedule(self, async_request: AsyncRequest) -> int:
        """Schedules an async save request on the ML Flashpoint queue.

        Args:
            async_request: The request returned by the save strategy.

        Returns:
            The scheduled call index.
        """
        call_idx = self._async_calls_queue.schedule_async_request(async_request)
        _LOGGER.debug("Scheduled ML Flashpoint async call #%d", call_idx)
        return call_idx

    def num_unfinalized_calls(self) -> int:
        """Returns how many scheduled saves have not been finalized yet."""
        return self._async_calls_queue.get_num_unfinalized_calls()

    def maybe_finalize(self, blocking: bool = False) -> bool:
        """Finalizes completed saves.

        Args:
            blocking: If True, waits for every pending save to complete.

        Returns:
            True if at least one call was finalized.
        """
        if self._closed or self._async_calls_queue.get_num_unfinalized_calls() == 0:
            return False
        finalized = self._async_calls_queue.maybe_finalize_async_calls(blocking)
        if finalized:
            _LOGGER.debug("Finalized ML Flashpoint async calls: %s", [f"#{idx}" for idx in finalized])
        return len(finalized) > 0

    def shutdown(self, remove_checkpoints: bool = True) -> None:
        """Tears down the runtime, releasing buffers and background workers.

        Safe to call more than once.

        Args:
            remove_checkpoints: Whether to delete the base container so the
                node's memory is reclaimed.
        """
        if self._closed:
            return
        self._closed = True

        try:
            self._replication_manager.shutdown()
        except Exception:
            _LOGGER.exception("Failed to shut down the ReplicationManager. Continuing teardown.")

        if remove_checkpoints and self._is_local_rank_zero():
            try:
                self._checkpoint_object_manager.delete_container(self._base_container)
            except Exception:
                _LOGGER.exception("Failed to delete container '%s'. Continuing teardown.", self._base_container)

        # The buffer pool lives in the persistent worker process, so its teardown
        # has to be scheduled onto that same process.
        try:
            self._async_calls_queue.schedule_async_request(
                AsyncRequest(
                    async_fn=self._checkpoint_object_manager.teardown_pool,
                    async_fn_args=(),
                    finalize_fns=[],
                )
            )
        except Exception:
            _LOGGER.debug("Could not schedule buffer pool teardown; the queue is likely already closed.")

        self._async_calls_queue.close()
        # PersistentAsyncCaller.__del__ calls close(), which calls
        # torch.distributed.get_rank() and crashes if the process group is gone
        # by interpreter shutdown. The queue is already closed, so make the
        # second close a no-op.
        caller = getattr(self._async_calls_queue, "persistent_caller", None)
        if caller is not None and hasattr(caller, "close"):
            caller.close = lambda: None

    @staticmethod
    def _is_local_rank_zero() -> bool:
        try:
            return dist.get_node_local_rank() == 0
        except Exception:
            return not dist.is_initialized() or dist.get_rank() == 0


_RUNTIME: Optional[MLFlashpointBridgeRuntime] = None


def get_runtime(config: MLFlashpointBridgeConfig) -> MLFlashpointBridgeRuntime:
    """Returns the process-wide runtime, building it on first use.

    Args:
        config: The configuration used when the runtime has to be built. Ignored
            when a runtime already exists.

    Returns:
        The shared runtime instance.
    """
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = MLFlashpointBridgeRuntime(config)
    return _RUNTIME


def shutdown_runtime(remove_checkpoints: bool = True) -> None:
    """Shuts down and clears the process-wide runtime, if one was built.

    Args:
        remove_checkpoints: Whether to delete the base container.
    """
    global _RUNTIME
    if _RUNTIME is None:
        return
    try:
        _RUNTIME.shutdown(remove_checkpoints=remove_checkpoints)
    finally:
        _RUNTIME = None
