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

import concurrent.futures
import logging
import multiprocessing
import os
from typing import Optional

import ray
import torch
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl

from ml_flashpoint.adapter.megatron.save_strategies import MLFlashpointMegatronAsyncSaveStrategy
from ml_flashpoint.adapter.megatron.save_utils import save_local_aware_megatron_checkpoint
from ml_flashpoint.adapter.pytorch.memory_storage_writer import MemoryStorageWriter
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.core.buffer_pool import BufferPoolConfig
from ml_flashpoint.core.checkpoint_saver import DefaultMLFlashpointCheckpointSaver
from ml_flashpoint.replication.replication_manager import ReplicationManager

_LOGGER = logging.getLogger(__name__)


class MLFlashpointMegatronPolicyWorkerImpl(MegatronPolicyWorkerImpl):
    """Custom Megatron Policy Worker that overrides save_checkpoint to use ML Flashpoint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mlf_save_strategy: Optional[MLFlashpointMegatronAsyncSaveStrategy] = None

        # Read ML Flashpoint config from self.cfg
        self.mlf_cfg = self.cfg.get("ml_flashpoint", {})
        self.mlf_enabled = self.mlf_cfg.get("enabled", True)
        self.flashpoint_base_container = self.mlf_cfg.get("base_container")

        # Initialize AsyncCallsQueue for ML Flashpoint
        self._mlf_async_queue = AsyncCallsQueue(persistent=True)

    def _init_mlf_strategy(self):
        """Lazily initialize ML Flashpoint save strategy on the worker."""
        if self._mlf_save_strategy is not None:
            return

        _LOGGER.info("[MLF Worker] Initializing ML Flashpoint strategy for rank %s", torch.distributed.get_rank())

        # 1. Initialize BufferPool and Object Manager
        pool_config = BufferPoolConfig(
            pool_dir_path=os.path.join(self.flashpoint_base_container, "buffer_pool"),
            rank=torch.distributed.get_rank(),
            num_buffers=self.mlf_cfg.get("write_thread_count", 1) * 2,
            buffer_size=int(self.mlf_cfg.get("buffer_size_bytes", 16 * 1024 * 1024 * 1024)),
        )
        ckpt_obj_manager = CheckpointObjectManager(pool_config=pool_config)

        # 2. Initialize Replication Manager
        replication_manager = ReplicationManager()
        replication_manager.initialize(checkpoint_object_manager=ckpt_obj_manager)

        # 3. Initialize Checkpoint Saver
        checkpoint_saver = DefaultMLFlashpointCheckpointSaver(
            global_rank_getter=torch.distributed.get_rank,
            local_rank_getter=torch.distributed.get_node_local_rank, # Assuming local rank can be determined or mapped
            global_barrier_func=torch.distributed.barrier,
            ckpt_obj_manager=ckpt_obj_manager,
            replication_manager=replication_manager,
            # We assume Ray worker context can use torch.distributed for rank getters and barriers
        )

        manager = multiprocessing.Manager()
        manager_future = concurrent.futures.Future()
        manager_future.set_result(manager)

        # 4. Initialize Storage Writer
        storage_writer = MemoryStorageWriter(
            checkpoint_saver=checkpoint_saver,
            mp_manager_future=manager_future,
            thread_count=self.mlf_cfg.get("write_thread_count", 1),
        )

        # 5. Initialize Strategy
        self._mlf_save_strategy = MLFlashpointMegatronAsyncSaveStrategy(storage_writer=storage_writer)

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        **kwargs,
    ):
        if self.mlf_enabled and not self.flashpoint_base_container:
            raise ValueError("flashpoint_base_container must be provided if ML Flashpoint is enabled.")
        if not self.mlf_enabled:
            return super().save_checkpoint(weights_path, optimizer_path, tokenizer_path, **kwargs)

        # Detect if this is an ML Flashpoint save by path
        is_mlf_save = os.path.abspath(weights_path).startswith(os.path.abspath(self.flashpoint_base_container))

        if not is_mlf_save:
            _LOGGER.debug("[MLF Worker] Standard save detected for path: %s", weights_path)
            return super().save_checkpoint(weights_path, optimizer_path, tokenizer_path, **kwargs)

        _LOGGER.info("[MLF Worker] ML Flashpoint save detected for path: %s", weights_path)
        self._init_mlf_strategy()

        # Build checkpoint dict matching Megatron Core dist_checkpointing save format
        checkpoint_dict = {
            "model": [self.model],
            "state": self.mcore_state,
        }

        if optimizer_path is not None:
            if hasattr(self, "optimizer") and self.optimizer is not None:
                checkpoint_dict["optimizer"] = self.optimizer
            if hasattr(self, "scheduler") and self.scheduler is not None:
                checkpoint_dict["opt_param_scheduler"] = self.scheduler

        if hasattr(self.mcore_state, "train_state"):
            checkpoint_dict["num_floating_point_operations_so_far"] = (
                self.mcore_state.train_state.floating_point_operations_so_far
            )
        if hasattr(self, "checkpointing_context"):
            checkpoint_dict["checkpointing_context"] = self.checkpointing_context

        async_request = save_local_aware_megatron_checkpoint(
            checkpoint=checkpoint_dict,
            checkpoint_dir=weights_path,
            save_strategy=self._mlf_save_strategy,
            async_save=True,
        )
        if async_request:
            self._mlf_async_queue.schedule_async_request(async_request)
            _LOGGER.info("[MLF Worker] Scheduled async ML Flashpoint checkpoint save to %s", weights_path)


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)
class MLFlashpointMegatronPolicyWorker(MLFlashpointMegatronPolicyWorkerImpl):
    """Empty Ray Remote class wrapping the implementation worker.

    This class serves two primary purposes:
      1. Ray Compatibility: NeMo RL's builder expects a class decorated with `@ray.remote` and uses `.options()`.
      2. Unit Testing: Implementation details reside inside `MLFlashpointMegatronPolicyWorkerImpl`
         to allow running standard pytest unit tests without spawning a real Ray cluster.

    Equivalently, NeMo RL creates `MegatronPolicyWorker` as an empty subclass of `MegatronPolicyWorkerImpl`
    decorated with `@ray.remote` in:
    https://github.com/NVIDIA-NeMo/RL/blob/29f58809310a621b1b36d9a473528f6d48ada909/nemo_rl/models/policy/workers/megatron_policy_worker.py#L1602-L1603
    """
    pass
