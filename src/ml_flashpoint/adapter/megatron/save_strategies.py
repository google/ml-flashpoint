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

import json
import logging
import os
from functools import partial
from pathlib import Path
from typing import Union

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
from megatron.core.dist_checkpointing.strategies.base import AsyncSaveShardedStrategy
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreSavePlanner,
    _replace_state_dict_keys_with_sharded_keys,
    mcore_to_pyt_state_dict,
)
from torch.distributed.checkpoint.utils import _DistWrapper
from typing_extensions import override

from ml_flashpoint.adapter.pytorch import custom_state_dict_saver as statedictsaver
from ml_flashpoint.adapter.pytorch.memory_storage_writer import MemoryStorageWriter
from ml_flashpoint.core import utils
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId
from ml_flashpoint.core.checkpoint_saver import MLFlashpointCheckpointSaver, ObjectWriteBucket
from ml_flashpoint.core.mlf_logging import get_logger
from ml_flashpoint.core.utils import log_execution_time

_LOGGER = get_logger(__name__)


def default_backend_format_name() -> str:
    return "ml_flashpoint"


def default_backend_format_version() -> int:
    return 1


class MLFlashpointMegatronAsyncSaveStrategy(AsyncSaveShardedStrategy):
    """
    Asynchronous checkpoint save strategy using ML Flashpoint.

    This strategy leverages the ML Flashpoint library to save sharded model
    checkpoints asynchronously.
    """

    def __init__(
        self,
        storage_writer: MemoryStorageWriter,
        backend: str = default_backend_format_name(),
        version: int = default_backend_format_version(),
    ):
        """
        Args:
            storage_writer (MemoryStorageWriter): The storage writer to use for saving operations.
            backend (str, optional): The name of the backend format. Defaults to "ml_flashpoint", which is recommended.
            version (int, optional): The version of the checkpoint format. Defaults to the latest version.
        """
        super().__init__(backend=backend, version=version)
        self._storage_writer: MemoryStorageWriter = storage_writer
        self._checkpoint_saver: MLFlashpointCheckpointSaver = storage_writer.checkpoint_saver

    @override
    def can_handle_sharded_objects(self) -> bool:
        # Not currently used, but in case it is, ensure this strategy is used for ShardedObjects as well.
        return True

    @override
    @log_execution_time(logger=_LOGGER, name="async_save", level=logging.INFO)
    def async_save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Union[str, Path]) -> AsyncRequest:
        """Implements async checkpoint saving according to AsyncSaveShardedStrategy's interface.

        Args:
            sharded_state_dict: The state dictionary to save.
            checkpoint_dir: The checkpoint ID (typically in the form of some hierarchical path uniquely identifying
                this checkpoint container).

        Returns:
            An AsyncRequest encapsulating the work to be done. Note that in accordance with existing
            implementations, the returned AsyncRequest MUST be scheduled by the caller,
            and the `preload_fn` MUST be invoked before or during scheduling.

        Raises:
            ValueError: If the `checkpoint_dir` is invalid.
        """
        # This strongly type wraps the checkpoint dir and validates it.
        checkpoint_id = CheckpointContainerId(checkpoint_dir)

        # 1a. First, initialize the checkpoint. This marks this checkpoint container as "dirty".
        # This must always be the very first operation.
        self._checkpoint_saver.initialize_checkpoint(checkpoint_id)
        # 1b. Re-initialize the StorageWriter to use a new instance per save to avoid hangs from shared state.
        self._storage_writer = MemoryStorageWriter(
            checkpoint_saver=self._checkpoint_saver,
            mp_manager=self._storage_writer._mp_manager,
            thread_count=self._storage_writer._thread_count,
        )
        # 1c. Reset the StorageWriter for this checkpoint version.
        self._storage_writer.reset(checkpoint_id.data)

        # 2. Flatten the state dict's keys for simplified iteration going forward, and replace keys with sharded keys.
        sharded_state_dict, _, _ = _replace_state_dict_keys_with_sharded_keys(
            sharded_state_dict=sharded_state_dict,
            keep_only_main_replica=True,  # TODO: ensure True is always safe to use
        )

        # 3. Convert to a PyTorch dist compatible format.
        pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict)

        # 4. Stage to CPU.
        # use_pyt_staging = str(utils.get_env_val_bool("USE_PYT_STAGING", False)).lower() == "true"
        # _LOGGER.debug("use_pyt_staging is: %s", use_pyt_staging)
        # pyt_state_dict = (
        #     self._storage_writer.stage(pyt_state_dict)
        #     if use_pyt_staging
        #     else self._checkpoint_saver.stage_data(checkpoint_id, pyt_state_dict, non_blocking=True)
        # )

        # 4. Plan and get global metadata.
        disable_dist = utils.get_env_val_bool("DISABLE_DIST", False)
        _LOGGER.debug("disable_dist is: %s", disable_dist)
        # Since loading has to be done via Megatron's LoadPlanner to satisfy certain expectations,
        # we also use Megatron's SavePlanner during saving for compatibility.
        planner: MCoreSavePlanner = MCoreSavePlanner(can_run_decentralized_global_plan=False)
        world_dist_wrapper = _DistWrapper(group=None, use_dist=not disable_dist, coordinator_rank=0)
        plan, write_buckets, global_metadata = statedictsaver.generate_plan(
            checkpoint_id=checkpoint_id,
            state_dict=pyt_state_dict,
            storage_writer=self._storage_writer,
            planner=planner,
            world_dist_wrapper=world_dist_wrapper,
        )

        # 5. Stage to CPU.
        # staged_write_buckets = self._storage_writer.stage_write_data_buckets(
        #     checkpoint_id, write_buckets, non_blocking=True
        # )
        statedictsaver.write_data(
            checkpoint_id=checkpoint_id,
            storage_writer=self._storage_writer,
            staged_write_buckets=write_buckets,
            replicate_after_write=False,
        )

        # Since loading will go through Megatron dist_checkpointing.load, which validates the metadata.json
        # file, we create a stub file to pass that validation. In practice this file is not actually used to
        # verify much, as all the checks are NO-OPs, but there's no other way to bypass it.
        metadata = {"sharded_backend": ""}
        with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        def _save_checkpoint(staged_buckets: list[ObjectWriteBucket]):
            """
            This function is the 'async_fn' run in Megatron's :class:`AsyncRequest`.
            """
            # statedictsaver.write_data(
            #     checkpoint_id=checkpoint_id,
            #     storage_writer=self._storage_writer,
            #     staged_write_buckets=staged_buckets,
            #     replicate_after_write=False,
            # )
            pass

        finalize_fns = [
            # Replicate written objects
            partial(
                self._storage_writer.replicate_written_objects,
                object_ids={bucket.object_id for bucket in write_buckets},
            ),
            # Update and write metadata
            partial(
                statedictsaver.finish_write,
                checkpoint_id=checkpoint_id,
                storage_writer=self._storage_writer,
                global_metadata=global_metadata,
                world_dist_wrapper=world_dist_wrapper,
            ),
            # Mark checkpoint as complete
            partial(
                self._checkpoint_saver.finalize_checkpoint,
                checkpoint_id=checkpoint_id,
            ),
        ]

        return AsyncRequest(
            async_fn=_save_checkpoint,
            async_fn_args=(),
            async_fn_kwargs={"staged_buckets": []},
            finalize_fns=finalize_fns,
        )
