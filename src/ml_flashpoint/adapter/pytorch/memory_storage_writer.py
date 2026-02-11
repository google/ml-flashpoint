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
import dataclasses
import logging
import os
import time
from typing import Optional, Union

import torch
from torch import multiprocessing as torch_mp
from torch.distributed.checkpoint import Metadata, SavePlan, SavePlanner, StorageWriter, staging
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, MetadataIndex, StorageMeta
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future as TorchFuture
from typing_extensions import override

from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId, CheckpointObjectId
from ml_flashpoint.core.checkpoint_saver import MLFlashpointCheckpointSaver, ObjectWriteBucket
from ml_flashpoint.core.hfid import generate_hfid
from ml_flashpoint.core.mlf_logging import get_logger
from ml_flashpoint.core.utils import log_execution_time

_LOGGER = get_logger(__name__)


@dataclasses.dataclass
class _StorageDataContext:
    """Internal class to hold context for storage operations for a SavePlan.

    Attributes:
        prefix: A string prefix to uniquely identify objects belonging to a
            specific rank's SavePlan. Generated in prepare_global_plan
            and used in write_data to prepend to storage keys.
    """

    prefix: str


class MemoryStorageWriter(StorageWriter, staging.BlockingAsyncStager):
    """MemoryStorageWriter represents a local, in-memory StorageWriter implementation, with replication
    **to** peer node(s).

    It also implements BlockingAsyncStager, to customize staging behavior before write_data().

    This implementation prioritizes async writes, which can easily be made synchronous by blocking
    on the Futures returned by write_data().

    There are 2 ways to perform the stage + write steps using this Writer:
    1) The classic way - `stage() -> write_data()` - which will automatically replicate objects after write.
    2) The Megatron-async-friendly way: `prepare_write_data() -> stage_prepared_data() -> write_staged_write_buckets()`,
    where the output of `stage_prepared_data()` is the input to `write_staged_write_buckets()`, and you can
    control whether to replicate after write.

    Only one of these paths should be used, not both, along with the other APIs.

    Implementation notes:
     1. This class supports multiprocess sharing - specifically, it uses
     `torch.multiprocessing.Manager.dict()` for its internal write result state, so it can be used in spawned
     sub-processes, as Megatron and PyTorch DCP (optionally) do.

     2. Unlike Megatron's FileSystemWriterAsync, this StorageWriter can be reused across checkpoints (and hence tracks
     write state per checkpoint ID in case checkpointing overlaps across steps). This is mainly to facilitate using
     PyTorch DCP's pinned memory option for staging data.

     3. This class maintains a "current_checkpoint_id" attribute, that is used for abstract methods that expect
     this instance to know what checkpoint ID it is operating on. It is set via reset(), which is required to be called
     during checkpoint initialization for this purpose.
     However, where possible, it is recommended to provide the checkpoint_id explicitly. Since this instance is reused
     across multiple checkpoint IDs, which can overlap, explicitness avoids race conditions, especially for any
     operations done after the initial blocking phase (i.e. anything done async or after async operations).
    """

    def __init__(
        self,
        checkpoint_saver: MLFlashpointCheckpointSaver,
        mp_manager: torch_mp.Manager,
        thread_count: int = 1,
    ):
        """Initializes the MemoryStorageWriter.

        Args:
            checkpoint_saver: An instance of `MLFlashpointCheckpointSaver` used for
                handling the actual checkpoint saving logic.
            mp_manager: A `torch.multiprocessing.Manager` instance for managing
                shared state across processes, particularly for write results and events.
            thread_count: Optional. The number of threads to use for writing checkpoint data.
                Defaults to 1. If a value less than 1 is provided, it will be reset to 1,
                and a warning will be logged.
        """
        super().__init__()
        self._current_checkpoint_id: CheckpointContainerId | None = None
        self._current_save_id: str | None = None
        self._checkpoint_saver: MLFlashpointCheckpointSaver = checkpoint_saver
        if thread_count < 1:
            _LOGGER.warning("thread_count must be >= 1, but was %d. Setting to 1.", thread_count)
            thread_count = 1
        self._thread_count = thread_count
        # mp_manager should only be used in the main process, not in the spawned processes.
        # This is because mp_manager is not picklable.
        self._mp_manager = mp_manager
        self._write_events_per_checkpoint_id: dict[CheckpointContainerId, torch_mp.Event] = mp_manager.dict()
        self._write_results_per_checkpoint_id: dict[CheckpointContainerId, list[WriteResult]] = mp_manager.dict()

    def __getstate__(self):
        """Custom pickling to exclude unpicklable mp_manager."""
        state = self.__dict__.copy()
        if "_mp_manager" in state:
            del state["_mp_manager"]
        return state

    def __setstate__(self, state):
        """Custom unpickling to restore state and set mp_manager to None."""
        self.__dict__.update(state)
        self._mp_manager = None

    def _check_checkpoint_id(self) -> None:
        if self._current_checkpoint_id is None:
            raise ValueError("MemoryStorageWriter has not been reset. Call reset() before using this method.")

    @property
    def current_checkpoint_id(self) -> Optional[CheckpointContainerId]:
        return self._current_checkpoint_id

    @property
    def path(self) -> Optional[str]:
        if self._current_checkpoint_id is None:
            return None
        return self._current_checkpoint_id.data

    @property
    def checkpoint_saver(self) -> MLFlashpointCheckpointSaver:
        return self._checkpoint_saver

    @override
    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        self._current_checkpoint_id = CheckpointContainerId(checkpoint_id)
        # Mimicking existing StorageWriter impls (e.g. `_FileSystemWriter`) by using a random ID as the save ID.
        self._current_save_id = generate_hfid("memwritersave")

    def storage_meta(self) -> Optional[StorageMeta]:
        self._check_checkpoint_id()
        return StorageMeta(checkpoint_id=self._current_checkpoint_id.data, save_id=self._current_save_id)

    @override
    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        # Nothing to do here.
        pass

    @override
    @log_execution_time(logger=_LOGGER, name="prepare_local_plan")
    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        self._check_checkpoint_id()
        # Initialize checkpoint if it wasn't already. Compare to PyTorch's `_FileSystemWriter.prepare_local_plan`.
        _LOGGER.info("Initializing checkpoint '%s', just in case it wasn't already.", self._current_checkpoint_id)
        self._checkpoint_saver.initialize_checkpoint(self._current_checkpoint_id)
        return plan

    @override
    @log_execution_time(logger=_LOGGER, name="prepare_global_plan")
    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        # Taken from PyTorch's `_FileSystemWriter.prepare_global_plan`
        new_plans = [
            dataclasses.replace(plan, storage_data=_StorageDataContext(f"__{i}_")) for i, plan in enumerate(plans)
        ]
        return new_plans

    @override
    @log_execution_time(logger=_LOGGER, name="stage", level=logging.INFO)
    def stage(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
        # For now, we reuse the default implementation to have access to its built-in memory pinning feature.
        # Eventually we will customize this and leverage self._checkpoint_saver.stage_data() instead.
        return super().stage(state_dict)

    @log_execution_time(logger=_LOGGER, name="prepare_write_data_buckets")
    def prepare_write_data_buckets(
        self, checkpoint_id: CheckpointContainerId, plan: SavePlan, planner: SavePlanner
    ) -> list[ObjectWriteBucket]:
        # Create a new, unset Event for this specific checkpoint save
        if checkpoint_id not in self._write_events_per_checkpoint_id:
            self._write_events_per_checkpoint_id[checkpoint_id] = self._mp_manager.Event()

        write_buckets = self.checkpoint_saver.prepare_write_data(
            checkpoint_id, plan.items, planner, plan.storage_data.prefix, bucket_count=self._thread_count
        )
        return write_buckets
        # self._write_buckets_per_checkpoint_id[checkpoint_id] = write_buckets

    @staticmethod
    @log_execution_time(logger=_LOGGER, name="stage_write_data_buckets", level=logging.INFO)
    def stage_write_data_buckets(
        _: CheckpointContainerId, write_buckets: list[ObjectWriteBucket], non_blocking: bool = True
    ):
        _LOGGER.debug(
            "Executing stage_write_data_buckets with non_blocking=%s (staging from GPU to CPU)...", non_blocking
        )
        results: list[ObjectWriteBucket] = []

        for bucket in write_buckets:
            tensor_data = [
                (item, tensor.to(device="cpu", non_blocking=non_blocking)) for item, tensor in bucket.tensor_data
            ]
            # Return new instances to avoid referencing or mutating pre-existing instances,
            # which may cause undesirable behavior.
            results.append(dataclasses.replace(bucket, tensor_data=tensor_data))

        if non_blocking and torch.cuda.is_available():
            _LOGGER.debug("Synchronizing after staging...")
            torch.cuda.synchronize()

        return results

    @log_execution_time(logger=_LOGGER, name="write_staged_write_buckets", level=logging.INFO)
    def write_staged_data_buckets(
        self,
        checkpoint_id: CheckpointContainerId,
        staged_write_buckets: list[ObjectWriteBucket],
        replicate_after_write: bool,
    ) -> TorchFuture[list[WriteResult]]:
        start_time = time.perf_counter()
        write_results = self._checkpoint_saver.write_data(
            checkpoint_id,
            write_buckets=staged_write_buckets,
            thread_count=self._thread_count,
            replicate_after_write=replicate_after_write,
        )
        end_time = time.perf_counter()
        duration = end_time - start_time
        total_bytes_written = sum(res.size_in_bytes for res in write_results)

        if duration > 0:
            throughput = (total_bytes_written / 1e9) / duration if duration > 0 else 0
            _LOGGER.info(
                "Written %d bytes in %.4f s (%.2f GB/s) from %d buckets",
                total_bytes_written,
                duration,
                throughput,
                len(staged_write_buckets),
            )

        self._write_results_per_checkpoint_id[checkpoint_id] = write_results

        # Signal that the write for this checkpoint_id is complete.
        _LOGGER.debug("Setting write event for checkpoint_id '%s'...", checkpoint_id)
        self._write_events_per_checkpoint_id[checkpoint_id].set()

        write_results_future = TorchFuture()
        write_results_future.set_result(write_results)
        return write_results_future

    @override
    @log_execution_time(logger=_LOGGER, name="write_data", level=logging.INFO)
    def write_data(self, plan: SavePlan, planner: SavePlanner) -> TorchFuture[list[WriteResult]]:
        self._check_checkpoint_id()
        if not isinstance(plan.storage_data, _StorageDataContext) or not plan.storage_data.prefix:
            raise ValueError(
                "SavePlan.storage_data is not a valid _StorageDataContext or prefix is empty. "
                "This is likely because prepare_global_plan was not called on this SavePlan."
            )

        write_data_buckets = self.prepare_write_data_buckets(self.current_checkpoint_id, plan, planner)

        return self.write_staged_data_buckets(
            self.current_checkpoint_id, write_data_buckets, replicate_after_write=True
        )

    def replicate_written_objects(self, object_ids: set[CheckpointObjectId]) -> list[concurrent.futures.Future]:
        """Replicates all the objects written for the given `checkpoint_id` asynchronously.

        Should only be called AFTER `write_staged_data_buckets` or `write_data` has completed, and BEFORE
        `finalize_checkpoint`.

        Args:
            object_ids: The set of object IDs to replicate.

        Returns:
            A list of futures representing the replication tasks.
        """
        _LOGGER.debug("Replicating %d objects: %s", len(object_ids), object_ids)
        futures = []
        for full_object_id in object_ids:
            futures.extend(self._checkpoint_saver.async_replicate_object(full_object_id))
        return futures

    @log_execution_time(logger=_LOGGER, name="get_write_results")
    def get_write_results(
        self, checkpoint_id: CheckpointContainerId, wait_timeout_sec: float = 15.0
    ) -> list[WriteResult]:
        """Retrieves the write results for a specific checkpoint ID.

        This method should be invoked only after `prepare_write_data_buckets()` and either
        `write_staged_data_buckets()` or `write_data()` have completed. It waits for an
        internal event to be set, indicating that the write operation for the given
        `checkpoint_id` has finished and its results are available. A generous default timeout
        is provided to account for potential inter-process communication delays, but
        a long wait time might indicate an underlying issue.

        Args:
            checkpoint_id: The unique identifier for the checkpoint container.
            wait_timeout_sec: The maximum time (in seconds) to wait for the write
                operation to complete and its results to become available. Defaults to 15.0 seconds.

        Returns:
            A list of `WriteResult` objects containing the outcomes of the write
            operations for the specified checkpoint.

        Raises:
            KeyError: If `checkpoint_id` is not found in the internal tracking dictionaries,
                meaning no write operation was initiated for it.
            RuntimeError: If the write event for the `checkpoint_id` is not set within
                the `wait_timeout_sec`, indicating that the write operation did not
                complete successfully or in a timely manner.
        """
        _LOGGER.debug("Waiting for event for checkpoint_id '%s' for up to %d sec...", checkpoint_id, wait_timeout_sec)
        event_set = self._write_events_per_checkpoint_id[checkpoint_id].wait(timeout=wait_timeout_sec)
        if not event_set:
            msg = (
                "Event was never set for checkpoint_id '%s', meaning we cannot confirm that the write "
                "has completed, and its results are available." % checkpoint_id
            )
            _LOGGER.error(msg)
            raise RuntimeError(msg)

        return self._write_results_per_checkpoint_id.get(checkpoint_id)

    @override
    @log_execution_time(logger=_LOGGER, name="finish")
    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        self._check_checkpoint_id()
        self.finish_checkpoint(self.current_checkpoint_id, metadata, results)

    @log_execution_time(logger=_LOGGER, name="finish_checkpoint", level=logging.INFO)
    def finish_checkpoint(
        self, checkpoint_id: CheckpointContainerId, metadata: Metadata, results: list[list[WriteResult]]
    ) -> None:
        """This implements the core of finish, but it accepts an explicit checkpoint_id to avoid race conditions with
        overlapping checkpoint saves.

        It is recommended to use this directly when possible instead of `finish()`.

        Args:
            checkpoint_id: The ID of the checkpoint container.
            metadata: The metadata associated with the checkpoint.
            results: A list of lists containing write results from all ranks.
        """
        storage_data: dict[MetadataIndex, _StorageInfo] = dict()
        _LOGGER.debug(
            "finish: got %d lists as write results, updating storage_data with each element in them...", len(results)
        )
        for idx, wr_list in enumerate(results):
            if wr_list is None:
                msg = (
                    "finish: write results[%d] is None! This suggests an error in producing or gathering "
                    "the write_results list from that local rank (%d)." % (idx, idx)
                )
                _LOGGER.error(msg)
                raise RuntimeError(msg)
            storage_data.update({wr.index: wr.storage_data for wr in wr_list})

        metadata.storage_data = storage_data
        metadata.storage_meta = self.storage_meta()

        self._checkpoint_saver.write_metadata(checkpoint_id, metadata)
        _LOGGER.debug(
            "finish: write_metadata complete, now removing write_results for '%s' from dict",
            checkpoint_id,
        )
        self._write_results_per_checkpoint_id.pop(checkpoint_id, None)

    @classmethod
    @override
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        try:
            # Wrapping the ID as a CheckpointContainerId will validate it, raising an error if invalid.
            CheckpointContainerId(checkpoint_id)
            return True
        except Exception as e:
            _LOGGER.warning("Unable to validate checkpoint_id: %s, %s", checkpoint_id, e)
            return False
