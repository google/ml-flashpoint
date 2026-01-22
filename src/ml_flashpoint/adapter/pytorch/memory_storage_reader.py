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
import os
import time
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Union

import torch
from torch.distributed.checkpoint import StorageReader
from torch.distributed.checkpoint import metadata as torchdistmeta
from torch.distributed.checkpoint.filesystem import FileSystem, _StorageInfo
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    ReadItem,
)
from torch.futures import Future
from typing_extensions import override

from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId, CheckpointObjectId
from ml_flashpoint.core.checkpoint_loader import (
    MLFlashpointCheckpointLoader,
)
from ml_flashpoint.core.hfid import generate_hfid
from ml_flashpoint.core.mlf_logging import get_logger
from ml_flashpoint.core.utils import log_execution_time

_LOGGER = get_logger(__name__)


class MemoryStorageReader(StorageReader):
    """
    MemoryStorageReader represents a local, in-memory StorageReader implementation, with replication
    **from** peer node(s) when necessary for missing files.

    Args:
        path (Union[str, os.PathLike]): The base path for checkpoint storage.
        default_storage_reader (Optional[StorageReader]): The base StorageReader implementation to use for some
            operations. If None, defaults to FileSystemReader.
    """

    # Fields declared explicitly here for linting purposes.
    _path: str
    _checkpoint_container_id: CheckpointContainerId
    _load_id: str
    _checkpoint_loader: MLFlashpointCheckpointLoader
    _storage_data: dict[torchdistmeta.MetadataIndex, _StorageInfo] | None

    def __init__(
        self,
        path: Union[str, os.PathLike],
        checkpoint_loader: MLFlashpointCheckpointLoader,
    ):
        """Initializes the MemoryStorageReader.

        Args:
            path: The base path for checkpoint storage.
            checkpoint_loader: The MLFlashpointCheckpointLoader instance to use.
        """
        super().__init__()
        self.reset(path)
        self._checkpoint_loader = checkpoint_loader
        self._storage_data = None

    @override
    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        self._path = checkpoint_id
        self._checkpoint_container_id = CheckpointContainerId(str(checkpoint_id))
        self._load_id = generate_hfid("memreaderload")

    @override
    @log_execution_time(logger=_LOGGER, name="read_metadata", level=logging.INFO)
    def read_metadata(self) -> torchdistmeta.Metadata:
        metadata = self._checkpoint_loader.read_metadata(self._checkpoint_container_id)
        if self._load_id:
            metadata.storage_meta.load_id = self._load_id
        return metadata

    @override
    def set_up_storage_reader(self, metadata: torchdistmeta.Metadata, is_coordinator: bool) -> None:
        self._storage_data = metadata.storage_data
        if self._storage_data is None:
            raise ValueError("metadata.storage_data cannot be None.")

    @override
    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    @override
    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        return plans

    @override
    @log_execution_time(logger=_LOGGER, name="read_data", level=logging.INFO)
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        per_file: Dict[str, List[ReadItem]] = dict()
        for read_item in plan.items:
            item_md = self._storage_data[read_item.storage_index]
            relative_path = item_md.relative_path
            per_file.setdefault(relative_path, []).append(read_item)

        task_futures: List[ConcurrentFuture] = []
        # logger.info(f"Rank {torch.distributed.get_rank()}: read_data for num_files:{len(per_file)}")
        start_time = time.perf_counter()
        if per_file:  # Only create executor if there are files to process
            with ThreadPoolExecutor() as executor:
                for relative_path, reqs_for_file in per_file.items():
                    object_id = CheckpointObjectId(str(Path(self._path) / relative_path))
                    task_future = executor.submit(
                        self._checkpoint_loader.read_data,
                        object_id,
                        reqs_for_file,
                        planner,
                        self._storage_data,
                    )
                    task_futures.append(task_future)

                # Wait for all submitted tasks to complete.
                # .result() will re-raise any exceptions that occurred in the threads.
                for task_future in task_futures:
                    task_future.result()

            end_time = time.perf_counter()
            duration = end_time - start_time
            total_bytes_read = sum(self._storage_data[req.storage_index].length for req in plan.items)
            if duration > 0:
                throughput = (total_bytes_read / 1e9) / duration if duration > 0 else 0
                _LOGGER.info(
                    "Read %d bytes in %.4f s (%.2f GB/s) from %d files",
                    total_bytes_read,
                    duration,
                    throughput,
                    len(per_file),
                )
        else:
            _LOGGER.info("Rank %d: No files to read for plan in '%s'", torch.distributed.get_rank(), self._path)
        # This future won't be used.
        fut: Future = Future()
        fut.set_result(None)
        return fut

    @classmethod
    @override
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)
