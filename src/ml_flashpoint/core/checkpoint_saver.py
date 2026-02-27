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

import abc
import concurrent.futures
import dataclasses
import io
import logging
import os
import pickle
from typing import Any, Callable, Optional, Protocol, Union

import torch
from torch.distributed.checkpoint import metadata as torchdistmeta
from torch.distributed.checkpoint.filesystem import _split_by_size_and_type, _StorageInfo
from torch.distributed.checkpoint.planner import WriteItem, WriteItemType
from torch.distributed.checkpoint.storage import WriteResult
from typing_extensions import override

from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.checkpoint_object_manager.object_manager import object_manager_ext
from ml_flashpoint.core.async_writer import async_writer_ext
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId, CheckpointObjectId
from ml_flashpoint.core.defaults import DIRTY_MARKER_SUFFIX, CheckpointFormat, default_metadata_object_name
from ml_flashpoint.core.mlf_logging import get_logger
from ml_flashpoint.core.tensor_header import TensorHeader
from ml_flashpoint.core.utils import get_accelerator_count, log_execution_time
from ml_flashpoint.replication.replication_manager import ReplicationManager

DEFAULT_INITIAL_BUFFER_SIZE_BYTES = 16 * 1000 * 1000 * 1000
"""The default initial buffer size in bytes - 16 GB."""

_DEFAULT_OBJ_NAME_SUFFIX = ".distcp"

_LOGGER = get_logger(__name__)


@dataclasses.dataclass
class ObjectWriteBucket:
    """Container for writes to a single object ID (equivalent to a single file)."""

    object_id: CheckpointObjectId
    """The full object ID for the object being written."""
    object_name: str
    """The object name of the object being written."""
    bytesio_data: list[tuple[WriteItem, io.BytesIO]]
    """The list of BytesIO data to write alongside their originating WriteItem, obtained by resolving the WriteItem via
    some WriteItemResolver."""
    tensor_data: list[tuple[WriteItem, torch.Tensor]]
    """The list of tensor data to write alongside their originating WriteItem, obtained by resolving the WriteItem via
    some WriteItemResolver."""


class WriteItemResolver(Protocol):
    """Structural interface for resolving WriteItems, akin to a PyTorch distributed `SavePlanner`.
    See :meth:`SavePlanner.resolve_data`.

    This dedicated Protocol abstracts the `SavePlanner` interface's `resolve_data` API, and is intended to:
    1. be compatible with a `SavePlanner` directly,
    2. while also supporting other mechanisms for accessing data in a state dict.

    Wherever this is expected, typical usage would supply a `SavePlanner` implementation as is, or some custom
    implementation that can translate a `WriteItem` to either a tensor or some binary data.
    """

    @abc.abstractmethod
    def resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
        """
        Transform and prepare ``write_item`` from a ``state_dict`` for storage, ensuring idempotency and thread-safety.

        Lookup the object associated with ``write_item`` in a ``state_dict`` and apply any
        transformation (such as serialization) prior to the storage layer consuming it.

        Called on each rank multiple times, at least once per WriteItem in the final SavePlan.

        This method should be idempotent and thread-safe. StorageWriter implementations
        are free to call it as frequently as they need.

        Any transformation that allocates memory should be lazily done when his method
        is called in order to reduce peak memory required by checkpointing.

        When returning tensors, they can be on any device or format, they can be views too.
        It's the storage layer responsibility to figure out how to save them.

        Args:
            write_item: The `WriteItem` to resolve.

        Returns:
            The resolved data, either as a `torch.Tensor` or `io.BytesIO`.
        """
        pass


class MLFlashpointCheckpointSaver(abc.ABC):
    """
    This is the main interface for saving checkpoints, providing functionality for the different
    stages in the save flow, which can be adapted into various framework strategies.

    The required order of operations is either:

    ## A. Megatron approach
    [Blocking / synchronous steps]
    1. initialize_checkpoint()
    2. prepare_write_data()
    3. stage_data()
    [Non-blocking / asynchronous steps, which can be blocked on for fully synchronous saves]
    4. write_data()
    5. async_replicate_object() for each written object, if `replicate_after_write=False` for write_data()
    5. write_metadata()
    6. finalize_checkpoint() - this also invokes remove_older_checkpoints() after a barrier

    ## B. PyTorch DCP approach
    [Blocking / synchronous steps]
    1. initialize_checkpoint()
    2. stage_data()
    [Non-blocking / asynchronous steps, which can be blocked on for fully synchronous saves]
    3. write_data()
    4. write_metadata()
    5. finalize_checkpoint() - this also invokes remove_older_checkpoints() after a barrier

    See `DefaultMLFlashpointCheckpointSaver` as the canonical implementation, and the `ml_flashpoint.adapter` package
    for out-of-the-box framework integrations.

    TODO: Think about serialization versioning, checkpoint cleanup (this interface or another?)
    """

    @abc.abstractmethod
    def initialize_checkpoint(self, checkpoint_id: CheckpointContainerId) -> None:
        """Does any preparation for the given checkpoint_id. This typically creates some marker to signify
        the checkpoint is "dirty" or in-progress, and cannot be recovered from. This must be invoked before
        any other operation is done, including creating or registering checkpoint_id, to ensure correctness.

        The converse is finalize_checkpoint().

        Args:
            checkpoint_id: The CheckpointContainerId to prepare for and mark as dirty.
        """
        pass

    @abc.abstractmethod
    def stage_data(self, checkpoint_id: CheckpointContainerId, state_dict: dict, non_blocking: bool = True) -> dict:
        """Ensures that state_dict tensors are in CPU memory.

        This is a blocking API - it returns when staging is complete, even if non_blocking=True.

        Preconditions:
            1. `state_dict` must already be flattened. This requirement may be relaxed in the future.

        Args:
            checkpoint_id: The checkpoint container ID (analogous to a checkpoint directory for this particular
                version).
            state_dict: The PyTorch-distributed state dictionary whose tensors to stage to CPU. Must be
                in a PyTorch-distributed compatible format, and _already flattened_!
            non_blocking (optional): Whether to copy to CPU in a non-blocking manner. Defaults to False.
                If True, will synchronize the copy operations at the end.

        Returns:
            The state_dict after staging. This may be the same state_dict, or a clone, so it should always be used
            going forward after invoking this method for correctness (in case it is a clone).
        """
        pass

    @abc.abstractmethod
    def prepare_write_data(
        self,
        checkpoint_id: CheckpointContainerId,
        write_items: list[WriteItem],
        write_item_resolver: WriteItemResolver,
        object_name_prefix: str,
        bucket_count: int,
    ) -> list[ObjectWriteBucket]:
        """Prepares data for writing by resolving WriteItems and grouping them into ObjectWriteBuckets.

        This method takes a list of `WriteItem` objects and uses the provided `write_item_resolver`
        to transform them into actual data (tensors or bytes). It then groups these resolved
        data items into `ObjectWriteBucket`s, each representing a single object (file) to be written.
        Each bucket is assigned a unique `CheckpointObjectId` and `storage_key`.

        This method can be called before `stage_data()`.

        Args:
            checkpoint_id: Unique hierarchical ID representing this checkpoint container.
                This typically follows a directory path structure.
            write_items: A list of `WriteItem` objects representing the data to be prepared (and eventually written).
            write_item_resolver: A resolver object that can transform a `WriteItem` into
                the actual data (e.g., torch.Tensor or io.BytesIO) to be saved.
            object_name_prefix: A prefix to use for the names of the objects created
                in the checkpoint storage.
            bucket_count: The number of buckets to create. Should be equal to the number of writer threads desired.

        Returns:
            A list of `ObjectWriteBucket`s, each containing resolved data ready for writing.
        """
        pass

    @abc.abstractmethod
    def write_data(
        self,
        checkpoint_id: CheckpointContainerId,
        write_buckets: list[ObjectWriteBucket],
        replicate_after_write: bool,
        thread_count: int,
    ) -> list[WriteResult]:
        """Performs the core write logic for the given write items and checkpoint_id.

        NOTE: This API is currently _synchronous_.
        Thus, for asynchronous writes, this must be executed in an async process or thread.

        This method is expected to be called AFTER `stage_data()`. It takes a list of
        `ObjectWriteBucket` objects that should already be staged, and writes their data to the checkpoint storage.

        Args:
            checkpoint_id: Unique hierarchical ID representing this checkpoint container.
                This typically follows a directory path structure.
            write_buckets: A list of `ObjectWriteBucket` objects, each containing resolved data ready for writing.
            thread_count: The number of threads to use for writing data.
            replicate_after_write: Whether to trigger async replication of each object after it is written.

        Returns:
            The list of WriteResults from the write operations.
        """
        pass

    @abc.abstractmethod
    def write_data_async(
        self,
        checkpoint_id: CheckpointContainerId,
        write_buckets: list[ObjectWriteBucket],
        replicate_after_write: bool,
        thread_count: int,
    ) -> concurrent.futures.Future:
        """Asynchronously writes the data to storage based on the provided buckets.

        This method leverages C++ implementation to perform the actual data copying
        without holding the Python GIL, allowing for true parallelism.

        Args:
            checkpoint_id: Unique hierarchical ID representing this checkpoint container.
            write_buckets: A list of `ObjectWriteBucket` objects, each containing resolved data ready for writing.
            replicate_after_write: Whether to trigger async replication of each object after it is written.
            thread_count: The number of threads to use for writing data.

        Returns:
            A concurrent.futures.Future resolving to a list of `WriteResult` objects.
        """
        pass

    def async_replicate_object(self, object_id: CheckpointObjectId) -> list[concurrent.futures.Future]:
        """Triggers asynchronous replication of the given object_id.

        Args:
            object_id: The ID of the object to replicate.

        Returns:
            A list of futures representing the replication tasks.
        """
        pass

    @abc.abstractmethod
    def write_metadata(
        self,
        checkpoint_id: CheckpointContainerId,
        metadata: torchdistmeta.Metadata,
        md_object_name: str = default_metadata_object_name(),
    ):
        """Writes the given metadata for the given checkpoint_id (using it as a directory path) if the current
        local rank is 0. Thus, this writes the metadata file once on each node.
        The CheckpointObjectId for the written metadata is formed as
        `CheckpointObjectId(f"{checkpoint_id}/{md_object_name}")`.

        The metadata write is atomic, so the metadata object with the identifier mentioned above will either be
        available in a complete state, or it will not be available.

        Args:
            checkpoint_id: Unique hierarchical ID representing this checkpoint container, analogous to a
                checkpoint directory.
            metadata: The Metadata object to serialize.
            md_object_name: The object name for the written metadata, used to construct the CheckpointObjectId.
                Optional, defaults to default_metadata_object_name().
        """
        pass

    @abc.abstractmethod
    def finalize_checkpoint(self, checkpoint_id: CheckpointContainerId) -> Optional[object_manager_ext.BasicFutureVoid]:
        """Finalize the checkpoint for checkpoint_id, indicating it is complete and safe to recover from.
        This specifically does the following:
          1. Cleans up the unfinished marker created by initialize_checkpoint().
          2. Waits on a barrier across all ranks, to ensure all ranks have completed checkpointing
          and marked completion.
          3. If the local rank is 0, removes older checkpoints asynchronously.

        Args:
            checkpoint_id: The CheckpointContainerId to mark as finalized.

        Returns:
            A future that completes when deletion of older checkpoints is done, or None if no deletion was started.
        """
        pass


class DefaultMLFlashpointCheckpointSaver(MLFlashpointCheckpointSaver):
    def __init__(
        self,
        global_rank_getter: Callable[[], int],
        local_rank_getter: Callable[[], int],
        global_barrier_func: Callable[[], None],
        ckpt_obj_manager: CheckpointObjectManager,
        replication_manager: ReplicationManager,
        initial_buffer_size_bytes: int = DEFAULT_INITIAL_BUFFER_SIZE_BYTES,
        use_optimized_save: bool = True,
    ):
        """Initializes the DefaultMLFlashpointCheckpointSaver.

        Args:
            global_rank_getter: A callable that returns the global rank.
            local_rank_getter: A callable that returns the local rank.
            global_barrier_func: A callable that performs a global barrier synchronization.
            ckpt_obj_manager: The checkpoint object manager to use for
                writing data.
            replication_manager: The ReplicationManager singleton used for replicating data
                across nodes.
            initial_buffer_size_bytes: The initial buffer size in bytes to use
                for writing data.
            use_optimized_save: Whether to use the optimized zero-copy tensor saving.
                Defaults to True.
        """
        self._global_rank_getter = global_rank_getter
        self._local_rank_getter = local_rank_getter
        self._barrier_func = global_barrier_func
        self._chkpt_obj_manager = ckpt_obj_manager
        self._replication_manager = replication_manager
        self._initial_buffer_size_bytes = initial_buffer_size_bytes
        self._use_optimized_save = use_optimized_save

    def __getstate__(self):
        """Custom pickling to exclude _replication_manager."""
        state = self.__dict__.copy()
        # Exclude _replication_manager from the pickled state as it is not needed in workers
        # and may be unpickleable or expensive to transfer.
        if "_replication_manager" in state:
            del state["_replication_manager"]
        return state

    def __setstate__(self, state):
        """Custom unpickling to restore state and set _replication_manager to None."""
        self.__dict__.update(state)
        # Restore _replication_manager as None in the worker process
        self._replication_manager = None

    @override
    @log_execution_time(logger=_LOGGER, name="initialize_checkpoint")
    def initialize_checkpoint(self, checkpoint_id: CheckpointContainerId) -> None:
        self._create_dirty_checkpoint_marker(checkpoint_id)
        os.makedirs(checkpoint_id.data, exist_ok=True)
        _LOGGER.info("Created checkpoint directory: '%s'", checkpoint_id.data)

    @override
    @log_execution_time(logger=_LOGGER, name="stage_data", level=logging.INFO)
    def stage_data(self, checkpoint_id: CheckpointContainerId, state_dict: dict, non_blocking: bool = True) -> dict:
        staged_state_dict = {}

        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                staged_state_dict[k] = v.to(device="cpu", non_blocking=non_blocking)
            else:
                staged_state_dict[k] = v

        if non_blocking and torch.cuda.is_available():
            # Guard the synchronization to avoid the cuda dependency and extra cost when not needed (e.g. in tests).
            torch.cuda.synchronize()

        return staged_state_dict

    @override
    @log_execution_time(logger=_LOGGER, name="prepare_write_data")
    def prepare_write_data(
        self,
        checkpoint_id: CheckpointContainerId,
        write_items: list[WriteItem],
        write_item_resolver: WriteItemResolver,
        object_name_prefix: str,
        bucket_count: int,
    ) -> list[ObjectWriteBucket]:
        bucket_count = max(bucket_count, 1)
        _LOGGER.debug(
            "%s prepare_write_data with prefix: '%s', thread_count: %d",
            self.__class__.__name__,
            object_name_prefix,
            bucket_count,
        )

        obj_count = 0

        def _gen_file_info() -> tuple[CheckpointObjectId, str]:
            nonlocal obj_count
            # Use the same naming convention as FileSystemWriter
            object_name = f"{object_name_prefix}_{obj_count}_src{self._global_rank_getter()}{_DEFAULT_OBJ_NAME_SUFFIX}"
            _full_object_id = CheckpointObjectId.from_container(checkpoint_id, object_name)
            obj_count += 1
            return _full_object_id, object_name

        def _clone_if_needed(tensor: torch.Tensor):
            """For some reason, this is needed in case we do non-blocking copies from GPU to CPU,
            to avoid CUDA kernel errors."""
            if tensor.device.type != "cpu":
                # Only CPU tensors need to be cloned.
                return tensor
            is_view = tensor.untyped_storage().size() != tensor.numel() * tensor.itemsize
            ret_tensor = tensor.clone() if is_view else tensor
            return ret_tensor.contiguous()

        # Queue of (full_object_id, storage_key, write_items_list)
        write_buckets: list[ObjectWriteBucket] = []

        # NOTE: There is support for multiple threads, to simplify modifying that setting, but we typically
        # only use 1 thread.

        # Group items into buckets, one bucket per file, up to thread_count files
        buckets = _split_by_size_and_type(bucket_count, write_items)
        for bucket in buckets:
            if not bucket:
                continue
            bytes_data = [
                (item, write_item_resolver.resolve_data(item)) for item in bucket if item.type == WriteItemType.BYTE_IO
            ]
            tensor_data = [
                (item, _clone_if_needed(write_item_resolver.resolve_data(item).detach()))
                for item in bucket
                if item.type != WriteItemType.BYTE_IO
            ]
            if len(bytes_data) > 0 or len(tensor_data) > 0:
                # object_name (relative path) used as storage key
                full_object_id, storage_key = _gen_file_info()
                write_buckets.append(
                    ObjectWriteBucket(
                        object_id=full_object_id,
                        object_name=storage_key,
                        bytesio_data=bytes_data,
                        tensor_data=tensor_data,
                    )
                )

        return write_buckets

    @override
    @log_execution_time(logger=_LOGGER, name="write_data", level=logging.INFO)
    def write_data(
        self,
        checkpoint_id: CheckpointContainerId,
        write_buckets: list[ObjectWriteBucket],
        replicate_after_write: bool,
        thread_count: int,
    ) -> list[WriteResult]:
        # Sync wrapper around async write for backward compatibility.
        future = self.write_data_async(checkpoint_id, write_buckets, replicate_after_write)
        return future.result()

    @override
    @log_execution_time(logger=_LOGGER, name="write_data_async", level=logging.INFO)
    def write_data_async(
        self,
        checkpoint_id: CheckpointContainerId,
        write_buckets: list[ObjectWriteBucket],
        replicate_after_write: bool,
        thread_count: int,
    ) -> concurrent.futures.Future:
        thread_count = max(thread_count, 1)
        num_cpus = os.cpu_count() or 1
        num_ranks = max(get_accelerator_count(), 1)
        # Use 50% of available CPU cores for PyTorch intra-op threads and evenly distribute them across ranks.
        torch_thread_count = max(1, num_cpus // 2 // num_ranks // thread_count)
        original_num_threads = torch.get_num_threads()
        # Explicitly set PyTorch intra-op threads to optimize for performance.
        # This also avoids potential runtime errors in tensor.copy_() with concurrent writers
        torch.set_num_threads(torch_thread_count)
        _LOGGER.debug(
            "%s starting async write_data via C++ extension. Bucket count: %d",
            self.__class__.__name__,
            len(write_buckets),
        )

        # 1. Map MetadataIndex to unique IDs to pass to C++ safely.
        index_map: dict[int, Any] = {}
        next_id = 0

        def _get_id(index: Any) -> int:
            nonlocal next_id
            id_val = next_id
            index_map[id_val] = index
            next_id += 1
            return id_val

        # 2. Convert to C++ buckets.
        cpp_buckets = []
        for bucket in write_buckets:
            tensor_items = []
            for item, tensor in bucket.tensor_data:
                tensor_header = TensorHeader(dtype=tensor.dtype, shape=tensor.shape)
                header_bytes = tensor_header.to_bytes()
                tensor_items.append(
                    async_writer_ext.CppTensorWriteItem(
                        _get_id(item.index),
                        header_bytes,
                        tensor,
                    )
                )

            bytes_items = []
            for item, data in bucket.bytesio_data:
                data.seek(0)
                bytes_items.append(async_writer_ext.CppBytesWriteItem(_get_id(item.index), data.read()))

            cpp_buckets.append(
                async_writer_ext.CppObjectWriteBucket(
                    str(bucket.object_id), bucket.object_name, tensor_items, bytes_items
                )
            )

        # 3. Call C++ extension.
        format_signature = (
            CheckpointFormat.MLF_FORMAT if self._use_optimized_save else CheckpointFormat.TORCH_SAVE
        )

        cpp_future = async_writer_ext.write_buckets_async(
            cpp_buckets, self._initial_buffer_size_bytes, format_signature
        )

        # 4. Wrap the C++ future to map results back to WriteResult.
        python_future = concurrent.futures.Future()

        def _on_future_complete(f: concurrent.futures.Future):
            try:
                cpp_results: list[async_writer_ext.CppWriteResult] = f.result()
                write_results = []
                for res in cpp_results:
                    item_storage_data = _StorageInfo(
                        relative_path=res.relative_path, offset=res.offset, length=res.length
                    )
                    write_results.append(
                        WriteResult(
                            index=index_map[res.index_id],
                            size_in_bytes=res.size_in_bytes,
                            storage_data=item_storage_data,
                        )
                    )

                if replicate_after_write:
                    for bucket in write_buckets:
                        self.async_replicate_object(bucket.object_id)

                python_future.set_result(write_results)
            except Exception as e:
                _LOGGER.exception("Error in C++ writer")
                python_future.set_exception(e)

        cpp_future.add_done_callback(_on_future_complete)
        return python_future

    @log_execution_time(logger=_LOGGER, name="async_replicate_object")
    def async_replicate_object(self, object_id: CheckpointObjectId) -> list[concurrent.futures.Future]:
        if self._replication_manager is None:
            # This can happen in worker processes where we don't pickle the manager.
            # If this is called, it means replicate_after_write=True was passed erroneously or
            # the strategy is trying to replicate in a worker where it shouldn't.
            raise RuntimeError("ReplicationManager is not available (None). Cannot replicate object.")
        object_buffer_io = self._chkpt_obj_manager.get_buffer(object_id)
        return self._replication_manager.async_replicate(object_buffer_io)

    @override
    @log_execution_time(logger=_LOGGER, name="write_metadata")
    def write_metadata(
        self,
        checkpoint_id: CheckpointContainerId,
        metadata: torchdistmeta.Metadata,
        md_object_name: str = default_metadata_object_name(),
    ):
        _LOGGER.info("Writing metadata for checkpoint ID: '%s', with object name: '%s'", checkpoint_id, md_object_name)
        metadata_path = os.path.join(checkpoint_id.data, md_object_name)
        tmp_metadata_path = metadata_path + ".tmp"

        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        with open(tmp_metadata_path, "wb") as tmp_file:
            pickle.dump(metadata, tmp_file)

        os.rename(tmp_metadata_path, metadata_path)

    @override
    @log_execution_time(logger=_LOGGER, name="finalize_checkpoint")
    def finalize_checkpoint(self, checkpoint_id: CheckpointContainerId) -> Optional[object_manager_ext.BasicFutureVoid]:
        self._remove_dirty_checkpoint_marker(checkpoint_id)
        # synchronize across ranks to guarantee they all completed checkpointing before proceeding
        with log_execution_time(logger=_LOGGER, name="finalize_checkpoint__barrier_func", level=logging.DEBUG):
            self._barrier_func()
        if self._local_rank_getter() == 0:
            return self._remove_older_checkpoints(older_than=checkpoint_id)
        return None

    def _create_dirty_checkpoint_marker(self, checkpoint_id: CheckpointContainerId) -> None:
        """Creates a dirty marker (typically a file) for the given checkpoint_id and the current local rank,
        to indicate that it is incomplete.

        This must be the very first operation done in a checkpoint save flow on each rank, before the checkpoint_id
        itself is even registered, to ensure correctness and avoid scenarios where the checkpoint ID is mistakenly
        considered complete but missing its data.

        Args:
            checkpoint_id: The checkpoint ID that is going to be created.
        """
        dirty_marker_file_path = self._get_dirty_marker_file_path(checkpoint_id)
        os.makedirs(os.path.dirname(dirty_marker_file_path), exist_ok=True)
        with open(dirty_marker_file_path, "w") as _:
            pass  # empty file

        _LOGGER.info(
            "Created dirty marker file for checkpoint_id '%s' on local rank '%d': '%s'",
            checkpoint_id,
            self._local_rank_getter(),
            dirty_marker_file_path,
        )

    def _remove_dirty_checkpoint_marker(self, checkpoint_id: CheckpointContainerId) -> None:
        """Removes the dirty marker created by create_dirty_checkpoint_marker. Must be called once the checkpoint_id
        checkpoint is fully complete on a node.

        Args:
            checkpoint_id: The checkpoint_id whose dirty marker will be removed.
        """
        dirty_marker_file_path = self._get_dirty_marker_file_path(checkpoint_id)

        if not os.path.exists(dirty_marker_file_path):
            _LOGGER.warning("Dirty marker file path: '%s' does not exist, skipping removal.", dirty_marker_file_path)
            return

        os.remove(dirty_marker_file_path)
        _LOGGER.info(
            "Removed dirty marker file for checkpoint_id '%s' on local rank '%d': '%s'",
            checkpoint_id,
            self._local_rank_getter(),
            dirty_marker_file_path,
        )

    def _get_dirty_marker_file_path(self, checkpoint_id: CheckpointContainerId):
        """Gets the file path for the dirty marker file.

        Args:
            checkpoint_id: The checkpoint container ID.

        Returns:
            The path to the dirty marker file.
        """
        if str(checkpoint_id) == "/":
            # Currently this is not possible because of validations in CheckpointContainerId, but adding check here
            # in case that changes.
            raise ValueError("CheckpointContainerId cannot be the root path '/'")
        checkpoint_id_str = str(checkpoint_id).rstrip("/")
        return f"{checkpoint_id_str}__{self._local_rank_getter()}__{DIRTY_MARKER_SUFFIX}"

    @log_execution_time(logger=_LOGGER, name="_remove_older_checkpoints")
    def _remove_older_checkpoints(
        self, older_than: CheckpointContainerId
    ) -> Optional[object_manager_ext.BasicFutureVoid]:
        """Scans for sibling checkpoint containers to `older_than`, by listing the children of its parent and filtering
        for those that match the expected format as a safety check, and then deletes all those that are considered
        older _async_.

        This should only be invoked once per node (i.e. on one local rank on every node).

        Args:
            older_than: The checkpoint container ID to compare against. Checkpoints
                older than this will be removed.

        Returns:
            A future that completes when deletion is done, or None if no deletion was started.
        """
        parent_dir = os.path.dirname(older_than.data)
        older_than_step = CheckpointContainerId.parse_version_container_step(os.path.basename(older_than.data))
        if older_than_step is None:
            _LOGGER.warning(
                "Could not parse step from 'older_than' checkpoint container: '%s'. "
                + "Skipping removal of older checkpoints.",
                older_than,
            )
            return None

        siblings_to_delete = set()
        for object_name in os.listdir(parent_dir):
            full_path = os.path.join(parent_dir, object_name)
            if os.path.isdir(full_path):
                step = CheckpointContainerId.parse_version_container_step(object_name)
                if step is not None and step < older_than_step:
                    siblings_to_delete.add(full_path)

        return object_manager_ext.delete_directories_async(list(siblings_to_delete))
