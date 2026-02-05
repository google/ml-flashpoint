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
import queue
import threading
from typing import Callable, Optional, Protocol, Union

import torch
from torch.distributed.checkpoint import metadata as torchdistmeta
from torch.distributed.checkpoint.filesystem import _split_by_size_and_type, _StorageInfo
from torch.distributed.checkpoint.planner import WriteItem, WriteItemType
from torch.distributed.checkpoint.storage import WriteResult
from typing_extensions import override

from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.checkpoint_object_manager.object_manager import object_manager_ext
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
        thread_count: int,
        replicate_after_write: bool,
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

    @override
    @log_execution_time(logger=_LOGGER, name="initialize_checkpoint")
    def initialize_checkpoint(self, checkpoint_id: CheckpointContainerId) -> None:
        self._create_dirty_checkpoint_marker(checkpoint_id)
        if self._local_rank_getter() == 0:
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
        thread_count: int = 1,
    ) -> list[WriteResult]:
        thread_count = max(thread_count, 1)
        num_cpus = os.cpu_count() or 1
        num_ranks = max(get_accelerator_count(), 1)
        torch_thread_count = max(1, num_cpus // 2 // num_ranks // thread_count)
        original_num_threads = torch.get_num_threads()
        # Use 50% of available CPU cores for PyTorch intra-op threads and evenly distribute them across ranks.
        # Explicitly set PyTorch intra-op threads to optimize for performance.
        # This also avoids potential runtime errors in tensor.copy_() with concurrent writers
        torch.set_num_threads(torch_thread_count)
        _LOGGER.debug(
            "%s starting multi-threaded write_data. thread_count: %d, original_num_threads: %d, "
            "num_cpus: %d, num_ranks: %d, torch_thread_count: %d",
            self.__class__.__name__,
            thread_count,
            original_num_threads,
            num_cpus,
            num_ranks,
            torch_thread_count,
        )
        try:
            # Queue of ObjectWriteBuckets
            object_items_queue: queue.Queue = queue.Queue()
            for bucket in write_buckets:
                object_items_queue.put(bucket)

            # NOTE: There is support for multiple threads, to simplify modifying that setting, but we typically
            # only use 1 thread.

            results_from_threads: queue.Queue = queue.Queue()  # Queue for tuple[List[WriteResult], Exception]
            threads = []

            # Kick off additional threads to main thread, if any.
            _LOGGER.debug("Spawning %d extra writer threads (in addition to the main thread).", thread_count - 1)
            for i in range(1, thread_count):
                thread = threading.Thread(
                    target=self._write_to_buffer_from_queue_worker,
                    args=(object_items_queue, results_from_threads, replicate_after_write, self._use_optimized_save),
                    name=f"{self.__class__.__name__}-Thread-{i}",
                )
                threads.append(thread)
                thread.start()

            # Main thread execution.
            self._write_to_buffer_from_queue_worker(
                object_items_queue, results_from_threads, replicate_after_write, self._use_optimized_save
            )

            for thread in threads:
                thread.join()

            all_results: list[WriteResult] = []
            exceptions_raised: list[Exception] = []
            # Collect all results, replication metadata, and exceptions
            while not results_from_threads.empty():
                try:
                    results, exception = results_from_threads.get_nowait()
                    if exception:
                        exceptions_raised.append(exception)
                    elif results:
                        all_results.extend(results)
                except queue.Empty:
                    break

            if exceptions_raised:
                _LOGGER.error(
                    "'%s' encountered %d error(s) during multi-threaded write (will propagate the first one):\n%s.",
                    self.__class__.__name__,
                    len(exceptions_raised),
                    exceptions_raised,
                )
                # Propagate the first exception encountered.
                # TODO: propagate some combined exception, then update log msg
                # (for now they are all logged above at least)
                raise exceptions_raised[0]

            return all_results
        finally:
            torch.set_num_threads(original_num_threads)

    @log_execution_time(logger=_LOGGER, name="async_replicate_object")
    def async_replicate_object(self, object_id: CheckpointObjectId) -> list[concurrent.futures.Future]:
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

    @log_execution_time(logger=_LOGGER, name="_write_to_buffer_from_queue_worker")
    def _write_to_buffer_from_queue_worker(
        self,
        object_write_bucket_queue: queue.Queue,
        results_from_threads: queue.Queue,
        replicate_after_write: bool,
        use_optimized_write: bool,
    ):
        """Worker function for writing data from a queue to buffer objects.

        Args:
            object_write_bucket_queue: A queue containing `ObjectWriteBucket` instances to process.
            results_from_threads: A queue to put `(List[WriteResult], Exception)` tuples into.
            replicate_after_write: Whether to trigger async replication of each object after it is written.
            use_optimized_write: Whether to use optimized write.
        """
        while not object_write_bucket_queue.empty():
            try:
                write_bucket_to_process: ObjectWriteBucket = object_write_bucket_queue.get_nowait()  # Non-blocking get
            except queue.Empty:
                break

            full_object_id, object_name, byteio_tuples, tensor_tuples = (
                write_bucket_to_process.object_id,
                write_bucket_to_process.object_name,
                write_bucket_to_process.bytesio_data,
                write_bucket_to_process.tensor_data,
            )

            try:
                thread_bucket_results: list[WriteResult] = []

                # Set overwrite to True, in case we are recovering from a previous checkpoint.
                # Example: we may recover from step 6, while step 7/8 may be there but unfinished or not selected
                # for some reason, so when we come here to write checkpoints for them, we want to overwrite whatever
                # exists instead of failing.
                with self._chkpt_obj_manager.create_buffer(
                    full_object_id,
                    self._initial_buffer_size_bytes,
                    overwrite=True,
                ) as buffer_io_writer:
                    # Set the format signature
                    if use_optimized_write:
                        buffer_io_writer.set_format_signature(CheckpointFormat.MLF_FORMAT)
                    else:
                        buffer_io_writer.set_format_signature(CheckpointFormat.TORCH_SAVE)

                    # First write tensors.
                    for tensor_item, tensor in tensor_tuples:
                        write_start_offset = buffer_io_writer.tell()
                        if use_optimized_write:
                            self._save_tensor_optimized(tensor, buffer_io_writer)
                        else:
                            torch.save(tensor, buffer_io_writer)

                        num_bytes_written = buffer_io_writer.tell() - write_start_offset
                        item_storage_data = _StorageInfo(
                            relative_path=object_name, offset=write_start_offset, length=num_bytes_written
                        )
                        write_result = WriteResult(
                            index=tensor_item.index, size_in_bytes=num_bytes_written, storage_data=item_storage_data
                        )
                        thread_bucket_results.append(write_result)

                    # Then write ByteIO objects.
                    for byteio_item, data in byteio_tuples:
                        data.seek(0)  # to be safe? maybe unnecessary
                        write_start_offset = buffer_io_writer.tell()

                        buffer_io_writer.write(data.read())

                        num_bytes_written = buffer_io_writer.tell() - write_start_offset
                        item_storage_data = _StorageInfo(
                            relative_path=object_name, offset=write_start_offset, length=num_bytes_written
                        )
                        write_result = WriteResult(
                            index=byteio_item.index, size_in_bytes=num_bytes_written, storage_data=item_storage_data
                        )
                        thread_bucket_results.append(write_result)

                results_from_threads.put((thread_bucket_results, None))
                if replicate_after_write:
                    self.async_replicate_object(full_object_id)
            except Exception as e:
                _LOGGER.exception("Error in writer thread for object '%s'", full_object_id)
                results_from_threads.put((None, e))
            finally:
                object_write_bucket_queue.task_done()

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

    def _save_tensor_optimized(self, tensor: torch.Tensor, buffer_io_writer):
        """Saves a tensor to the buffer using a zero-copy approach where possible.

        NOTE: This method saves the tensor's data in a C-contiguous format,
        regardless of its original memory layout (stride).
        The stride information is not saved.

        Format:
        [4 bytes HEADER_LEN] [HEADER_BYTES (JSON)] [RAW_BYTES]

        Args:
            tensor: The tensor to save.
            buffer_io_writer: The BufferIO instance to write to.
        """
        # Metadata
        tensor_header = TensorHeader(dtype=tensor.dtype, shape=tensor.shape)

        # Write Header (Len + JSON)
        header_data = tensor_header.to_bytes()
        buffer_io_writer.write(header_data)

        # Write Data (Zero Copy)
        num_bytes = tensor.numel() * tensor.element_size()

        # Get a writable slice of the underlying C++ buffer
        if num_bytes > 0:
            try:
                dest_mv = buffer_io_writer.next_buffer_slice(num_bytes)
            except AttributeError:
                _LOGGER.exception("BufferIO does not support next_buffer_slice, try to disable use_optimized_save.")
                raise

            # Create a tensor wrapper around the buffer slice
            dest_tensor = torch.frombuffer(dest_mv, dtype=tensor.dtype, count=tensor.numel()).reshape(tensor.shape)

            # Perform the actual copy.
            dest_tensor.copy_(tensor)
