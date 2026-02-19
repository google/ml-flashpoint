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
import threading
from typing import Any, Optional, Union, cast

import torch
from lightning.fabric.utilities.types import _PATH
from megatron.core import dist_checkpointing as mcore_dist_checkpointing
from megatron.core.dist_checkpointing.strategies.async_utils import (
    AsyncCallsQueue,
)
from megatron.core.dist_checkpointing.strategies.async_utils import (
    AsyncRequest as MegatronAsyncRequest,
)
from megatron.core.dist_checkpointing.strategies.common import TorchCommonLoadStrategy
from nemo.lightning.io.pl import MegatronCheckpointIO, TrainerContext, _fix_tensors_device
from nemo.lightning.pytorch.trainer import Trainer
from nemo.utils.callbacks.dist_ckpt_io import AsyncCompatibleCheckpointIO, AsyncFinalizableCheckpointIO
from typing_extensions import override

from ml_flashpoint.adapter.megatron.load_strategies import (
    MLFlashpointMegatronLoadStrategy,
)
from ml_flashpoint.adapter.megatron.save_strategies import (
    MLFlashpointMegatronAsyncSaveStrategy,
)
from ml_flashpoint.adapter.megatron.save_utils import save_local_aware_megatron_checkpoint
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import (
    CheckpointObjectManager,
)
from ml_flashpoint.core.buffer_pool import BufferPool
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId
from ml_flashpoint.core.checkpoint_saver import DEFAULT_INITIAL_BUFFER_SIZE_BYTES
from ml_flashpoint.core.mlf_logging import get_logger
from ml_flashpoint.core.utils import log_execution_time

_LOGGER = get_logger(__name__)


def _is_ml_flashpoint_checkpoint(flashpoint_base_dir: Union[str, CheckpointContainerId], path: _PATH) -> bool:
    """Checks if the given path corresponds to an ML Flashpoint checkpoint.

    Args:
        path (_PATH): The path to check.

    Returns:
        bool: True if the path is an ML Flashpoint checkpoint, False otherwise.
    """
    return str(path).startswith(str(flashpoint_base_dir))


def _init_pool_in_worker(path: str, r: int, num_buffers: int, size: int):
    """Initializes the BufferPool in the worker process."""
    _LOGGER.info(
        "Initializing BufferPool with directory: %s, rank: %d, num_buffers: %d, size: %d", path, r, num_buffers, size
    )
    BufferPool(path, rank=r, num_buffers=num_buffers, buffer_size=size)


def _teardown_pool_in_worker():
    """Teardown the BufferPool in the worker process."""
    try:
        BufferPool.get_instance().teardown()
    except Exception as e:
        _LOGGER.debug("Failed to teardown BufferPool, it might have been closed already: %s", e)


class MLFlashpointCheckpointIO(AsyncCompatibleCheckpointIO):
    """
    This is a wrapper, hierarchical `CheckpointIO` implementation that supports both:
     * ML Flashpoint checkpointing for faster, memory-based crash recovery,
     * and any traditional strategy as an alternative, typically used for long-term storage (and recovery from it).

    Today, only `MegatronCheckpointIO` is supported as an alternative. In the future, this may be expanded.

    The saving path relies on the `storage_options` provided, where a specific key/value signifies ML Flashpoint saving.

    The loading path prioritizes ML Flashpoint, falling back to the alternative strategy if ML Flashpoint cannot
    recover from its checkpoints.
    """

    def __init__(
        self,
        flashpoint_base_path: Union[str, CheckpointContainerId],
        alt_checkpoint_io: MegatronCheckpointIO,
        chkpt_obj_manager: CheckpointObjectManager,
        save_strategy: MLFlashpointMegatronAsyncSaveStrategy,
        load_strategy: MLFlashpointMegatronLoadStrategy,
        trainer: Trainer,
        async_save: bool = True,
        always_save_context: bool = False,
    ):
        """Initializes `MLFlashpointCheckpointIO` so that it uses the given base path as a container ID to store and
        find child checkpoint versions (containers), and the given fallback implementation to delegate to when
        necessary.

        Args:
            flashpoint_base_path: The base container ID (path) to store and find checkpoint version containers
                (subdirectories per checkpoint instance).
            alt_checkpoint_io: The alternative `MegatronCheckpointIO` implementation to use when necessary.
            chkpt_obj_manager: The checkpoint object manager to use for managing checkpoint objects.
            save_strategy: The save strategy to use for ML Flashpoint checkpoints.
            load_strategy: The load strategy to use for ML Flashpoint checkpoints.
            trainer: The trainer to use for training.
            async_save: Whether to save checkpoints asynchronously. Defaults to `True`.
            always_save_context: Whether to always save the context. Defaults to `False`.
        """
        self.flashpoint_base_dir = CheckpointContainerId(flashpoint_base_path)
        self.fallback_checkpoint_io = alt_checkpoint_io
        self.chkpt_obj_manager = chkpt_obj_manager
        self.save_strategy = save_strategy
        self.load_strategy = load_strategy
        self.trainer = trainer
        self.async_save = async_save
        self.always_save_context = always_save_context

    @override
    @log_execution_time(logger=_LOGGER, name="MLFlashpointCheckpointIO.save_checkpoint", level=logging.INFO)
    def save_checkpoint(
        self,
        checkpoint: dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> Optional[MegatronAsyncRequest]:
        """Saves a checkpoint to either ML Flashpoint or the alternative storage.

        Args:
            checkpoint: The checkpoint to save.
            path: The path to save the checkpoint to.
            storage_options: Optional storage options.

        Returns:
            An `Optional[MegatronAsyncRequest]` if `async_save` is `True` and the save is successful,
                otherwise `None`.
        """
        if not _is_ml_flashpoint_checkpoint(self.flashpoint_base_dir, path):
            _LOGGER.info("Fallback to alternative checkpoint io.")
            return self.fallback_checkpoint_io.save_checkpoint(checkpoint, path)
        _LOGGER.info("Use ML Flashpoint checkpoint io. Async_save: '%s'", self.async_save)

        # Use the helper for local-aware megatron save
        optional_async_request = save_local_aware_megatron_checkpoint(
            checkpoint=checkpoint,
            checkpoint_dir=path,
            save_strategy=self.save_strategy,
            async_save=self.async_save,
        )

        # Handle optional context save (only if enabled)
        if self.always_save_context:
            _LOGGER.debug("Saving context...")
            self._save_context(path)

        return optional_async_request

    @log_execution_time(logger=_LOGGER, name="MLFlashpointCheckpointIO._save_context", level=logging.INFO)
    def _save_context(self, path: _PATH) -> Optional[threading.Thread]:
        """Saves the training context to the checkpoint directory.

        Args:
            path: The path to the checkpoint directory.
        """
        context_path = os.path.join(path, "context")
        # Only global rank 0 generates the context
        context_data = {}
        if torch.distributed.get_rank() == 0:
            _LOGGER.info("Saving training context...")
            try:
                # Generate context on rank 0
                TrainerContext.from_trainer(self.trainer).io_dump(context_path, yaml_attrs=["model"])
                # Read generated files for broadcasting
                for root, _, files in os.walk(context_path):
                    for file in files:
                        abs_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_path, context_path)
                        with open(abs_path, "rb") as f:
                            context_data[rel_path] = f.read()
            except Exception as e:
                _LOGGER.warning(f"Failed to dump/read context for broadcast: {e}")

        # Serialize and broadcast the context data to all ranks
        # We use a list to wrap the dictionary as broadcast_object_list expects serializable objects
        object_list = [context_data]
        torch.distributed.broadcast_object_list(object_list, src=0)
        context_data = object_list[0]

        # Write context to disk on all other nodes (local rank 0)
        # Global rank 0 already has it on disk
        if torch.distributed.get_node_local_rank() == 0 and torch.distributed.get_rank() != 0:

            def _write_context_propagated(path: str, data: dict):
                try:
                    os.makedirs(path, exist_ok=True)
                    for rel_path, content in data.items():
                        full_path = os.path.join(path, rel_path)
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        with open(full_path, "wb") as f:
                            f.write(content)
                except Exception as e:
                    _LOGGER.warning(f"Failed to write propagated context: {e}")

            thread = threading.Thread(target=_write_context_propagated, args=(context_path, context_data))
            thread.start()
            return thread
        return None

    @override
    @log_execution_time(logger=_LOGGER, name="MLFlashpointCheckpointIO.load_checkpoint", level=logging.INFO)
    def load_checkpoint(
        self, path: _PATH, sharded_state_dict=None, map_location: Optional[Any] = None, **kwargs
    ) -> dict[str, Any]:
        """Loads a checkpoint from either ML Flashpoint or the alternative storage.

        Args:
            path: The path to load the checkpoint from.
            sharded_state_dict: state dict of the existing model populated with ShardedTensors. Used as a mapping
            to determine which parts of global tensors stored in the checkpoint should be loaded.
            map_location: Optional argument to specify how to remap storage locations.
            kwargs: Additional keyword arguments - NeMo provides additional kwargs that will fail without this.
            These are ignored for now in ML Flashpoint's implementation, and passed through to the fallback.

        Returns:
            The loaded checkpoint as a dictionary.

        Raises:
            Exception: If loading from ML Flashpoint fails.
        """
        if not _is_ml_flashpoint_checkpoint(self.flashpoint_base_dir, path):
            _LOGGER.info("No ML Flashpoint checkpoint found, falling back to alternative checkpoint io.")
            return self.fallback_checkpoint_io.load_checkpoint(
                path, sharded_state_dict=sharded_state_dict, map_location=map_location, **kwargs
            )

        try:
            _LOGGER.info("Loading ML Flashpoint checkpoint from '%s'", path)
            # Given the existing load function doesn't do anything rank-specific,
            # it is suitable for us to use directly.
            state_dict = mcore_dist_checkpointing.load(
                sharded_state_dict=sharded_state_dict,
                checkpoint_dir=str(path),
                sharded_strategy=self.load_strategy,
                common_strategy=TorchCommonLoadStrategy(),
            )

            if torch.cuda.is_initialized():
                _fix_tensors_device(state_dict)

            return state_dict
        except Exception as e:
            _LOGGER.exception(
                "Failed to load ML Flashpoint checkpoint. If this problem persists, "
                "consider disabling ML Flashpoint or "
                "deleting the Flashpoint base container across all training nodes. "
                "Re-raising the exception."
            )
            raise e

    @override
    @log_execution_time(logger=_LOGGER, name="MLFlashpointCheckpointIO.remove_checkpoint", level=logging.INFO)
    def remove_checkpoint(self, path: _PATH) -> None:
        """Removes a checkpoint.

        Args:
            path: The path to the checkpoint to remove.
        """
        _LOGGER.info("Attempting to remove checkpoint directory: '%s'", path)
        if _is_ml_flashpoint_checkpoint(self.flashpoint_base_dir, path):
            self.chkpt_obj_manager.delete_container(CheckpointContainerId(path))
        else:
            self.fallback_checkpoint_io.remove_checkpoint(path)


class MLFlashpointAsyncFinalizableCheckpointIO(AsyncFinalizableCheckpointIO):
    """CheckpointIO wrapper for async checkpoint saving and synchronous finalization
    that handles ML Flashpoint and other strategy finalization separately.

    This is needed instead of `AsyncFinalizableCheckpointIO` because ML Flashpoint checkpoints
    complete much quicker, and that implementation finalizes async calls in the order they are scheduled,
    not in the order they terminate. This results in ML Flashpoint async call finalizations queuing up until the last
    regular checkpoint is complete, which can quickly lead to OOM errors as new async calls continue to get
    scheduled, but not finalized, meaning older checkpoints are not removed.

    Lastly, this must be a subtype of `AsyncFinalizableCheckpointIO` because validation checks expect that type.

    From `AsyncFinalizableCheckpointIO`:
    Runs main part of the checkpoint save in a separate process (not thread as the PTL
    AsyncCheckpointIO does). Allows to perform a (synchronous) finalization
    function after all ranks finish checkpoint saving.

    NOTE: for correctness, this plugin must be used together with the
    AsyncFinalizerCallback callback which performs the finalization checks.
    """

    def __init__(self, checkpoint_io: AsyncCompatibleCheckpointIO):
        """Initializes `MLFlashpointAsyncFinalizableCheckpointIO`.

        Args:
            checkpoint_io: The `AsyncCompatibleCheckpointIO` instance to wrap.

        Raises:
            ValueError: If the provided `checkpoint_io` is not a compatible type.
        """
        if not isinstance(checkpoint_io, AsyncCompatibleCheckpointIO) or not isinstance(
            checkpoint_io, MLFlashpointCheckpointIO
        ):
            raise ValueError("Incompatible wrapped checkpoint_io type: %s", type(checkpoint_io))

        super().__init__(checkpoint_io)
        self._mlf_async_calls_queue = AsyncCallsQueue(persistent=True)
        self._alt_async_calls_queue = AsyncCallsQueue()

        # We want to have enough buffers to cover all threads + some slack.
        # 3x thread count seems reasonable to avoid waiting for buffers.
        # We access the storage writer thread count from the save strategy.
        # Note: accessing private member _storage_writer and _thread_count.
        # This assumes MLFlashpointMegatronAsyncSaveStrategy structure.
        # TODO: Get the thread count from the save strategy in a public way.
        # TODO: Revisit this magic number 3.
        num_buffers = 3 * self.mlf_checkpoint_io.save_strategy._storage_writer._thread_count

        self._pool_init_args = (
            os.path.join(str(self.mlf_checkpoint_io.flashpoint_base_dir), "buffer_pool"),
            self.mlf_checkpoint_io.trainer.global_rank,
            num_buffers,
            DEFAULT_INITIAL_BUFFER_SIZE_BYTES,
        )
        self._pool_initialized = False

    @property
    def mlf_checkpoint_io(self) -> MLFlashpointCheckpointIO:
        """Helper to return the underlying checkpoint_io cast as MLFlashpointCheckpointIO to satisfy type checkers.

        This is enforced in __init__.

        Returns:
            The underlying checkpoint_io cast as MLFlashpointCheckpointIO.
        """
        return cast(MLFlashpointCheckpointIO, self.checkpoint_io)

    @override
    @log_execution_time(
        logger=_LOGGER, name="MLFlashpointAsyncFinalizableCheckpointIO.save_checkpoint", level=logging.INFO
    )
    def save_checkpoint(self, checkpoint: dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        """Executes async request returned from the underlying checkpoint_io asynchronously.

        Requires the underlying checkpoint_io.save_checkpoint to return an AsyncRequest.
        It is then applied with the corresponding AsyncCallsQueue asynchronously.

        Args:
            checkpoint (Dict[str, Any]): checkpoint to save. Passed to underlying
                checkpoint_io without modifications.
            path (_PATH): path to save the checkpoint. Passed to underlying
                checkpoint_io without modifications.
            storage_options (Any, optional): storage control modifiers. This class
                consumed the `finalize_fn` parameter (if any), which is expected to be
                a callback and is appended to async finalization functions.

        Applies underlying checkpoint_io finalize callback first, then the external one (postfix order).
        """
        external_finalize_fn = (storage_options or {}).pop("finalize_fn", None)
        async_request = self.checkpoint_io.save_checkpoint(checkpoint, path, storage_options)
        if external_finalize_fn is not None:
            async_request.add_finalize_fn(external_finalize_fn)
        if _is_ml_flashpoint_checkpoint(self.mlf_checkpoint_io.flashpoint_base_dir, path):
            queue_type = "ml_flashpoint"
            corresponding_async_calls_queue = self._mlf_async_calls_queue
            if not self._pool_initialized:
                _LOGGER.info("Initializing BufferPool on first save...")
                self._mlf_async_calls_queue.schedule_async_request(
                    MegatronAsyncRequest(
                        async_fn=_init_pool_in_worker, async_fn_args=self._pool_init_args, finalize_fns=[]
                    )
                )
                self._pool_initialized = True
        else:
            queue_type = "alternative"
            corresponding_async_calls_queue = self._alt_async_calls_queue
        call_idx = corresponding_async_calls_queue.schedule_async_request(async_request)
        _LOGGER.debug("Scheduled an async call #%d for path '%s' on '%s' queue", call_idx, path, queue_type)

    @override
    @log_execution_time(
        logger=_LOGGER, name="MLFlashpointAsyncFinalizeCheckpointIO.maybe_finalize_save_checkpoint", level=logging.INFO
    )
    def maybe_finalize_save_checkpoint(self, blocking: bool = False) -> bool:
        """Performs checkpoint finalization (if possible).

        Args:
            blocking (bool, optional): if True, waits until all async saves are
                completed. Otherwise, finalizes only those async calls which are
                already done on all ranks. Defaults to False.

        Returns:
            `True` if any checkpoints were finalized, `False` otherwise.
        """
        if (
            self._mlf_async_calls_queue.get_num_unfinalized_calls() == 0
            and self._alt_async_calls_queue.get_num_unfinalized_calls() == 0
        ):
            return False

        with log_execution_time(
            logger=_LOGGER, name="MLFlashpointAsyncFinalizableCheckpointIO.maybe_finalize_save_checkpoint.mlf"
        ):
            mlf_call_idx_finalized = self._mlf_async_calls_queue.maybe_finalize_async_calls(blocking)
            if mlf_call_idx_finalized:
                _LOGGER.debug("Finalized ml_flashpoint async calls: %s", [f"#{idx}" for idx in mlf_call_idx_finalized])

        with log_execution_time(
            logger=_LOGGER, name="MLFlashpointAsyncFinalizableCheckpointIO.maybe_finalize_save_checkpoint.alt"
        ):
            alt_call_idx_finalized = self._alt_async_calls_queue.maybe_finalize_async_calls(blocking)
            if alt_call_idx_finalized:
                _LOGGER.debug("Finalized alternative async calls: %s", [f"#{idx}" for idx in alt_call_idx_finalized])

        return len(mlf_call_idx_finalized) + len(alt_call_idx_finalized) > 0

    @override
    @log_execution_time(logger=_LOGGER, name="MLFlashpointAsyncFinalizeCheckpointIO.teardown")
    def teardown(self) -> None:
        """Warns if there are any pending checkpoint saves."""
        super().teardown()
        if (
            self._mlf_async_calls_queue.get_num_unfinalized_calls()
            + self._alt_async_calls_queue.get_num_unfinalized_calls()
            > 0
        ):
            # Can't do finalization now because some ranks might be lost
            _LOGGER.warning("Some async checkpoint saves might be not finalized properly.")

        # Teardown BufferPool in the worker
        # We schedule a task to teardown.
        if hasattr(self, "_mlf_async_calls_queue") and self._mlf_async_calls_queue:
            try:
                self._mlf_async_calls_queue.schedule_async_request(
                    MegatronAsyncRequest(async_fn=_teardown_pool_in_worker, async_fn_args=(), finalize_fns=[])
                )
            except Exception:
                # Queue might be closed already
                pass

        if hasattr(self._mlf_async_calls_queue, "close"):
            self._mlf_async_calls_queue.close()
            # Monkeypatch persistent caller's close method to prevent double-close error at exit
            # which happens if __del__ is called after process group destruction.
            # We access the caller directly if possible as AsyncCallsQueue might store it as 'persistent_caller'.
            caller = getattr(self._mlf_async_calls_queue, "persistent_caller", None)
            if caller and hasattr(caller, "close"):
                # We already closed the queue (and hopefully the caller), so we prevent future closes.
                # Specifically, PersistentAsyncCaller.__del__ calls close() which calls torch.distributed.get_rank(),
                # causing a crash if the process group is already destroyed.
                caller.close = lambda: None
