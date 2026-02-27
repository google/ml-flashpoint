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

from typing import Union

import torch
from nemo import lightning as nl
from nemo.lightning.io.pl import MegatronCheckpointIO
from nemo.lightning.pytorch import strategies as nl_strategies
from nemo.lightning.pytorch import trainer as nl_trainer
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO

from ml_flashpoint.adapter.megatron.load_strategies import MLFlashpointMegatronLoadStrategy
from ml_flashpoint.adapter.megatron.save_strategies import MLFlashpointMegatronAsyncSaveStrategy
from ml_flashpoint.adapter.nemo.auto_resume import MLFlashpointAutoResume
from ml_flashpoint.adapter.nemo.checkpoint_callback import MLFlashpointCheckpointCallback
from ml_flashpoint.adapter.nemo.checkpoint_io import MLFlashpointAsyncFinalizableCheckpointIO, MLFlashpointCheckpointIO
from ml_flashpoint.adapter.nemo.nemo_checkpoint_loader import NeMoMLFlashpointCheckpointLoader
from ml_flashpoint.adapter.pytorch.memory_storage_writer import MemoryStorageWriter
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId
from ml_flashpoint.core.checkpoint_loader import DefaultMLFlashpointCheckpointLoader
from ml_flashpoint.core.checkpoint_saver import DEFAULT_INITIAL_BUFFER_SIZE_BYTES, DefaultMLFlashpointCheckpointSaver
from ml_flashpoint.replication.replication_manager import ReplicationManager


def wrap_trainer_and_auto_resume_with_mlflashpoint(
    trainer: nl_trainer.Trainer,
    flashpoint_base_container: Union[str, CheckpointContainerId],
    async_save: bool,
    default_auto_resume: nl.AutoResume = None,
    always_save_context: bool = False,
    write_thread_count: int = 1,
    initial_write_buffer_size_bytes: int = DEFAULT_INITIAL_BUFFER_SIZE_BYTES,
    use_optimized_save: bool = True,
) -> MLFlashpointAutoResume:
    """Wraps the trainer and creates an MLFlashpointAutoResume instance wrapping `default_auto_resume`.

    This function initializes the necessary managers (CheckpointObjectManager,
    ReplicationManager), wraps the trainer's checkpoint IO with ML Flashpoint
    capabilities, and returns a new MLFlashpointAutoResume instance configured
    with the provided parameters, wrapping `default_auto_resume`.

    Args:
        trainer: The NeMo Trainer instance to wrap.
        flashpoint_base_container: The base container ID or path for ML Flashpoint checkpoints.
        async_save: Whether to enable asynchronous saving for checkpoints.
        default_auto_resume: The default AutoResume configuration to inherit from.
        always_save_context: Whether to always save the context. Defaults to `False`.
        write_thread_count: Optional. The number of threads to use for writing checkpoint data. Defaults to 1.
        initial_write_buffer_size_bytes: Optional. The initial size of the buffer for writing checkpoint data
            in bytes. Defaults to `DEFAULT_INITIAL_BUFFER_SIZE_BYTES`.
    Returns:
        An MLFlashpointAutoResume instance configured for ML Flashpoint, wrapping `default_auto_resume`.
    """
    if not flashpoint_base_container:
        raise ValueError("The 'flashpoint_base_container' argument cannot be empty.")

    flashpoint_base_container = CheckpointContainerId(flashpoint_base_container)
    ckpt_obj_manager = CheckpointObjectManager()
    replication_manager = ReplicationManager()
    replication_manager.initialize(checkpoint_object_manager=ckpt_obj_manager)

    ckpt_loader = NeMoMLFlashpointCheckpointLoader(
        checkpoint_object_manager=ckpt_obj_manager,
        replication_manager=replication_manager,
        recover_context=always_save_context,
    )

    wrap_trainer_checkpoint_io_with_mlflashpoint(
        trainer=trainer,
        flashpoint_base_container=flashpoint_base_container,
        ckpt_obj_manager=ckpt_obj_manager,
        replication_manager=replication_manager,
        async_save=async_save,
        checkpoint_loader=ckpt_loader,
        always_save_context=always_save_context,
        write_thread_count=write_thread_count,
        initial_write_buffer_size_bytes=initial_write_buffer_size_bytes,
        use_optimized_save=use_optimized_save,
    )

    default_auto_resume_args = vars(default_auto_resume) if default_auto_resume else {}
    mlf_auto_resume = MLFlashpointAutoResume(
        checkpoint_base_container=flashpoint_base_container, checkpoint_loader=ckpt_loader, **default_auto_resume_args
    )

    return mlf_auto_resume


def wrap_trainer_checkpoint_io_with_mlflashpoint(
    trainer: nl_trainer.Trainer,
    flashpoint_base_container: Union[str, CheckpointContainerId],
    ckpt_obj_manager: CheckpointObjectManager,
    replication_manager: ReplicationManager,
    async_save: bool,
    checkpoint_loader: DefaultMLFlashpointCheckpointLoader,
    always_save_context: bool = False,
    write_thread_count: int = 1,
    initial_write_buffer_size_bytes: int = DEFAULT_INITIAL_BUFFER_SIZE_BYTES,
    use_optimized_save: bool = True,
):
    """Wraps the trainer's checkpoint I/O with ML Flashpoint capabilities.

    This function is borrowed from NeMo with tweaks for ML Flashpoint. See
    https://sourcegraph.com/github.com/NVIDIA/NeMo@8c6fd8bfc6dbea32336ee53da9f0e06db979f520/-/blob/nemo/lightning/pytorch/local_ckpt.py?L116

    and https://docs.nvidia.com/nemo-framework/user-guide/latest/resiliency.html#local-checkpointing

    1. Determines if ML Flashpoint is enabled by looking for the
       `MLFlashpointCheckpointCallback` in the trainer's registered callbacks.
    2. Extracts `trainer`'s checkpoint_io implementation.
    3. Instantiates an `MLFlashpointCheckpointIO` wrapping the underlying checkpoint_io
       obtained from the trainer.
    4. Sets the trainer's checkpoint_io to be the wrapper `MLFlashpointCheckpointIO`, which itself is wrapped
       in an `AsyncFinalizableCheckpointIO` if the original trainer's checkpoint_io was.

    Args:
        trainer: The PyTorch Lightning trainer instance.
        flashpoint_base_container: The base container ID for ML Flashpoint checkpoints.
        ckpt_obj_manager: The singleton CheckpointObjectManager for this rank.
        replication_manager: The singleton ReplicationManager for this rank.
        async_save: Whether to enable asynchronous saving.
        checkpoint_loader: The checkpoint loader to use.
        always_save_context: Whether to always save the context. Defaults to `False`.
        write_thread_count: Optional. The number of threads to use for writing checkpoint data. Defaults to 1.
        initial_write_buffer_size_bytes: Optional. The initial size of the buffer for writing checkpoint data
            in bytes. Defaults to `DEFAULT_INITIAL_BUFFER_SIZE_BYTES`.

    Returns:
        None. The trainer's checkpoint_io is modified in-place.
    """
    if trainer is None:
        raise ValueError("The 'trainer' argument cannot be None.")
    if not flashpoint_base_container:
        raise ValueError("The 'flashpoint_base_container' argument cannot be empty.")
    if checkpoint_loader is None:
        raise ValueError("The 'checkpoint_loader' argument cannot be None.")
    if ckpt_obj_manager is None:
        raise ValueError("The 'ckpt_obj_manager' argument cannot be None.")
    if replication_manager is None:
        raise ValueError("The 'replication_manager' argument cannot be None.")
    if write_thread_count < 1:
        raise ValueError(f"write_thread_count must be >= 1, got {write_thread_count}.")
    if initial_write_buffer_size_bytes <= 0:
        raise ValueError(f"initial_write_buffer_size_bytes must be > 0, got {initial_write_buffer_size_bytes}.")

    callbacks = trainer.callbacks
    mlflashpoint_enabled = any(isinstance(cb, MLFlashpointCheckpointCallback) for cb in callbacks)
    if not mlflashpoint_enabled:
        return

    if not isinstance(trainer.strategy, nl_strategies.MegatronStrategy):
        raise ValueError(
            "Only MegatronStrategy is supported for ML Flashpoint, but got "
            + f"{trainer.strategy.__class__.__name__} instead."
        )

    checkpoint_io = trainer.strategy.checkpoint_io

    # Idempotency and validation check
    if isinstance(checkpoint_io, MLFlashpointAsyncFinalizableCheckpointIO):
        if not async_save:
            raise ValueError(
                "checkpoint_io is of type '%s', but async_save is False. This is an invalid configuration.",
                type(checkpoint_io),
            )
        return

    # When async_save is True, the checkpoint_io is wrapped within AsyncFinalizableCheckpointIO.
    # Thus, check if it is of that type, to extract the inner CheckpointIO object.
    async_finalizable_wrapped = False
    if isinstance(checkpoint_io, AsyncFinalizableCheckpointIO):
        async_finalizable_wrapped = True
        checkpoint_io = checkpoint_io.checkpoint_io

    # Idempotency check
    if isinstance(checkpoint_io, MLFlashpointCheckpointIO):
        return

    expected_type = MegatronCheckpointIO
    if not isinstance(checkpoint_io, expected_type):
        raise ValueError(
            f"Expected checkpoint_io to be of type '{expected_type.__name__}', but was: "
            + f"'{checkpoint_io.__class__.__name__}'."
        )

    # Use 'spawn' instead of 'fork' for the multiprocessing context.
    # By default, 'fork' causes the background SyncManager process to inherit
    # the parent's CUDA context. If the main training process is forcefully
    # killed (e.g., via SIGKILL during NVRX in-job restarts), the orphaned
    # manager process keeps the GPU memory locked, leading to CUDA Out-Of-Memory
    # (OOM) errors upon restart. 'spawn' launches a clean interpreter without
    # the inherited CUDA state, allowing the GPU memory to be freed instantly.
    ctx = torch_mp.get_context("spawn")
    save_strategy = MLFlashpointMegatronAsyncSaveStrategy(
        storage_writer=MemoryStorageWriter(
            checkpoint_saver=DefaultMLFlashpointCheckpointSaver(
                global_rank_getter=torch.distributed.get_rank,
                local_rank_getter=torch.distributed.get_node_local_rank,
                global_barrier_func=torch.distributed.barrier,
                ckpt_obj_manager=ckpt_obj_manager,
                replication_manager=replication_manager,
                initial_buffer_size_bytes=initial_write_buffer_size_bytes,
                use_optimized_save=use_optimized_save,
            ),
            thread_count=write_thread_count,
        )
    )
    load_strategy = MLFlashpointMegatronLoadStrategy(
        replication_manager=replication_manager,
        checkpoint_loader=checkpoint_loader,
    )

    ml_flashpoint_checkpoint_io = MLFlashpointCheckpointIO(
        flashpoint_base_path=flashpoint_base_container,
        alt_checkpoint_io=checkpoint_io,
        chkpt_obj_manager=ckpt_obj_manager,
        save_strategy=save_strategy,
        load_strategy=load_strategy,
        trainer=trainer,
        async_save=async_save,
        always_save_context=always_save_context,
    )

    if async_finalizable_wrapped:
        ml_flashpoint_checkpoint_io = MLFlashpointAsyncFinalizableCheckpointIO(ml_flashpoint_checkpoint_io)

    trainer.strategy.checkpoint_io = ml_flashpoint_checkpoint_io
