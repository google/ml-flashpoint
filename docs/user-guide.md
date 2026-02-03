# User Guide

Below are instructions for using ML Flashpoint with the different frameworks supported.
For finer-grained control, use the [core](https://github.com/google/ml-flashpoint/tree/main/src/ml_flashpoint/core) library APIs, which the framework adapters build on top of.
The adapters also provide a good working example of how to use the core library.

If interested in a native integration with another framework, please let us know by creating a [feature request](https://github.com/google/ml-flashpoint/issues/new?template=feature_request.md) or upvoting an [existing one](https://github.com/google/ml-flashpoint/issues?q=is%3Aissue%20state%3Aopen%20label%3Aenhancement).

## Install

You can install from source:

```bash
# Clone the repo
git clone https://github.com/google/ml-flashpoint.git
# Install in editable mode
pip install -e ml-flashpoint
```

This assumes you are managing your own dependencies for PyTorch, Megatron-LM, NeMo, etc.

To install with this library's adapter-specific dependencies, specify the adapter of interest, such as `nemo`, `megatron`, `pytorch`:

```bash
# Example for the NeMo adapter.
pip install -e ml-flashpoint[nemo]
```

See the project's [README](http://cs/h/cloud-mlnet/ml-flashpoint/+/main:README.md) and [pyproject.toml](http://cs/h/cloud-mlnet/ml-flashpoint/+/main:pyproject.toml) for the latest and more detailed info.

## Frameworks

### NeMo 2.0 & Pytorch Lightning

Code: See the [`ml_flashpoint.adapter.nemo`](https://github.com/google/ml-flashpoint/tree/main/src/ml_flashpoint/adapter/nemo) package.

!!! note

    NeMo 2.0 relies on PyTorch Lightning, so the usage for either is very similar.
    NeMo provides additional logic for discovering the checkpoint path to resume from, if any, via `AutoResume`.

In your recipe script, once you've determined a base container path for the job (e.g. `/tmp/mlf-jobs/job-145`), add the following:

#### Imports

```python
import os
from ml_flashpoint.adapter.nemo.wrapper_util import wrap_trainer_and_auto_resume_with_mlflashpoint
from ml_flashpoint.adapter.nemo.checkpoint_callback import MLFlashpointCheckpointCallback
```

#### Recipe Changes

1. Determine the base path however you choose to.
```python
mlflashpoint_base_path = _get_my_mlf_base_path()

# Ensure the base path exists on each node.
os.makedirs(mlflashpoint_base_path, exist_ok=True)
```

See the [system requirements](README.md#systemenvironment-requirements) for how to set up the filesystem to use.

2. Configure the callback to trigger saves periodically.
```python
# Add this callback to your Trainer's callbacks.
callbacks.append(
    MLFlashpointCheckpointCallback(
        mlflashpoint_base_path,
        args.ckpt_mlf_every_n_steps, # How frequently to save ML Flashpoint checkpoints
        skip_every_n_steps=args.ckpt_std_every_n_steps, # How frequently the standard checkpointing strategy will run, to skip those steps
    )
)
```

3. Wrap the Trainer's CheckpointIO and the AutoResume instance.

```python
# Your standard AutoResume definition (optional).
auto_resume = nemo.lightning.AutoResume(...)

# Ensure the process group is set up if it wasn't already, for
# ML Flashpoint APIs below (and its AutoResume), to initialize replication.
trainer.strategy.setup_environment()

# Wrap the trainer and AutoResume to configure ML Flashpoint using the provided helper.
auto_resume = wrap_trainer_and_auto_resume_with_mlflashpoint(
    trainer=trainer, # The PyTorch Lightning Trainer
    flashpoint_base_container=mlflashpoint_base_path,
    async_save=not args.sync_save,
    default_auto_resume=auto_resume, # Optional
    # always_save_context=False, # Optional, defaults to False
    # write_thread_count=1, # Optional, defaults to 1
    # initial_write_buffer_size_bytes=DESIRED_NUM_BYTES, # Optional, defaults to 16 GB
)
```

And then you can use `trainer` and `auto_resume` as you normally do.

A complete recipe example that puts this all together can be found [here](http://google3/cloud/ml/applications/vision/model_garden/model_oss/train/nemo/recipes/pretrain/llama3p1_8b_cpt.py;rcl=842312043).

Limitations:

1. You must use the `MegatronStrategy` as the strategy for your PyTorch Lightning Trainer.
Other strategies have not been tested.
1. Ensure that the `base_container` for ML Flashpoint is job-specific (i.e. has a job ID in it), and on some ramdisk path (e.g. tmpfs).
The job ID should be unique across jobs, but sticky (reused) when a job is interrupted and restarted/rescheduled (so it can recover from the latest checkpoint available for that particular job).
New jobs however should have an independent job ID, so as not to conflict with prior jobs' checkpoints.
1. It is recommended to supply the `MLFlashpointCheckpointCallback` with the standard checkpoint strategy's interval (its `every_n_steps` configuration), so ML Flashpoint can skip its own saves when the standard strategy will save.
This reduces blocking time by avoiding duplicate work, at the cost of having a longer write time for that step.

### Megatron-LM

Code: See the [`ml_flashpoint.adapter.megatron`](https://github.com/google/ml-flashpoint/tree/main/src/ml_flashpoint/adapter/megatron) package.

The Megatron strategies depend on the PyTorch DCP implementations.
Below are instructions for setting up ML Flashpoint checkpointing, which you should configure alongside regular checkpointing to long-term storage.

#### Imports

```python
# Saving
from ml_flashpoint.adapter.pytorch.memory_storage_writer import MemoryStorageWriter
from ml_flashpoint.adapter.megatron.save_strategies import (
    MLFlashpointMegatronAsyncSaveStrategy,
)

# Loading
from ml_flashpoint.adapter.megatron.load_strategies import MLFlashpointMegatronLoadStrategy
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.core.checkpoint_loader import DefaultMLFlashpointCheckpointLoader
from ml_flashpoint.replication.replication_manager import ReplicationManager

# Megatron Checkpointing
from megatron.core import dist_checkpointing as mcore_dist_checkpointing
```

#### Save Strategy

First create a `MemoryStorageWriter`  instance as outlined in [PyTorch DCP](#pytorch-dcp).
Then use that to instantiate the Megatron save strategy.

```python
# Instantiate the MemoryStorageWriter
memory_storage_writer = MemoryStorageWriter(...)

# Use it to instantiate the Save Strategy
megatron_save_strategy = MLFlashpointMegatronAsyncSaveStrategy(
    storage_writer=memory_storage_writer,
)
```

Because Megatron's `dist_checkpointing.save()` function writes "common" data only on global rank 0, which does not align with local checkpointing, you can orchestrate saves using the save strategy the same way it's done in [`MLFlashpointCheckpointIO.save_checkpoint()`](https://github.com/google/ml-flashpoint/blob/b9767583520106f59743b9e8050769523cfbef6e/src/ml_flashpoint/adapter/nemo/checkpoint_io.py#L137-L171) in the `ml_flashpoint.adapter.nemo` package.
You'll notice that the logic there aims to mimic `dist_checkpointing.save`, but it saves common data on each node (via local rank 0) as opposed to solely on the coordinator node (global rank 0).

!!! note

    Make sure to specify the checkpoint ID/path when saving based on the current step using: 
    `CheckpointContainerId.create_child(base_container, CheckpointContainerId.format_version_container(current_step))`
    where `base_container` is the base path CheckpointContainerId used for all checkpoints for the current job, e.g. `"/tmp/mlf-checkpoints/job123"`.


Use this strategy on a more frequent interval than your regular long-term storage checkpointing strategy.

#### Load Strategy

Instantiate the singleton `ReplicationManager` with a singleton `CheckpointObjectManager`, and make sure to `initialize()` the `ReplicationManager` before using it.
Also create an `MLFlashpointCheckpointLoader` with those dependencies, and use these instances to create the load strategy:

```python
# Initialize dependencies (shared singletons)
checkpoint_object_manager = CheckpointObjectManager()
replication_manager = ReplicationManager()
replication_manager.initialize(checkpoint_object_manager)

checkpoint_loader = DefaultMLFlashpointCheckpointLoader(
    checkpoint_object_manager=checkpoint_object_manager,
    replication_manager=replication_manager,
)

# Instantiate the Load Strategy with the dependencies
mlflashpoint_load_strategy = MLFlashpointMegatronLoadStrategy(
    replication_manager=replication_manager,
    checkpoint_loader=checkpoint_loader,
)
```

Now you can use the load strategy with Megatron-LM's `dist_checkpointing.load` function directly:

```python
# First determine if an ML Flashpoint checkpoint is available, using the base container path you've configured
latest_saved_checkpoint_id = checkpoint_loader.get_latest_complete_checkpoint(checkpoint_base_container)

if local_checkpoint_container:
    # Given the existing load function doesn't do anything rank-specific,
    # it is suitable for us to use directly.
    state_dict = mcore_dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict,
        checkpoint_dir=str(latest_saved_checkpoint_id),
        sharded_strategy=mlflashpoint_load_strategy,
        common_strategy=TorchCommonLoadStrategy(),
    )
else:
    # Load using your regular sharded strategy from your long-term storage path
    state_dict = mcore_dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict,
        checkpoint_dir=str(long_term_storage_path),
        sharded_strategy=regular_megatron_load_strategy,
        common_strategy=TorchCommonLoadStrategy(),
    )
```

### PyTorch DCP

Code: See the [`ml_flashpoint.adapter.pytorch`](https://github.com/google/ml-flashpoint/tree/main/src/ml_flashpoint/adapter/pytorch) package.

To use directly with PyTorch DCP, use the provided `StorageWriter` and `StorageReader` implementations.
You can use whatever `Planner` implementations work for your use case, or resort to the defaults.

If your per-rank checkpoint data exceeds the default buffer size (16 GB as of this writing), you can increase it using the optional `initial_buffer_size_bytes` parameter. 

#### Imports
```python
import torch
from torch import multiprocessing as torch_mp
import torch.distributed.checkpoint as dcp

from ml_flashpoint.adapter.megatron.load_strategies import MLFlashpointMegatronLoadStrategy
from ml_flashpoint.adapter.pytorch.memory_storage_writer import MemoryStorageWriter
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId
from ml_flashpoint.core.checkpoint_loader import DefaultMLFlashpointCheckpointLoader
from ml_flashpoint.core.checkpoint_saver import DefaultMLFlashpointCheckpointSaver
from ml_flashpoint.replication.replication_manager import ReplicationManager
```

#### Initialization
```python
# Initialize dependencies (shared singletons)
checkpoint_object_manager = CheckpointObjectManager()
replication_manager = ReplicationManager()
replication_manager.initialize(checkpoint_object_manager)

# Instantiate the StorageWriter
memory_storage_writer = MemoryStorageWriter(
    checkpoint_saver=DefaultMLFlashpointCheckpointSaver(
        global_rank_getter=torch.distributed.get_rank,
        local_rank_getter=torch.distributed.get_node_local_rank,
        global_barrier_func=lambda: torch.distributed.barrier(),
        ckpt_obj_manager=checkpoint_object_manager,
        replication_manager=replication_manager,
        # initial_buffer_size_bytes=initial_write_buffer_size_bytes, # Optional - increase for larger checkpoint sizes per rank
    ),
    mp_manager=torch_mp.Manager(),
)

# Instantiate the CheckpointLoader and StorageReader
checkpoint_loader = DefaultMLFlashpointCheckpointLoader(
    checkpoint_object_manager=checkpoint_object_manager,
    replication_manager=replication_manager,
)
memory_storage_reader = MemoryStorageReader(
    path=checkpoint_dir,
    checkpoint_loader=checkpoint_loader,
)
```

#### Saving

Now you can use the `MemoryStorageWriter` when saving checkpoints as you normally do with DCP e.g.:

```python
# Assuming base_container is the base path CheckpointContainerId used for all checkpoints for the current job, e.g. `"/tmp/mlf-checkpoints/job123"`:
curr_step_checkpoint_id = CheckpointContainerId.create_child(
    base_container, CheckpointContainerId.format_version_container(current_step)
)

# Sync save
metadata = dcp.save(state_dict,
                    checkpoint_id=str(curr_step_checkpoint_id),
                    storage_writer=memory_storage_writer)

# Async save
future = dcp.async_save(state_dict,
                        checkpoint_id=str(curr_step_checkpoint_id),
                        storage_writer=memory_storage_writer,
                        async_checkpointer_type=dcp.AsyncCheckpointerType.PROCESS)
```

#### Recovery
During a recovery scenario, use the `checkpoint_loader` to first identify the latest available ML Flashpoint checkpoint, if any, to recover from.
If none, fallback to your long-term storage checkpoint.

```python
# First determine if an ML Flashpoint checkpoint is available, using the base container path you've configured
latest_saved_checkpoint_id = checkpoint_loader.get_latest_complete_checkpoint(checkpoint_base_container)

if latest_saved_checkpoint_id:
    dcp.load(state_dict,
             checkpoint_id=str(latest_saved_checkpoint_id),
             storage_reader=memory_storage_reader)
else:
    # Load using your regular sharded strategy from your long-term storage path
    dcp.load(state_dict,
             checkpoint_id=str(long_term_checkpoint_path),
             ...)
```
