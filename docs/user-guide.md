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
    # use_optimized_save=True, # Optional, defaults to True. Uses the optimized save method to reduce write time.
    # use_cached_ckpt_structure=False, # Optional, defaults to False. Caches the checkpoint structure after identifying 2 consecutive save plan structures that are equal.
    # use_fully_parallel_wrapper=True, # Optional, defaults to True. Uses the fully parallel wrapper for save and load.
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

### NeMo RL

The NeMo RL framework does not use PyTorch Lightning natively, and instead uses its own `CheckpointManager` and policy workers. ML Flashpoint provides a specialized wrapper adapter designed to inject fast checkpointing transparently into your training loops.

#### Imports

Import the NeMo RL wrapper provided by the ML Flashpoint adapter:

```python
from ml_flashpoint.adapter.nemo_rl.wrapper_util_rl import wrap_rl_components_with_mlflashpoint
from ml_flashpoint.core.checkpoint_loader import MLFlashpointCheckpointLoader
```

Additionally, you will need to instantiate your preferred save strategy to tell the manager how to commit ML Flashpoint blobs:

```python
from ml_flashpoint.adapter.megatron import MLFlashpointMegatronAsyncSaveStrategy
```

#### Recipe Changes

NeMo RL organizes its training entry points into Python scripts (like `examples/run_grpo.py`), which orchestrate the initialization steps and are driven heavily by configurations in YAML files.

Instead of modifying the upstream framework loops themselves (such as `async_grpo_train` in `nemo_rl/algorithms/grpo.py`), you should wrap the checkpointer instantiation within these NeMo RL script entry points.

```python
# 1. Your original NeMo RL initializers
# Typically instantiated via setup() or directly:
policy = Policy(cluster=train_cluster, config=policy_config, ...)
checkpointer = CheckpointManager(
    checkpoint_dir=args.checkpoint_dir,
    metric_name=args.metric_name, ...
)

# 2. Add the ML Flashpoint dual manager
flashpoint_save_strategy = MLFlashpointMegatronAsyncSaveStrategy(...)
checkpointer = wrap_rl_components_with_mlflashpoint(
    checkpointer=checkpointer,
    # Some tmpfs path for this job like /tmp/mlf/job-12345
    flashpoint_base_container=_get_my_mlf_base_path(),
    standard_save_period=1000, # Dictates when standard saves execute
    save_strategy=flashpoint_save_strategy,
    checkpoint_loader=MLFlashpointCheckpointLoader(...),
)

# 3. Supply the wrapper backwards as if it were the standard checkpointer
# For example, within GRPO:
async_grpo_train(
   policy=policy,
   checkpointer=checkpointer, # Dual checkpointer takes over routing
   ...
)
```

#### Limitations / Requisites

1. **Standard `save_period` override:** You must coordinate the standard save properties. The `save_period` configured inside your NeMo RL configurations (typically in the YAML config under `checkpointing: save_period: ...` or [see an example here](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/grpo_math_1B.yaml)) should now be set aggressively low (e.g. `1` or `10`), dictating how frequently *ML Flashpoint* triggers.
1. `standard_save_period` dictates how frequently your standard long-term persistence will actually run instead. For instance, configuring NeMo RL YAML `save_period: 10` and injecting `standard_save_period=1000` via our wrapper means ML Flashpoint saves every 10 steps, and standard checkpoints save every 1000 steps.

#### NeMo RL Configuration (Worker Side)

When using the custom worker extension (`MLFlashpointMegatronPolicyWorker`), it reads configuration from `self.cfg` (which is the `PolicyConfig` TypedDict passed during initialization).

You can define the `ml_flashpoint` configuration block in your recipe or config file. It should be a dictionary nested within the policy configuration.

##### Configuration Schema

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `enabled` | `bool` | No (default `True`) | Enable/disable ML Flashpoint on the worker. |
| `base_container` | `str` | **Yes** | The base directory (typically in `tmpfs`) for ML Flashpoint checkpoints. |
| `write_thread_count` | `int` | No (default `1`) | Number of threads for asynchronous writing. |
| `buffer_size_bytes` | `int` | No (default `16 GB`) | Size of the shared memory buffers in bytes. |

Example configuration in a YAML or dict:

```yaml
policy:
  ml_flashpoint:
    # enabled: true # default
    base_container: "/tmp/mlf-checkpoints/job-12345"
    # buffer_size_bytes: 17179869184 # default (16 GB)
```

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
import torch.distributed as dist
from ml_flashpoint.adapter.megatron.load_strategies import MLFlashpointMegatronLoadStrategy
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.core.checkpoint_loader import DefaultMLFlashpointCheckpointLoader
from ml_flashpoint.replication.replication_manager import ReplicationManager

# Megatron Checkpointing
from megatron.core import dist_checkpointing as mcore_dist_checkpointing
from ml_flashpoint.adapter.megatron.save_utils import save_local_aware_megatron_checkpoint
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
    # use_cached_ckpt_structure=True, # Optional, defaults to False. Caches the checkpoint structure after identifying 2 consecutive save plan structures that are equal.
)
```

Because Megatron's `dist_checkpointing.save()` function writes "common" data only on global rank 0, which does not align with local checkpointing, use the provided helper function `save_local_aware_megatron_checkpoint()` from the `ml_flashpoint.adapter.megatron.save_utils` module.

This helper mimics `dist_checkpointing.save()`, but saves common data on each node (via local rank 0) rather than solely on the coordinator node (global rank 0).

```python
# In your save loop
async_request = save_local_aware_megatron_checkpoint(
    checkpoint=state_dict,
    checkpoint_dir=str(curr_step_checkpoint_id),
    save_strategy=megatron_save_strategy,
    async_save=True,
)
```

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
    global_rank_getter=dist.get_rank,
    local_rank_getter=torch.distributed.get_node_local_rank,
    broadcast_object_list_func=dist.broadcast_object_list,
    all_gather_object_func=dist.all_gather_object,
    world_size_getter=dist.get_world_size,
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
import torch.distributed as dist
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
        # use_optimized_save=True, # Optional, defaults to True. Uses the optimized save method to reduce write time.
    ),
    mp_manager=torch_mp.Manager(),
)

# Instantiate the CheckpointLoader and StorageReader
checkpoint_loader = DefaultMLFlashpointCheckpointLoader(
    checkpoint_object_manager=checkpoint_object_manager,
    replication_manager=replication_manager,
    global_rank_getter=dist.get_rank,
    local_rank_getter=torch.distributed.get_node_local_rank,
    broadcast_object_list_func=dist.broadcast_object_list,
    all_gather_object_func=dist.all_gather_object,
    world_size_getter=dist.get_world_size,
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
