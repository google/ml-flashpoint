# User Guide

Below are instructions for using ML Flashpoint with the different frameworks supported.

## Install

You can install from source:

```bash
# Clone the repo
git clone https://github.com/google/ml-flashpoint.git
# Install in editable mode
pip install -e ml-flashpoint
```

This assumes you are managing your own dependencies for PyTorch, Megatron-LM, NeMo, etc.

To install with adapter specific dependencies, specify the adapter of interest, such as `nemo`, `megatron`, `pytorch`:

```bash
# Example for the NeMo adapter.
pip install -e ml-flashpoint[nemo]
```

See the project's [README](http://cs/h/cloud-mlnet/ml-flashpoint/+/main:README.md) and [pyproject.toml](http://cs/h/cloud-mlnet/ml-flashpoint/+/main:pyproject.toml) for the latest and more detailed info.

## Frameworks

### NeMo 2.0 & Pytorch Lightning

Code: Check out the `adapter/nemo` package.

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

2. Configure the callback, to trigger saves periodically.
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

# Wrap the trainer and AutoResume to configure ML Flashpoint.
auto_resume = wrap_trainer_and_auto_resume_with_mlflashpoint(
    trainer=trainer, # The PyTorch Lightning Trainer
    flashpoint_base_container=mlflashpoint_base_path,
    async_save=not args.sync_save,
    default_auto_resume=auto_resume, # Optional
    # always_save_context=True, # Optional, defaults to False
    # write_thread_count=1, # Optional, defaults to 1
    # initial_write_buffer_size_bytes=DESIRED_NUM_BYTES, # Optional, defaults to 16 GB
)
```

And then you can use `trainer` and `auto_resume` as you normally do.

A complete recipe example that puts this all together can be found [here](http://google3/cloud/ml/applications/vision/model_garden/model_oss/train/nemo/recipes/pretrain/llama3p1_8b_cpt.py;rcl=842312043).

Limitations:

1. Must use the `MegatronStrategy` as the strategy for your PyTorch Lightning Trainer.
Other strategies have not been tested.
1. Ensure that the `base_container` for ML Flashpoint is job-specific (i.e. has a job ID in it), and on some ramdisk path (e.g. tmpfs).
The job ID should be unique across jobs, but sticky (reused) when a job is interrupted and restarted/rescheduled (so it can recover from the latest checkpoint available for that particular job).
New jobs however should have an independent job ID, so as not to conflict with prior jobs' checkpoints.
1. It is recommended to supply the `MLFlashpointCheckpointCallback` with the standard checkpoint strategy's interval (its `every_n_steps` configuration), so ML Flashpoint can skip its own saves when the standard strategy will save.
This reduces blocking time by avoiding duplicate work, at the cost of having a longer write time for that step.

### Megatron-LM

Check out the `adapter/megatron` package.

### PyTorch DCP

Check out the `adapter/pytorch` package.
