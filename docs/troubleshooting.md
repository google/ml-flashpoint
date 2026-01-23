# Troubleshooting

## FAQ

### Checkpoint saves are running out of space, what can I do?

1. Ensure you have sufficient space on your base container mount.
1. If you have enough memory, but are running out of buffer space during writes, you can:
    1. Increase the default initial buffer size via `initial_write_buffer_size_bytes` in the `wrap` API you are using (the default is 16 GB).
    1. Increase the write thread count, so that each rank writes to multiple buffers, effectively cutting the size of each buffer proportionally, via `write_thread_count` in the `wrap` API you are using (the default is 1).

### How can I clean up ML Flashpoint checkpoints after job completion?

We may add some utility to do so automatically, but for now (and for safety), you can run a command explicitly on each node after job completion to delete the entire base container on each node.
This can be a part of your training script.
If you are using a `tmpfs` mount as recommended, it will automatically be deleted upon reboot.

### If nothing else works, how can I quickly disable ML Flashpoint?

To quickly disable ML Flashpoint:

* For NeMo 2.0 and PyTorch Lightning
    * Saving: You can simply set `enabled=False` when initializing `MLFlashpointCheckpointCallback` (it defaults to `True`).
      This will disable checkpoint saves (you may see logs saying they are being skipped).
    * Recovering: If there are no ML Flashpoint checkpoints saved in the specified base container, the `MLFlashpointAutoResume` will simply delegate to the parent `nemo.lightning.AutoResume`.
      If there are ML Flashpoint checkpoints, to avoid recovering from them, you can either delete them from all nodes in the training cluster, or skip the wrap call e.g. comment out `wrap_trainer_and_auto_resume_with_mlflashpoint(...)` from your recipe.
    * This can be done when resuming from a job interruption. 
    It will attempt to resume using your regular `AutoResume` configuration, if any.
* For Megatron-LM
    * Swap `MLFlashpointMegatronAsyncSaveStrategy` with an alternative like `TorchDistSaveShardedStrategy`.
    * Swap `MLFlashpointMegatronLoadStrategy` with an alternative like `TorchDistLoadShardedStrategy`.
* For PyTorch DCP
    * Swap `MemoryStorageWriter` with an alternative like `FileSystemWriter`.
    * Swap `MemoryStorageReader` with an alternative like `FileSystemReader`.

## Contact

To raise bugs, questions or feature requests, please see if there is already an [issue](https://github.com/google/ml-flashpoint/issues) for it, and if not, create a [new one](https://github.com/google/ml-flashpoint/issues/new/choose).
