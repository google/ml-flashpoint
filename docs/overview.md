# Overview

ML Flashpoint is a memory-first checkpointing solution designed specifically for fast ML crash recovery.
It operates as a complementary layer to traditional persistent storage strategies.

## Core Concepts

* **Containers vs. Objects**: Checkpoints are stored in a hierarchical structure.
A `CheckpointContainerId` represents a checkpoint version within a job (e.g. a directory like `/tmp/job-123/step-100_ckpt`), while a `CheckpointObjectId` represents a specific data blob (e.g. a file) within a `CheckpointContainerId`.
`CheckpointObjectId`s are used to represent checkpoint data, but not necessarily smaller metadata-like files.
* **Memory-First Storage**: Data is written to mmap-ed buffers in CPU RAM (typically via tmpfs). 
This avoids the latency of disk storage and traditional network filesystems.

## The Checkpoint Save Flow

The saving process is designed to minimize the training loop's "critical path" (blocking time).
Asynchronous saving is recommended to maximize ML runtime goodput.
Synchronous checkpointing saves are expected to be significantly faster than alternatives' synchronous checkpointing, but asynchronous checkpointing will have lower blocking time.

### High-level Operations
This is the general sequence of operations.
If async saving is used, writing will be done async, and potentially the other steps as well (depending on the layer used and how).

1. **Initialization**: A "dirty" marker file (suffixed with `_unfinished`) is created on each node before the checkpoint directory is even registered.
This prevents recovery from a partial or corrupted checkpoint.
1. **Planning**: The system determines which ranks write which portions of the `state_dict` using planners (e.g. `MCoreSavePlanner` if using Megatron/NeMo).
1. **Staging (Critical Path)**: Data is copied from GPU memory to CPU memory.
This is (currently) a key blocking step.
Once complete, if using async saves, training resumes while (at least some of) the remaining steps happen in the background.
1. **Writing**: An `MLFlashpointCheckpointSaver` uses `CheckpointObjectManager` to obtain a memory buffer (`BufferObject`), and writes data to it using a custom structure.
1. **Replication**: Once local writing is finished, the `MLFlashpointCheckpointSaver` uses the `ReplicationManager` to asynchronously replicate the buffer to designated peer(s).
This step is always done asynchronously, even for synchronous checkpoint saves.
1. **Metadata Persistence**: A global metadata file (`.metadata`) is written atomically to all local rank 0s after ranks have gathered local results from all nodes.
1. **Finalization**: Ranks enter a barrier to ensure global completion.
The dirty marker is removed, and prior checkpoint versions are deleted asynchronously.

## The Checkpoint Load Flow (Recovery)

Recovery prioritizes finding and using the most recent in-memory state available across the cluster.
The general expectation is to first attempt to find an in-memory ML Flashpoint checkpoint, and fallback to your long-term checkpoint storage otherwise.

Recovery is always done synchronously, as training is dependent on it.

### High-level Operations
1. **Candidate Identification**: Each node scans local storage for complete checkpoint containers (those without dirty markers).
An `all-gather` or equivalent operation finds the intersection of these candidates across the cluster.
1. **Latest Version Selection**: Candidates are sorted chronologically.
The system iterates from newest to oldest to find the first version where all globally expected files are available.
If none is found, it short-circuits to fallback to long-term storage (which is framework/implementation dependent).
1. **Alternative Storage Fallback**: If no valid ML Flashpoint checkpoint is found, recovery falls back to the persistent storage implementation (e.g., standard NeMo `AutoResume` or PyTorch `FileSystemReader` pointing to some shared filesystem).
Otherwise, it continues.
1. **Data Retrieval**: If a node has been replaced or its memory was cleared, it must retrieve its required shards from peers:
    * The node identifies missing objects using the source rank embedded in filenames.
    * It requests missing data from peers via the `ReplicationManager`'s retrieval API.
1. **Loading**: Data is read from shared memory buffers and deserialized back into the model's `state_dict`.
This uses load planners (e.g. `MCoreLoadPlanner` if using Megatron/NeMo) and ultimately an `MLFlashpointCheckpointLoader` to read and deserialize the `state_dict`.
