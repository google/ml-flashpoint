# Pre-GitHub Migration Releases

These are releases that were created prior to migrating to GitHub, for historical purposes.

## v0.0.3

_Release Notes: v0.0.2 -> 5e049c394923573331c2ed53e4f1cfeb8fd5d655_

-----

### :trophy: Features
* (e5e1bd9) transfer_service: Add detailed timing instrumentation for tasks.

### :white_check_mark: Bug Fixes
* (db2dc3d) core/saver: ensure tensor memory is contiguous

### :nut_and_bolt: Chores
* (5e049c3) oss: add license header to source code files
* (d39afe7) oss: add standard OSS files
* (fbf8e74) scripts: parse logs regardless of level for metric summary

-----

_Generated with: `./scripts/create_release.py --from-ref v0.0.2 --to-ref v0.0.3`_

## v0.0.2

_Release Notes: v0.0.1 -> fe9ab542a09e2a068b9975ddbed2e3b4e1534972_

-----

### :art: Styles
* (cae4aa0) transfer_service: Use formatted log in transfer_service and include rank information.

### :notebook_with_decorative_cover: Documentation
* (5301619) clarify system requirements for recovery reusing the same nodes

### :nut_and_bolt: Chores
* (fe9ab54) make default log level for execution time DEBUG
* (db98ed2) changelog: add v0.0.1 release changelog

-----

_Generated with: `./scripts/create_release.py --from-ref v0.0.1 --to-ref v0.0.2`_

## v0.0.1

_Release Notes: v0.0.0 -> 8267b6e03704359e803a2df164cac4466ae8d9c2_

-----

### :white_check_mark: Bug Fixes
* (8267b6e) adapter/nemo: ensure sharded_state_dict is passed on to fallback_checkpoint_io during recovery

### :notebook_with_decorative_cover: Documentation
* (886dd4a) user-guide: Add installation instructions.
* (2772bf7) user-guide: Update user guide and point to full recipe example.

### :nut_and_bolt: Chores
* (74cf5f2) update release script and instructions; add changelog

-----

_Generated with: `./scripts/create_release.py --from-ref v0.0.0 --to-ref v0.0.1`_

## v0.0.0

_Release Notes: BEGINNING -> 194b781e75807afaba682f9eef2826464fcc120e_

-----

### :trophy: Features
* (c60b248) checkpoint_loader: Implement get_latest_complete_checkpoint method with retrieval.
* (7b62c7c) adapter/nemo: add simplified wrapper API that returns MLFlashpointAutoResume
* (b8735f3) adapter/nemo: add enabled flag to easily disable Callback without removing it
* (8168a7b) core/saver: replicate checkpoint objects async
* (9d1316a) replication_manager: Implement sync_bulk_retrieve method.
* (d0b8770) replication/transfer_service: Save received data to tmp object before finalizing.
* (60dd014) replication_manager: Implement async_replicate of replication_manager.
* (3cb4c38) adapter/pytorch: make writer thread_count and buffer size configurable with defaults
* (af05afb) replication/transfer_service: Implement async_get method.
* (6c4fe99) replication/transfer_service: Implement async_put method.
* (e62cd71) replication/transfer_service: Implement transfer_service initialize and shutdown.
* (ece436c) transfer_service: Implement thread_pool, connection_pool and task_queue.
* (d62f183) logging: Customize logger to include rank information.
* (f25d749) core/saver: move global barrier into finalize_checkpoint; add tests
* (4acf1ad) core/saver: delete older checkpoint versions async
* (5adc690) wrapper_util: Init MLFlashpointCheckpointIO in wrapper.
* (06068da) adapter/nemo: Implement MLFlashpointCheckpointIO.
* (f41ff42) adapter/nemo: implement checkpoint_io wrapper helper for NeMo recipes
* (d906e75) adopt CheckpointObjectManager for all buffer operations
* (6fc0415) loader: assign values to storage_data in MemoryStorageReader.
* (d9af26b) adapter/megatron: Implement Megatron Save Strategy
* (f2b7d59) checkpoint: Add CheckpointObjectManager
* (308e8d2) adapter/pytorch: Implement custom state_dict_saver orchestration functions compatible with Megatron async strategy
* (8e43c36) adapter/pytorch: store list[WriteResult] per checkpoint_id in MemoryStorageWriter
* (c47ae8b) buffer_io: add BufferIO Python wrapper and unit tests
* (e67652f) loader: Implement loader.get_latest_complete_checkpoint and autoresume.get_trainer_ckpt_path.
* (cd1cf62) adapter/pytorch: Implement MemoryStorageWriter
* (7f9d91f) core/saver: Implement CheckpointSaver.write_metadata
* (d369fb7) core/saver: Implement CheckpointSaver.write_data
* (0f64718) core/saver: Implement CheckpointSaver.stage_data
* (14a1069) core/saver: implement initialize and finalize checkpoint APIs
* (c630452) loader: Implement data load.
* (7893174) loader: Implement metadata load.
* (6d54996) adapter/nemo: implement MLFlashpointCheckpointCallback; add CheckpointContainerId.from_parent() helper

### :white_check_mark: Bug Fixes
* (b33ddfa) wrapper_util: Expose write_thread_count and initial_write_buffer_size_bytes to user.
* (21cc19e) adapter/nemo: make CheckpointObjectManager a param to wrapper_util; passthrough kwargs in MLFlashpointAutoResume to parent
* (23daae9) Fix implementation of PairwiseReplicationStrategy and add more tests.
* (4620be7) core/saver: ensure writer can overwrite unfinished checkpoint data after recovery
* (cf4fb0c) core/loader: ensure loader works with Megatron expectations by using MCore Save/Load Planner
* (18157fe) scripts: mark missing step/rank as -1 to easily capture and report in logs
* (2604771) adapter/nemo: use separate AsyncCallsQueues for ML Flashpoint vs alt checkpointing
* (4c75106) adapter/nemo: remove missing log variable causing errors in the logs
* (39d34bb) adapter/megatron: remove non-existent finish_write parameter
* (7bc285a) adapter/pytorch: ensure global metadata is broadcasted in generate_plan
* (624ff44) adapter/nemo: ensure common.pt is loaded by using megatron dist_checkpointing.load
* (9fa03d2) adapter/nemo: ensure the actual ckpt version path is used, not the base dir, for save and load
* (4febac0) adapter/nemo: extract "common" state dict and save it as common.pt file
* (95d580f) adapter/nemo: handle string base_container and subclass Callback directly
* (f4280a1) loader: Fix read_data errors.

### :clock1: Performance
* (ee6b032) core/saver: restructure to make staging non-blocking by default to improve execution time
* (36e4597) checkpoint_object_manager: make delete dirs actually async by detaching thread

### :recycle: Refactoring
* (7afe61d) replication: Refactor `ReplicationManager` to allow dependency injection and add retry configuration.

### :art: Styles
* (1422b42) Use placeholder logging and _LOGGER.exception
* (cbe0d0c) reformat python code according to `ruff format` for consistency
* (9b0690d) checkpoint_object_manager: improve logging and test organization
* (0401fc8) enforce python import ordering in lint
* (a65fcc6) python: adopt Google-style docstrings; fix formatting; update GEMINI.md

### :mag: Tests
* (2ef0a06) add test coverage scripts and setup for python and C++
* (d2adeb6) checkpoint_object_manager: make boolean assertions explicit
* (3edd52d) core/saver: make remove_older_checkpoints tests less flaky on param order
* (78fed2f) adapter/nemo: add extra MLFlashpointCheckpointIO initialization tests.
* (eb5fbd6) adapter/megatron: add failure case tests for Megatron Save Strategy
* (38fcace) buffer_object: add python bindings and C++ tests
* (31eb6a0) replication: Set up test framework for transfer service && fix import error for buffer object module.

### :notebook_with_decorative_cover: Documentation
* (35a6e97) add user docs via mkdocs-material
* (3a23d0e) add missing Args and Returns sections to docstrings
* (d990464) format docstrings in Google style and update Args/Return sections where outdated
* (adeca18) readme: add test placement and naming conventions; fix test command comment

### :twisted_rightwards_arrows: Build
* (46398fa) upgrade C++ 17->20; remove redundant CXX settings in child build files
* (084c925) cmake: move parent CMakeLists.txt to top-level
* (a4f79c5) cmake: only configure and link test code & deps if BUILD_TESTING enabled
* (c0be1bb) python: relax python version requirements, set .python-version
* (5ff0513) python: downgrade to python3.10 for nemo dependencies; depend on nemo_toolkit[all]
* (def9a09) configure pytest-cpp to discover and run C++ tests via `pytest`

### :arrows_clockwise: CI
* (2a35fee) add release script and README instructions for it
* (97ad8ca) add lint check to presubmit
* (5773ef7) Add presubmit file. This file will be used by louhi workflow.

### :nut_and_bolt: Chores
* (5ab3f95) deps: Upgrade abseil-cpp 20230125.3 -> 20250814.1
* (32b4089) scripts: capture and report train_step_timing and avg of all metrics
* (34278e8) scripts: make parse_log_and_summarize.py executable
* (ef2555f) scripts: remove unused list in parse_log_and_summarize.py
* (802beea) add script to analyze and summarize execution time logs
* (6d5bb44) core: add training step to logs and shorten format
* (119620b) add @log_execution_time decorator to measure function latency for meaningful, infrequent operations
* (c24fbb7) adapter/pytorch: make storage_data field in MemoryStorageReader "private" and declare up front
* (526cc15) replication: rename replication_manager/ to replication/
* (1c456cf) core: enhance CheckpointId and subclass validations and helper APIs
* (787f255) adapter/nemo: Add MLFlashpointAutoResume class skeleton
* (d92100d) add .gitattributes to normalize line endings to LF
* (642d85c) buffer: Add CheckpointObjectManager, BufferIO, BufferObject skeleton.
* (682f525) Add MLFlashpointMegatronLoadStrategy skeleton; update Save equivalent class name and docs [b/445404891]
* (5470074) Add MLFlashpointCheckpointIO skeleton with nemo dependencies, and update docs throughout
* (d488b5b) setup initial project structure and skeleton APIs.

### ¯\(ツ)/¯ Other
* (4479f3e) Merge "chore(core): enhance CheckpointId and subclass validations and helper APIs" into main
* (c146097) Implement BufferObject for memory-mapped file I/O
* (fa758c0) Add clang-format to enforce cpp files format.
* (540ad88) Merge "feat(adapter/nemo): implement MLFlashpointCheckpointCallback; add CheckpointContainerId.from_parent() helper" into main
* (2f79c13) Merge "chore: add .gitattributes to normalize line endings to LF" into main
* (9e13adc) Add ReplicationManager and TransferService skeleton.
* (0dabb44) Initial empty repository

-----

_Generated with: `./scripts/create_release.py --to-ref v0.0.0`_
