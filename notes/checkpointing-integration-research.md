# Checkpointing integration research notes

Reference notes for anyone (human or agent) extending ML Flashpoint into the Megatron Bridge / NeMo RL stack.
Everything here was read out of source, not documentation, on the dates below. Docs for these projects lag the code;
where they disagreed, the code won, and the disagreement is called out.

**Sources read**

| Repo | Ref | What was read |
|---|---|---|
| `google/ml-flashpoint` | `97f5c54` (main) | `src/ml_flashpoint/adapter/{megatron,nemo,pytorch}`, `src/ml_flashpoint/core` |
| `NVIDIA-NeMo/Megatron-Bridge` | `main` @ 2026-09-05 | `src/megatron/bridge/training/{checkpointing,train,setup,config}.py`, `utils/instantiate_utils.py`, `docs/training/checkpointing.md` |
| `NVIDIA-NeMo/RL` | `main` @ 2026-09-05 | `nemo_rl/models/megatron/setup.py`, `nemo_rl/models/policy/workers/megatron_policy_worker.py`, `nemo_rl/utils/checkpoint.py`, `nemo_rl/algorithms/*` |

Line numbers are from the refs above and will drift. Treat them as "look near here", not as addresses.

---

## 1. The one-paragraph version

Megatron Bridge added a public `CheckpointManager` protocol that a run can swap in via
`CheckpointConfig.custom_manager_class`. That is the right hook for ML Flashpoint, and the Megatron Bridge adapter uses
it. **NeMo RL does not use it.** NeMo RL builds its Megatron state with Bridge but calls Bridge's *functional*
`save_checkpoint` directly from its policy worker, so `custom_manager_class` is never read anywhere in that repo. A
NeMo RL integration therefore has to attach to the worker, not to the config.

---

## 2. Megatron Bridge

### 2.1 The custom checkpoint manager hook

`megatron/bridge/training/checkpointing.py`:

| Symbol | ~Line | Notes |
|---|---|---|
| `CheckpointSaveContext` | 622 | `state, model, optimizer, opt_param_scheduler, num_floating_point_operations_so_far, train_data_iterator, non_persistent_ckpt, pg_collection, module_name` |
| `CheckpointLoadContext` | 648 | `state, model, optimizer, opt_param_scheduler, strict, skip_load_to_model_and_opt, pg_collection, module_name` |
| `CheckpointManager` (Protocol) | 673 | `__init__(checkpoint_config)`, `save(ctx, callback_manager)`, `load(ctx) -> (int, int)`, `finalize_async_saves(state, blocking, terminate)` |
| `DefaultCheckpointManager` | 720 | Wraps the functional `save_checkpoint` / `load_checkpoint`. Owns `checkpointing_context`. |
| `create_checkpoint_manager` | 815 | Factory. Imports `custom_manager_class`, checks the protocol, and checks `save` accepts `(ctx, callback_manager)`. |

**Gotcha — the import allowlist.** `create_checkpoint_manager` calls
`megatron.bridge.utils.instantiate_utils._validate_target_prefix`, which rejects any target whose module prefix is not
allowlisted. The defaults cover `megatron.*`, `torch.*`, `transformers.*`, `nvidia.*`, `numpy.*`, `nemo.*` — **not**
`ml_flashpoint`. Call `megatron.bridge.utils.instantiate_utils.register_allowed_target_prefix("ml_flashpoint")` before
the factory runs or instantiation raises `InstantiationException`. The same validator also rejects any target with a
private (underscore-prefixed) path segment.

### 2.2 The two checkpoint cadences

`megatron/bridge/training/train.py::checkpoint_and_decide_exit` (~1425):

```python
if save and save_interval and step % save_interval == 0:
    save_checkpoint_and_time(..., non_persistent_ckpt=False)
elif save and non_persistent_save_interval and step % non_persistent_save_interval == 0:
    save_checkpoint_and_time(..., non_persistent_ckpt=True)
```

It is an `elif`, so a step that is a durable-checkpoint step never also takes the non-persistent branch. This is what
makes "durable on `save_interval`, ML Flashpoint on `non_persistent_save_interval`" collision-free without any extra
skip logic (contrast the NeMo 2.0 adapter, which needs `skip_every_n_steps`).

`save_checkpoint_and_time` (~1302) is also where the timings come from:

```python
timer_key = "save-checkpoint-non-persistent" if non_persistent_ckpt else "save-checkpoint"
timers(timer_key, log_level=0).start(barrier=True)
checkpoint_manager.save(CheckpointSaveContext(...), callback_manager)
timers(timer_key).stop(barrier=True)
timers.log([timer_key])
```

Barriers on both sides, so the logged value is the slowest rank — i.e. what the loop actually paid. These two timer
names are the measurement surface for any checkpoint-time experiment. Megatron logs them in **milliseconds** as
`name ....: (min, max)`.

`finalize_async_saves` is called from the train loop at ~394 (`blocking=False`, every iteration), ~816
(`blocking=True, terminate=False`) and ~845 / `_finish_train` (`blocking=True, terminate=True`).

### 2.3 `non_persistent_ckpt_type` and the local-checkpoint duck type

`save_checkpoint` (~1201) decides the checkpoint type:

* `non_persistent_ckpt and non_persistent_ckpt_type == "local"` → `CheckpointType.LOCAL`, and
  `save_dir = checkpointing_context["local_checkpoint_manager"].local_ckpt_dir`
* `non_persistent_ckpt and non_persistent_ckpt_type == "global"` → `CheckpointType.GLOBAL` into a `non_persistent/`
  subdir
* otherwise → `CheckpointType.GLOBAL`

Bridge only ever touches `checkpointing_context["local_checkpoint_manager"]` when
`non_persistent_ckpt_type == "local"`. The places it does:

| Where | ~Line | Call |
|---|---|---|
| `save_checkpoint` | 1209 | `.local_ckpt_dir` |
| `save_checkpoint` | 1496 | `.save(state_dict_for_save, step, is_async=...)` |
| `load_checkpoint` | 3279 | `.local_ckpt_dir` (LayerWise optimizer only) |
| `_get_non_persistent_iteration` | 3592 | `.find_latest()` |
| `_load_non_persistent_base_checkpoint` | 3640 | `.load()` |
| `setup._should_load_checkpoint` | 160 | `.find_latest() != -1` |

That last one is the important one: **`megatron/bridge/training/setup.py::_should_load_checkpoint` is how a run decides
whether to attempt a resume at all.** It reads `getattr(checkpoint_manager, "checkpointing_context", {})`. A custom
manager that wants its own checkpoints to trigger a resume must expose a `checkpointing_context` property containing a
`local_checkpoint_manager` with a `find_latest()` that returns a step (or `-1`). ML Flashpoint's
`MLFlashpointLocalCheckpointIndex` is exactly that duck type, and nothing more — it deliberately does not implement
`save()`/`load()`, because Bridge's own local paths expect an NVRx `MCoreTensorAwareStateDict` container that
ML Flashpoint does not produce. Before delegating to Bridge, `disable()` it so `find_latest()` reports `-1`.

`init_checkpointing_context` (~3409) raises if `non_persistent_ckpt_type == "local"` and `nvidia_resiliency_ext` is not
installed. A custom manager that provides its own local checkpointing should catch that and continue with `{}`.

### 2.4 Where strategies can and cannot be injected

* **Save: injectable.** `save_checkpoint` (~1402) does
  `if checkpointing_context is not None and "save_strategy" in checkpointing_context: save_strategy = ...` before
  building a `TorchDistSaveShardedStrategy`. Pre-seeding `checkpointing_context["save_strategy"]` works, and works for
  *any* caller of `save_checkpoint`, including NeMo RL.
* **Load: not injectable.** Both `_load_global_dist_base_checkpoint` (~3689) and `_load_model_weights_from_checkpoint`
  (~2420) construct `TorchDistLoadShardedStrategy()` unconditionally. The `checkpointing_context["load_strategy"]` key
  is *written* there but never read back as an override. A custom load path has to call
  `megatron.core.dist_checkpointing.load(...)` itself with its own `sharded_strategy`.

**Why the save-strategy hook is not enough on its own for a local checkpointer.** `dist_checkpointing.save` writes
`common.pt` via the common strategy on **global rank 0 only**, and writes into the directory the caller passed. For a
node-local memory checkpoint you need common state on *every node* (or no node but rank 0's can recover) and you need
the container under the node-local base path, not the durable one. ML Flashpoint's
`adapter/megatron/save_utils.py::save_local_aware_megatron_checkpoint` exists precisely to solve the first half: it
splits with `mcore_state_dict_utils.save_preprocess` and `torch.save`s the common part on every
`torch.distributed.get_node_local_rank() == 0`.

### 2.5 Rebuilding the state dict outside `save_checkpoint`

A custom manager that does not delegate has to assemble what `save_checkpoint` assembles. The pieces, all in
`checkpointing.py`:

* `get_rng_state(data_parallel_random_init, ckpt_format, *, pg_collection, module_name)` (~525) — gated on
  `ckpt_cfg.save_rng`.
* `get_rerun_state_machine().state_dict(data_iterator=..., ckpt_format=...)`.
* `_build_sharded_state_dict_metadata(use_distributed_optimizer, ckpt_cfg)` (~3991) — **private**. Then
  `metadata["dp_cp_group"] = pg_collection.dp_cp`.
* `generate_state_dict(ckpt_cfg, model, optimizer, opt_param_scheduler, rng_state, iteration=..., optim_sd_kwargs=dict(metadata=...), model_sd_kwargs=dict(metadata=...), rerun_state=..., pg_collection=...)` (~2180).
* For a **load** skeleton, the same call with `optim_sd_kwargs=dict(metadata=..., is_loading=True)` and no `iteration`.

`dp_cp_group` is a `ProcessGroup` and cannot be pickled — strip it before persisting the metadata. Do not rely on
`megatron.core.dist_checkpointing.utils._clean_metadata_for_serialization` to catch it; drop the key explicitly.

Local checkpoints have no `latest_train_state.pt` next to them, so Bridge embeds `state_dict["train_state_metadata"] =
train_state.state_dict()` for `CheckpointType.LOCAL` (~1487) and restores from it on load (~3175). Any custom local
format should do the same, plus carry cumulative FLOPs, since `load` must return
`(step, num_floating_point_operations_so_far)`.

Applying a loaded state dict mirrors `load_checkpoint` ~3170–3390: `set_checkpoint_version`, restore `TrainState`,
`update_num_microbatches`, `_load_model_state_dict` (**private**, ~2779) per chunk (`"model"` for one chunk, `"model%d"`
for many, skipping absent keys = empty PP stages), `optimizer.load_state_dict` under `torch.no_grad()`, scheduler from
`"lr_scheduler"` or `"opt_param_scheduler"`, rerun state, then RNG.

**Version drift to guard for.** These moved between Megatron Core releases and should be resolved defensively:

* `unwrap_model` — in `megatron.core.utils` on the mcore that Bridge 0.6 requires; in `megatron.training.utils` on
  older ones (e.g. mcore 0.13.1 has it in neither).
* `tensor_parallel.is_graph_safe_cuda_rng_tracker` / `tensor_parallel.convert_cuda_rng_state` — absent on older mcore.
  Fall back to setting tracker states directly.
* `_build_sharded_state_dict_metadata`, `_load_model_state_dict`, `_clean_metadata_for_serialization` are private and
  can move; `getattr` them and raise a clear error rather than failing at import.

### 2.6 Config surface

`CheckpointConfig` (`megatron/bridge/training/config.py` ~498) extends Megatron-LM's. Fields that matter here:
`save`, `load`, `save_interval`, `non_persistent_save_interval`, `non_persistent_ckpt_type`,
`non_persistent_local_ckpt_dir`, `async_save`, `async_strategy` (`"nvrx"` | `"mcore"`), `ckpt_format`
(`"torch_dist"` | `"fsdp_dtensor"` | `"torch"`), `ckpt_assume_constant_structure`, `most_recent_k`,
`fully_parallel_save`, `custom_manager_class`.

**The factory passes only `CheckpointConfig` to the custom manager.** There is nowhere in Bridge config to put
adapter-specific settings, so they have to come from a module-level registration call or environment variables.

### 2.7 Where the docs are stale

`docs/training/checkpointing.md` §"Implementing a Custom Manager" is broadly right but:

* it does not mention the `_validate_target_prefix` allowlist at all, which is the first thing a third-party manager
  hits;
* its `CheckpointSaveContext` / `CheckpointLoadContext` tables omit `pg_collection` and `module_name`, both of which
  the real dataclasses carry and `DefaultCheckpointManager` forwards;
* its example `save()` signature omits `pg_collection=` and `module_name=` in the `save_checkpoint` call.

---

## 3. NeMo RL

### 3.1 The headline finding

`grep -r "custom_manager_class\|create_checkpoint_manager"` across the whole NeMo RL repo (`.py`, `.yaml`, `.md`)
returns **zero hits**. NeMo RL uses Megatron Bridge for model/optimizer construction and for the checkpoint *functions*,
but never for the checkpoint *manager*.

What it actually does:

* `nemo_rl/models/megatron/setup.py:1858` — `checkpointing_context = init_checkpointing_context(megatron_cfg.checkpoint)`,
  stored on `ModelAndOptimizerState` and then on the worker as `self.checkpointing_context`
  (`megatron_policy_worker.py:586`).
* `nemo_rl/models/policy/workers/megatron_policy_worker.py:3714` — `MegatronPolicyWorker.save_checkpoint(weights_path,
  optimizer_path=None)`:
  1. `maybe_finalize_async_save(..., blocking=True)` — blocks on the *previous* save,
  2. onloads model (and optimizer) to CUDA, `torch.cuda.synchronize()`,
  3. temporarily overwrites `self.mcore_state.cfg.checkpoint.save = weights_path`,
  4. calls Bridge's functional `save_checkpoint(state=..., model=[self.model], ..., checkpointing_context=self.checkpointing_context)`,
  5. if sync, `maybe_finalize_async_save(..., blocking=True)` again,
  6. restores `cfg.checkpoint.save` in a `finally`.
* `MegatronPolicyWorker.load_checkpoint` (~3871) **raises `NotImplementedError`** — resume happens only through the
  worker's init path.

Consequences for an integration:

1. `custom_manager_class` is inert. Do not ship a NeMo RL story that depends on it.
2. `checkpointing_context["save_strategy"]` *would* be honoured (it flows into Bridge's `save_checkpoint`), but see
   §2.4 — that puts the container under NeMo RL's durable `weights_path` and writes `common.pt` on rank 0 only, which
   is wrong for a node-local checkpointer.
3. The worker attributes needed to build a Bridge `CheckpointSaveContext` are all present and stable:
   `worker.mcore_state` (a Bridge `GlobalState`), `worker.model`, `worker.optimizer`, `worker.scheduler`. Wrapping
   `worker.save_checkpoint` and driving a Bridge-shaped context from those is the least-coupled option, and is what
   `ml_flashpoint.adapter.nemo_rl.install_into_worker` does.

### 3.2 NeMo RL has no local/non-persistent cadence

`nemo_rl/utils/checkpoint.py::CheckpointingConfig` has `save_period`, `ft_save_period`, `ft_keep_latest_k`,
`keep_top_k`, `checkpoint_must_save_by`. In the algorithm loops (`grpo.py` ~3765, and the same shape in `ppo.py`,
`dpo.py`, `distillation.py`, `single_controller.py`):

```python
should_save_by_step = (
    is_last_step
    or early_stop_message is not None
    or (total_steps + 1) % checkpointing["save_period"] == 0
    or (ft_save_period is not None and (total_steps + 1) % ft_save_period == 0)
)
```

`ft_save_period` checkpoints go through the **same** `save_checkpoint` to the **same** `step_N` layout; they differ only
in retention. So there is no upstream notion of "cheap crash-recovery checkpoint" to map ML Flashpoint onto — unlike
Bridge's `non_persistent_save_interval`. Adding one means either a fork/upstream change, or deciding per-save on the
adapter side (which is why the adapter offers `augment` vs `replace` + `durable_every_n_saves`).

### 3.3 Measurement surface for NeMo RL

Bridge's `save_checkpoint_and_time` timers do **not** fire for NeMo RL, because NeMo RL calls `save_checkpoint`
directly and not through `train.py`. So `save-checkpoint` / `save-checkpoint-non-persistent` will be absent from a NeMo
RL log. The adapter therefore wraps the worker method with `log_execution_time(name="nemo_rl.save_checkpoint")`, and
that is the timer to compare across arms. Note it includes the blocking `maybe_finalize_async_save` for the *previous*
save, which is a fair thing to measure — that stall is real and is charged to the RL loop.

---

## 4. ML Flashpoint side

### 4.1 What already existed

`src/ml_flashpoint/adapter/megatron/` is framework-agnostic and reusable as-is:

* `save_strategies.MLFlashpointMegatronAsyncSaveStrategy` — an `AsyncSaveShardedStrategy`. Takes a
  `MemoryStorageWriter`; `async_save(sharded_state_dict, checkpoint_dir)` returns a Megatron `AsyncRequest` that the
  **caller must schedule**, and whose `preload_fn` must run before/at scheduling. It writes a stub
  `metadata.json` = `{"sharded_backend": ""}` into `checkpoint_dir` purely to satisfy Megatron's loader validation
  (all the checks against it are no-ops).
* `load_strategies.MLFlashpointMegatronLoadStrategy` — a `LoadShardedStrategy` used via
  `mcore_dist_checkpointing.load(..., sharded_strategy=<this>, common_strategy=TorchCommonLoadStrategy())`.
* `save_utils.save_local_aware_megatron_checkpoint` — the node-local `common.pt` writer described in §2.4. Swallows
  save exceptions and returns `None`.

Both strategies only handle `torch_dist`-shaped sharded state dicts. `fsdp_dtensor` and legacy `torch` are out.

### 4.2 Per-rank singletons

The NeMo 2.0 adapter's `wrapper_util.py` is the reference for what has to be constructed once per rank, and the
Megatron Bridge adapter's `runtime.py` mirrors it:

`BufferPoolConfig(pool_dir_path=<base>/buffer_pool, rank, num_buffers=threads*2, buffer_size)` →
`CheckpointObjectManager` → `ReplicationManager().initialize(...)` (a **collective**: needs
`torch.distributed` up on all ranks) → `DefaultMLFlashpointCheckpointSaver` → `MemoryStorageWriter` →
save strategy; and `DefaultMLFlashpointCheckpointLoader` → load strategy. Optionally wrapped in
`FullyParallel{Save,Load}StrategyWrapper`. Plus an `AsyncCallsQueue(persistent=True)`.

Two non-obvious details carried over from the NeMo adapter, both load-bearing:

* The `torch.multiprocessing` context must be **`spawn`**, not `fork`. A forked `SyncManager` inherits the CUDA
  context; if the trainer is SIGKILLed (NVRx in-job restart) the orphan pins GPU memory and the restart OOMs.
* On teardown, after closing the queue, monkeypatch `queue.persistent_caller.close = lambda: None`.
  `PersistentAsyncCaller.__del__` calls `close()` → `torch.distributed.get_rank()`, which crashes at interpreter
  shutdown once the process group is gone.

### 4.3 Separate async queues

`MLFlashpointAsyncFinalizableCheckpointIO` in the NeMo adapter keeps ML Flashpoint's `AsyncCallsQueue` separate from
the durable one, and the Bridge adapter does the same. The reason is not tidiness: a single queue finalizes in
*scheduling* order, not completion order, so fast ML Flashpoint finalizations queue behind a slow durable save while
new ones keep getting scheduled — the buffers stay pinned and the pool OOMs.

---

## 5. Things worth re-checking before trusting this

* None of the Megatron Bridge or NeMo RL integration paths have been executed. The unit tests stub
  `megatron.bridge.*` (see `tests/adapter/conftest.py`) and megatron-core 0.13.1 — the version this repo pins — is
  older than what Bridge 0.6 needs, so a real run needs the `megatron-bridge` extra and a matching mcore.
* `FullyParallelSaveStrategyWrapper` is constructed without a parallelization group (matching the NeMo adapter, which
  works). Bridge passes `pg_collection.dp_cp`. If sharding-distribution behaviour looks wrong, this is the first thing
  to look at.
* `megatron-bridge` on PyPI is at 0.6.0/0.6.1; `nemo-rl` on PyPI is a `0.0.0` placeholder — NeMo RL is installed from
  source, so the `nemo-rl` extra here only pulls the Bridge stack.
