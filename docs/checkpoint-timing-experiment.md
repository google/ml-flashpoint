# Measuring the checkpoint-time difference

This is the procedure for validating an ML Flashpoint integration by measuring what checkpointing costs the training
loop, with and without it. It is deliberately short: a handful of steps is enough, because the quantity of interest is
per-checkpoint wall clock, not convergence.

!!! note

    The numbers below are placeholders. This page describes how to produce them; it does not report a result. Fill in
    the results table from your own run.

## What is being measured

Megatron Bridge brackets each checkpoint with barriers and logs the elapsed time from
`megatron.bridge.training.train.save_checkpoint_and_time`:

* `save-checkpoint` — a durable checkpoint.
* `save-checkpoint-non-persistent` — a non-persistent one, which is the ML Flashpoint checkpoint when the adapter is
  enabled.

Both are logged through Megatron's timers in **milliseconds**, in one of two shapes depending on
`logger.timing_log_option`:

```
(min, max) time across ranks (ms):
    save-checkpoint ................................: (18450.20, 18512.90)   # minmax (default)
    save-checkpoint ................................: 18512.90               # max
```

The parser reads either shape and always keeps the **max** — the slowest rank — because the save is barrier-bracketed
on both sides, so the whole job waits for that rank. `(18450.20, 18512.90)` therefore contributes one sample of
**18.51 s**.

For NeMo RL, Bridge's timers never fire at all: NeMo RL calls `save_checkpoint` directly rather than through
`train.py`. The adapter emits `nemo_rl.save_checkpoint` instead, covering the whole worker save including the blocking
`maybe_finalize_async_save` that precedes it.

The headline comparison is the mean and max of those timers between two runs that differ only in whether ML Flashpoint
is enabled.

## Cluster setup

The workload is the prebuilt NVIDIA NeMo RL job for Google Cloud training clusters, described in
[Run prebuilt workloads](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/training/training-clusters/run-prebuilt-workloads#nvidia-nemo-rl).
Follow that page to create the cluster and get a working NeMo RL run first; do not add ML Flashpoint until an unmodified
run completes and logs checkpoint timings.

Requirements specific to this experiment:

* **At least two nodes.** ML Flashpoint replicates each node's checkpoint objects to a peer, and a single-node run does
  not exercise that path.
* **`/dev/shm` sized for the checkpoint.** Each node holds its own shard plus a peer's replica, so size shared memory to
  at least twice the per-node checkpoint plus headroom. The pods' shared-memory volume default is usually too small.
* **A durable checkpoint destination** (a GCS mount or a network filesystem) for the baseline arm and for the durable
  cadence of the ML Flashpoint arm. Both arms must write durable checkpoints to the same kind of destination, or the
  comparison measures the storage backend rather than the adapter.
* **The same node pool, model, parallelism and batch size across both arms.** Run them back to back.

## Run configuration

Keep the run short and make it checkpoint often enough to collect several samples:

* 20–30 training steps.
* Durable checkpoints every 10 steps, giving 2–3 `save-checkpoint` samples per arm.
* ML Flashpoint checkpoints every 2 steps in the candidate arm, giving ~10 non-persistent samples.
* `logger.timing_log_level: 0` or higher, so Megatron logs its timers.
* ML Flashpoint logging at `INFO`.

A run of this length produces single-digit sample counts for the durable timer. Report the max alongside the mean, and
do not read a small mean difference as significant.

## Arm A — baseline

Run the workload unchanged. For a Megatron Bridge run, leave `custom_manager_class` unset. For a NeMo RL run, leave
`MLFLASHPOINT_NEMO_RL_ENABLED` unset, so `install_from_env` is a no-op and the worker behaves exactly as upstream.

Capture stdout from every rank; rank 0 carries the timer lines.

```bash
kubectl logs -f job/<baseline-job> --all-containers --prefix > logs/baseline.log
```

## Arm B — ML Flashpoint

Same job, same everything, with the adapter enabled.

Megatron Bridge:

```yaml
checkpoint:
  save: /gcs/my-run/checkpoints
  save_interval: 10
  non_persistent_save_interval: 2
  non_persistent_ckpt_type: local
  custom_manager_class: ml_flashpoint.adapter.megatron_bridge.MLFlashpointBridgeCheckpointManager
```

NeMo RL, as environment variables on the worker pods:

```bash
MLFLASHPOINT_NEMO_RL_ENABLED=true
MLFLASHPOINT_NEMO_RL_MODE=replace
MLFLASHPOINT_NEMO_RL_DURABLE_EVERY_N_SAVES=5
MLFLASHPOINT_BASE_CONTAINER=/dev/shm/ml_flashpoint/${JOB_ID}
```

`replace` is the mode that shows the difference: it lets most checkpoints go to memory alone. `augment` keeps every
durable write and therefore cannot make the loop faster — use it to check correctness, not to measure a speedup.

```bash
kubectl logs -f job/<flashpoint-job> --all-containers --prefix > logs/flashpoint.log
```

## Comparing

```bash
scripts/benchmarks/parse_checkpoint_timings.py \
    --label baseline logs/baseline.log --output baseline.json

scripts/benchmarks/parse_checkpoint_timings.py \
    --label flashpoint logs/flashpoint.log --output flashpoint.json

scripts/benchmarks/compare_checkpoint_timings.py \
    --baseline baseline.json --candidate flashpoint.json
```

Which prints one row per timer name, with the sample count, the mean in each arm, and the delta as an absolute change,
a percentage and a speedup:

```
Checkpoint timing: flashpoint vs baseline

timer                                           n      baseline    flashpoint  mean delta
----------------------------------------------------------------------------------------------------
save-checkpoint                                 3       18.306s       18.402s  +0.096s (+0.5%, 0.99x)
save-checkpoint-non-persistent                 10             -        0.621s  n/a
```

Add `--json` for a machine-readable form.

## Reading the result

Read the table **down the rows, then across the arms** — and be aware that the tool's own `mean delta` column does not
show the headline number, for the reason below.

**Row 1, `save-checkpoint`, is the control.** 18.306 s vs 18.402 s: durable checkpoints cost the same in both arms.
That is the expected and desired result. The adapter does not touch the durable path, so any real difference here means
the two runs differed in something else — a different node pool, a cold storage cache, contention — and everything else
in the table should be distrusted until that is explained.

**Row 2, `save-checkpoint-non-persistent`, is the new work.** It exists only in the ML Flashpoint arm, because the
baseline has no non-persistent cadence at all. Hence `-` for baseline and `n/a` for the delta: the tool compares
like-named timers across arms, and there is nothing to subtract from.

**The headline number is the cross-row comparison the tool cannot compute for you:**

```
baseline  save-checkpoint                 18.306 s   <- what a checkpoint used to cost
flashpoint save-checkpoint-non-persistent  0.621 s   <- what the substituted checkpoint costs now
                                          ---------
                                          ~29x faster, 17.7 s off each substituted checkpoint
```

That is the claim: on the steps where ML Flashpoint now holds the checkpoint, the loop stalls for ~0.6 s instead of
~18 s. It is only a real saving in `replace` mode, where those steps genuinely skip the durable write. In `augment`
mode the durable write still happens, so row 2 is pure added cost — correct, but not faster.

Two further checks before treating the integration as validated:

* **Sample counts are small.** Three durable samples per arm is enough to spot an order-of-magnitude difference, not a
  5% one. Read `max_s` in the JSON alongside the mean.
* **A fast checkpoint is not the whole claim — recovery has to work.** Confirm the ML Flashpoint arm logs
  `Recovered from ML Flashpoint checkpoint` after a deliberate restart.

## Results

Fill this in from your own run.

| Timer | Baseline mean | ML Flashpoint mean | Delta |
|---|---|---|---|
| `save-checkpoint` | | | |
| `save-checkpoint-non-persistent` | n/a | | |
| `nemo_rl.save_checkpoint` | | | |

Record alongside it: node count, GPUs per node, model, parallelism, per-rank checkpoint size, durable destination, and
the commit of each repository involved.
