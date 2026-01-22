#!/usr/bin/env python3
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

import argparse
import re
import time
from collections import defaultdict

import parse_log_and_summarize as base


def parse_log_file(log_file):
    """Parses the NeMo log file and extracts performance data including ranks."""
    step_pattern = re.compile(r"global_step: (\d+)")

    log_patterns = [
        (
            re.compile(
                r"\[NeMo D .*? (nemo_logging):\d+ rank:(\d+)\] "
                r"AsyncFinalizableCheckpointIO\.maybe_finalize_save_checkpoint took ([\d.eE+-]+)s"
            ),
            "maybe_finalize_save_checkpoint",
        ),
        (
            re.compile(r"\[NeMo [DI] .*? (fully_parallel):\d+ rank:(\d+)\] parallel save sharding, time: ([\d.eE+-]+)"),
            "parallel save sharding",
        ),
        (re.compile(r"\[NeMo [DI] .*? (state_dict_saver):\d+ rank:(\d+)\] .*?, plan time: ([\d.eE+-]+)"), "plan time"),
        (
            re.compile(
                r"\[NeMo D .*? (filesystem_async):\d+ rank:(\d+)\] "
                r"FileSystemWriterAsync\.prepare_decentralized_global_plan took ([\d.eE+-]+)s"
            ),
            "prepare_decentralized_global_plan",
        ),
        (
            re.compile(r"\[NeMo [DI] .*? (filesystem_async):\d+ rank:(\d+)\] bucket_prep, time: ([\d.eE+-]+)"),
            "bucket_prep",
        ),
        (
            re.compile(r"\[NeMo [DI] .*? (filesystem_async):\d+ rank:(\d+)\] D2H and push, time: ([\d.eE+-]+)"),
            "D2H and push",
        ),
        (
            re.compile(
                r"\[NeMo I .*? (nemo_logging):\d+ rank:(\d+)\] Global Checkpoint Save :.*? "
                r"Save duration: ([\d.eE+-]+)s"
            ),
            "Global Checkpoint Save",
        ),
        (
            re.compile(
                r"\[NeMo D .*? (filesystem_async):\d+ rank:(\d+)\] "
                r"FileSystemWriterAsync\.preload_tensors took ([\d.eE+-]+)s"
            ),
            "preload_tensors",
        ),
        (re.compile(r"\[NeMo D .*? (async_utils):\d+ rank:(\d+)\].*? takes ([\d.eE+-]+) to finish D2H"), "finish D2H"),
        (
            re.compile(r"\[NeMo D .*? (async_utils):\d+ rank:(\d+)\].*? takes ([\d.eE+-]+) to schedule async ckpt"),
            "schedule async ckpt",
        ),
        (
            re.compile(
                r"\[NeMo D .*? (nemo_logging):\d+ rank:(\d+)\] "
                r"AsyncFinalizableCheckpointIO\.save_checkpoint took ([\d.eE+-]+)s"
            ),
            "save_checkpoint",
        ),
        (
            re.compile(
                r"\[NeMo D .*? (filesystem_async):\d+ proc:(\d+)\] "
                r"FileSystemWriterAsync\.write_preloaded_data took ([\d.eE+-]+)s"
            ),
            "write_preloaded_data",
        ),
        (
            re.compile(r"\[NeMo [DI] .*? (filesystem_async):\d+ rank:(\d+)\] .*? write\(sync,parallel\): ([\d.eE+-]+)"),
            "write sync parallel",
        ),
        (
            re.compile(
                r"\[NeMo D .*? (filesystem_async):\d+ rank:(\d+)\] "
                r"FileSystemWriterAsync\.write_preloaded_data_multiproc took ([\d.eE+-]+)s"
            ),
            "write_preloaded_data_multiproc",
        ),
        (
            re.compile(
                r"\[NeMo D .*? (async_utils):\d+ rank:(\d+)\] "
                r"TemporalAsyncCaller: Async process join finished after ([\d.eE+-]+)s from forking"
            ),
            "async process join",
        ),
        (re.compile(r"\[NeMo D .*? (state_dict_saver):\d+ rank:(\d+)\].*?, gather: ([\d.eE+-]+)"), "gather"),
        (
            re.compile(r"\[NeMo D .*? (state_dict_saver):\d+ rank:(\d+)\].*?, metadata_write: ([\d.eE+-]+)"),
            "metadata_write",
        ),
        (
            re.compile(
                r"\[NeMo D .*? (filesystem_async):\d+ rank:(\d+)\] FileSystemWriterAsync\.finish took ([\d.eE+-]+)s"
            ),
            "finish",
        ),
        (re.compile(r"\[NeMo D .*? (utils):\d+ rank:(\d+)\] finalize took ([\d.eE+-]+)s"), "finalize"),
        (
            re.compile(r"\[NeMo I .*? (nemo_logging):\d+ rank:(\d+)\] Async finalization time took ([\d.eE+-]+) s"),
            "Async finalization time",
        ),
        (
            re.compile(
                r"\[NeMo D .*? (utils):\d+ rank:(\d+)\] FullyParallelLoadStrategyWrapper\.load\."
                r"self\.apply_loading_parallelization took ([\d.eE+-]+)s"
            ),
            "apply_loading_parallelization_took",
        ),
        (
            re.compile(
                r"\[NeMo D .*? (utils):\d+ rank:(\d+)\] FullyParallelLoadStrategyWrapper\.load\."
                r"base_load_ShardedObjects took ([\d.eE+-]+)s"
            ),
            "base_load_sharded_objects_took",
        ),
        (
            re.compile(
                r"\[NeMo D .*? (utils):\d+ rank:(\d+)\] FullyParallelLoadStrategyWrapper\.load\."
                r"base_load_ShardedTensors took ([\d.eE+-]+)s"
            ),
            "base_load_sharded_tensors_took",
        ),
        (
            re.compile(
                r"\[NeMo D .*? (utils):\d+ rank:(\d+)\] FullyParallelLoadStrategyWrapper\.load\."
                r"self\.exchange_loaded_tensors took ([\d.eE+-]+)s"
            ),
            "exchange_loaded_tensors_took",
        ),
        (
            re.compile(
                r"\[NeMo I .*? (nemo_logging):\d+ rank:(\d+)\] Global Checkpoint Load : Rank : \d+ : "
                r"Start time : [\d.]+s : Time spent in load_checkpoint: ([\d.eE+-]+)s"
            ),
            "global_checkpoint_load",
        ),
        (
            re.compile(
                r"\[NeMo D .*? (nemo_logging):\d+ rank:(\d+)\] MegatronCheckpointIO\.load_checkpoint took ([\d.eE+-]+)s"
            ),
            "load_checkpoint_took",
        ),
    ]

    data = defaultdict(lambda: defaultdict(list))
    ordered_functions = []
    current_step = 0

    with open(log_file, "r") as f:
        for line in f:
            step_match = step_pattern.search(line)
            if step_match:
                current_step = int(step_match.group(1))

            for pattern, func_name in log_patterns:
                match = pattern.search(line)
                if match:
                    logger_name, rank, time_taken = match.groups()
                    metric_name = f"{logger_name}.{func_name}"
                    if metric_name not in ordered_functions:
                        ordered_functions.append(metric_name)
                    data[metric_name][current_step].append((float(time_taken), rank))
                    break
    return data, ordered_functions


def print_line(name, overall, prefix=""):
    ranks_str = ",".join(map(str, overall["ranks"]))
    print(
        f"{prefix}{name: <70} | "
        f"min: {overall['min']:.4f}s | max: {overall['max']:.4f}s | avg: {overall.get('avg', 0):.4f}s | "
        f"p50: {overall['p50']:.4f}s | p75: {overall['p75']:.4f}s | p90: {overall['p90']:.4f}s | "
        f"p95: {overall['p95']:.4f}s | p99: {overall['p99']:.4f}s | ranks: [{ranks_str}]"
    )

    if "step_max" in overall:
        sm = overall["step_max"]
        print(
            f"    Step Max ({sm['num_steps']} steps) | "
            f"min: {sm['min']:.4f}s | max: {sm['max']:.4f}s | avg: {sm['avg']:.4f}s | "
            f"p50: {sm['p50']:.4f}s | p99: {sm['p99']:.4f}s"
        )

    if "step_min" in overall:
        sm = overall["step_min"]
        print(
            f"    Step Min ({sm['num_steps']} steps) | "
            f"min: {sm['min']:.4f}s | max: {sm['max']:.4f}s | avg: {sm['avg']:.4f}s | "
            f"p50: {sm['p50']:.4f}s | p99: {sm['p99']:.4f}s"
        )


def main():
    parser = argparse.ArgumentParser(description="Parse and summarize NeMo performance logs.")
    parser.add_argument("log_file", help="Path to the nemo log file.")
    args = parser.parse_args()

    print(f"Analyzing: {args.log_file}\nGenerated at: {time.asctime()}\n" + "*" * 72 + "\n")

    data, ordered_functions = parse_log_file(args.log_file)
    stats = base.calculate_statistics(data)
    overall_stats = base.calculate_overall_statistics(data)

    print("--- Overall Summary ---")
    for func in ordered_functions:
        if func in overall_stats:
            print_line(func, overall_stats[func], prefix="  ")

    print("\n" + "=" * 150 + "\n--- Per-Step Breakdown ---")
    for func in ordered_functions:
        if func in stats:
            print(f"--- Function: {func} ---")
            for step in sorted(stats[func].keys()):
                print_line(f" Step {step}:", stats[func][step], prefix="    ")
            print()


if __name__ == "__main__":
    main()
