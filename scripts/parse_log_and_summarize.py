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

"""
To use this script:
1) Copy it to your cluster or training environment (anywhere with access to the log file to analyze).
2) Ensure numpy is installed in your venv (and activate your venv):
    ```bash
    source .venv/bin/activate
    pip install numpy
    ```
2) Run it against a log file, redirecting output to some txt file for consumption:
    ```bash
    # In case it is not already executable
    chmod a+rx parse_log_and_summarize.py
    ./parse_log_and_summarize.py ./path/to/job-1458/log.out > job-1458-summary.txt
    ```
"""

import argparse
import glob
import json
import os.path
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np


def find_instrumented_functions(source_dir):
    """Finds all functions decorated with @log_execution_time."""
    functions = []
    for filepath in glob.glob(f"{source_dir}/**/*.py", recursive=True):
        filename = os.path.basename(filepath).replace(".py", "")
        with open(filepath, "r") as f:
            content = f.read()
            matches = re.findall(r'@log_execution_time\(.*?name="([^"]+)"', content, re.DOTALL)
            for match in matches:
                functions.append(f"{filename}.{match}")
    return functions


def parse_log_file(log_file):
    """Parses the log file and extracts performance data including ranks."""
    # Standard performance log: "... took X.Xs"
    log_pattern = re.compile(r"\[MLF.* Step=(-?\d+) Rank=(-?[\d/]+) (.*?):[\d]+\] (.*?) took ([\d.]+)s")

    # Format: [MLF YYYY-MM-DD HH:MM:SS,mmm ...]
    timestamp_prefix = r"\[MLF (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"

    # Throughput logs
    # Read/Write patterns to capture timestamp and bytes
    # Capture groups: 1: Timestamp, 2: Step, 3: Rank, 4: Bytes, 5: Duration
    # Timestamp + Read 123 bytes in 0.123 s (x.xx GB/s) from 1 buckets
    read_throughput_pattern = re.compile(
        timestamp_prefix + r".*? Step=(-?\d+) Rank=(-?[\d/]+) .*?\] Read (\d+) "
        r"bytes in ([\d.]+) s \(([\d.]+) GB/s\) from (\d+) files"
    )
    # Timestamp + Written 123 bytes in 0.123 s (X.XX GB/s) from 1 buckets
    write_throughput_pattern = re.compile(
        timestamp_prefix + r".*? Step=(-?\d+) Rank=(-?[\d/]+) .*?\] Written (\d+) "
        r"bytes in ([\d.]+) s \(([\d.]+) GB/s\) from (\d+) buckets"
    )

    # Dictionary to store raw records for totalthroughput calculation
    # Format: { 'Read': { step: [ (end_time, duration, bytes, rank), ... ] }, 'Write': { ... } }
    raw_throughput_records = {"Read": defaultdict(list), "Write": defaultdict(list)}

    train_step_pattern = re.compile(r"global_step: (\d+).*?train_step_timing in s: ([\d.]+)")
    data = defaultdict(lambda: defaultdict(list))
    ordered_functions = []

    with open(log_file, "r") as f:
        for line in f:
            # Check for standard "took" logs
            match = log_pattern.search(line)
            if match:
                step, rank, logger_name, func_name, time_taken = match.groups()
                logger_name = logger_name.split(".")[-1]
                metric_name = f"{logger_name}.{func_name}"
                if metric_name not in ordered_functions:
                    ordered_functions.append(metric_name)
                data[metric_name][int(step)].append((float(time_taken), rank))
                continue

            # Check for Read Throughput
            read_match = read_throughput_pattern.search(line)
            if read_match:
                timestamp, step, rank, bytes_val, duration, _, _ = read_match.groups()
                end_time = datetime.strptime(timestamp.replace(",", "."), "%Y-%m-%d %H:%M:%S.%f")
                raw_throughput_records["Read"][int(step)].append(
                    {"end_time": end_time, "duration": float(duration), "bytes": int(bytes_val), "rank": rank}
                )
                continue

            # Check for Write Throughput
            write_match = write_throughput_pattern.search(line)
            if write_match:
                timestamp, step, rank, bytes_val, duration, _, _ = write_match.groups()
                end_time = datetime.strptime(timestamp.replace(",", "."), "%Y-%m-%d %H:%M:%S.%f")
                raw_throughput_records["Write"][int(step)].append(
                    {"end_time": end_time, "duration": float(duration), "bytes": int(bytes_val), "rank": rank}
                )
                continue

            # Check for Train Step
            train_match = train_step_pattern.search(line)
            if train_match:
                step, time_taken = train_match.groups()
                metric_name = "train_step_timing"
                if metric_name not in ordered_functions:
                    ordered_functions.append(metric_name)
                data[metric_name][int(step)].append((float(time_taken), "NA"))

    return data, ordered_functions, raw_throughput_records


def calculate_total_throughput(records):
    """
    Calculates Total Throughput = Total Bytes / (Latest End - Earliest Start).
    """
    node_stats = {}
    for step, entries in records.items():
        if not entries:
            continue

        total_bytes = sum(e["bytes"] for e in entries)

        # Calculate individual rank start/end times
        starts = [e["end_time"] - timedelta(seconds=e["duration"]) for e in entries]
        ends = [e["end_time"] for e in entries]

        # Determine the total read/write duration across all ranks
        duration = (max(ends) - min(starts)).total_seconds()

        if duration > 0:
            throughput = (total_bytes / 1e9) / duration
            node_stats[step] = {"throughput": throughput, "duration": duration, "total_gb": total_bytes / 1e9}
    return node_stats


# Function to calculate per-node throughput based on Rank to Node mapping
def calculate_per_node_throughput(records, ranks_per_node):
    """
    Calculates Per-Node Throughput = Node Total Bytes / (Latest Node End - Earliest Node Start).
    """
    per_node_stats = defaultdict(lambda: defaultdict(dict))
    for step, entries in records.items():
        # Group entries by node ID
        node_groups = defaultdict(list)
        for e in entries:
            try:
                node_id = int(e["rank"]) // ranks_per_node
                node_groups[node_id].append(e)
            except (ValueError, IndexError):
                continue

        for node_id, node_entries in node_groups.items():
            total_bytes = sum(ne["bytes"] for ne in node_entries)
            starts = [ne["end_time"] - timedelta(seconds=ne["duration"]) for ne in node_entries]
            ends = [ne["end_time"] for ne in node_entries]

            duration = (max(ends) - min(starts)).total_seconds()
            if duration > 0:
                per_node_stats[step][node_id] = {
                    "throughput": (total_bytes / 1e9) / duration,
                    "duration": duration,
                    "total_gb": total_bytes / 1e9,
                }
    return per_node_stats


def analyze_step_time_breakdown(log_file_path):
    """
    Analyzes the wall-clock time gap between consecutive global steps to identify overheads.

    This function calculates:
    1. Total Gap: Wall-clock time elapsed between the end of step N-1 and step N.
    2. Other Time (Overhead): Total Gap minus the reported training time (train_step_timing from NeMo logs).
    """
    timestamp_pattern = re.compile(r"\[NeMo \w (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    train_step_pattern = re.compile(r"global_step: (\d+).*?train_step_timing in s: ([\d.]+)")

    step_data = []
    last_seen_timestamp = None

    try:
        with open(log_file_path, "r") as f:
            for line in f:
                ts_match = timestamp_pattern.search(line)
                if ts_match:
                    try:
                        last_seen_timestamp = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        pass

                step_match = train_step_pattern.search(line)
                if step_match and last_seen_timestamp:
                    step_data.append(
                        {
                            "step": int(step_match.group(1)),
                            "finish_time": last_seen_timestamp,
                            "train_time": float(step_match.group(2)),
                        }
                    )
    except Exception as e:
        print(f"Error analyzing breakdown: {e}")
        return []

    results = []
    for i in range(1, len(step_data)):
        prev, curr = step_data[i - 1], step_data[i]
        if curr["step"] > prev["step"]:
            time_delta = max((curr["finish_time"] - prev["finish_time"]).total_seconds(), 0.0)
            other_time = time_delta - curr["train_time"]
            results.append(
                {
                    "step": curr["step"],
                    "timestamp": curr["finish_time"],
                    "total_gap": time_delta,
                    "train_time": curr["train_time"],
                    "other_time": other_time,
                }
            )
    return results


def calculate_statistics(data):
    """Calculates statistics for the collected data, including ranks."""
    stats = defaultdict(dict)
    for function_name, steps in data.items():
        for step, time_rank_pairs in steps.items():
            times = [pair[0] for pair in time_rank_pairs]
            ranks = [int(pair[1]) if pair[1] != "NA" else -1 for pair in time_rank_pairs]
            stats[function_name][step] = {
                "min": np.min(times),
                "max": np.max(times),
                "avg": np.mean(times),
                "p50": np.percentile(times, 50),
                "p75": np.percentile(times, 75),
                "p90": np.percentile(times, 90),
                "p95": np.percentile(times, 95),
                "p99": np.percentile(times, 99),
                "ranks": sorted(ranks),
            }
    return stats


def calculate_overall_statistics(data):
    """Calculates overall statistics for each function across all steps."""
    overall_stats = defaultdict(dict)
    for function_name, steps in data.items():
        all_times = []
        all_ranks = []
        all_steps_max = []
        all_steps_min = []
        for step, time_rank_pairs in steps.items():
            all_times.extend([pair[0] for pair in time_rank_pairs])
            all_ranks.extend([int(pair[1]) if pair[1] != "NA" else -1 for pair in time_rank_pairs])
            all_steps_max.append(max(pair[0] for pair in time_rank_pairs))
            all_steps_min.append(min(pair[0] for pair in time_rank_pairs))

        if all_times:
            unique_ranks = sorted(list(set(all_ranks)))
            overall_stats[function_name] = {
                "min": np.min(all_times),
                "max": np.max(all_times),
                "avg": np.mean(all_times),
                "p50": np.percentile(all_times, 50),
                "p75": np.percentile(all_times, 75),
                "p90": np.percentile(all_times, 90),
                "p95": np.percentile(all_times, 95),
                "p99": np.percentile(all_times, 99),
                "ranks": unique_ranks,
            }

        if all_steps_max:
            overall_stats[function_name]["step_max"] = {
                "num_steps": len(all_steps_max),
                "min": np.min(all_steps_max),
                "max": np.max(all_steps_max),
                "avg": np.mean(all_steps_max),
                "p50": np.percentile(all_steps_max, 50),
                "p75": np.percentile(all_steps_max, 75),
                "p90": np.percentile(all_steps_max, 90),
                "p95": np.percentile(all_steps_max, 95),
                "p99": np.percentile(all_steps_max, 99),
            }
        if all_steps_min:
            overall_stats[function_name]["step_min"] = {
                "num_steps": len(all_steps_min),
                "min": np.min(all_steps_min),
                "max": np.max(all_steps_min),
                "avg": np.mean(all_steps_min),
                "p50": np.percentile(all_steps_min, 50),
                "p75": np.percentile(all_steps_min, 75),
                "p90": np.percentile(all_steps_min, 90),
                "p95": np.percentile(all_steps_min, 95),
                "p99": np.percentile(all_steps_min, 99),
            }
    return overall_stats


def calculate_total_training_time(log_file_path):
    """Calculates the total training time (summation of train_step_timing) from the log file."""
    total_training_time = 0.0
    time_pattern = re.compile(r"train_step_timing in s: (\d+\.?\d*)")
    try:
        with open(log_file_path, "r") as f:
            for line in f:
                match = time_pattern.search(line)
                if match:
                    total_training_time += float(match.group(1))
    except Exception:
        return None
    return total_training_time


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Parse and summarize ML Flashpoint performance logs.",
        epilog="""
Example Usage:

1. To generate a snapshot of all instrumented functions:
   python parse_log_and_summarize.py --save-functions instrumented_functions.json

2. To parse a log file and print the summary:
   python parse_log_and_summarize.py /path/to/your/logfile.log
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("log_file", nargs="?", default=None, help="Path to the log file to parse.")
    parser.add_argument("--src-dir", default="src", help="Source directory to scan for instrumented functions.")
    parser.add_argument("--save-functions", help="Path to save the instrumented functions list as a JSON file.")
    parser.add_argument("--ranks-per-node", type=int, default=8, help="Number of ranks per node.")

    args = parser.parse_args()

    if args.save_functions:
        functions = find_instrumented_functions(args.src_dir)
        with open(args.save_functions, "w") as f:
            json.dump(functions, f, indent=2)
        print(f"Instrumented functions saved to {args.save_functions}")
        return

    if not args.log_file:
        parser.error("the following arguments are required: log_file")

    print("Analyzing: ", args.log_file)
    print("Generated at: ", time.asctime())
    print()
    print("*********" * 8)
    print()

    data, ordered_functions, raw_throughput_records = parse_log_file(args.log_file)
    stats = calculate_statistics(data)
    overall_stats = calculate_overall_statistics(data)

    print("--- Overall Summary ---")
    for function_name in ordered_functions:
        if function_name in overall_stats:
            overall = overall_stats[function_name]
            ranks_str = ",".join(map(str, overall["ranks"]))
            suffix = "s" if "Throughput" not in function_name else ""
            print(
                f"  {function_name: <70} | "
                f"min: {overall['min']:.4f}{suffix} | "
                f"max: {overall['max']:.4f}{suffix} | "
                f"avg: {overall['avg']:.4f}{suffix} | "
                f"p50: {overall['p50']:.4f}{suffix} | "
                f"p75: {overall['p75']:.4f}{suffix} | "
                f"p90: {overall['p90']:.4f}{suffix} | "
                f"p95: {overall['p95']:.4f}{suffix} | "
                f"p99: {overall['p99']:.4f}{suffix} | "
                f"ranks: [{ranks_str}]"
            )
            if "step_max" in overall:
                sm = overall["step_max"]
                print(
                    f"    Step Max ({sm['num_steps']} steps) | "
                    f"min: {sm['min']:.4f}{suffix} | "
                    f"max: {sm['max']:.4f}{suffix} | "
                    f"avg: {sm['avg']:.4f}{suffix} | "
                    f"p50: {sm['p50']:.4f}{suffix} | "
                    f"p99: {sm['p99']:.4f}{suffix}"
                )
            if "step_min" in overall:
                sm = overall["step_min"]
                print(
                    f"    Step Min ({sm['num_steps']} steps) | "
                    f"min: {sm['min']:.4f}{suffix} | "
                    f"max: {sm['max']:.4f}{suffix} | "
                    f"avg: {sm['avg']:.4f}{suffix} | "
                    f"p50: {sm['p50']:.4f}{suffix} | "
                    f"p99: {sm['p99']:.4f}{suffix}"
                )

    print("\n" + "=" * 150 + "\n")

    print("--- Per-Step Breakdown ---")
    for function_name in ordered_functions:
        if function_name in stats:
            print(f"--- Function: {function_name} ---")
            for step in sorted(stats[function_name].keys()):
                step_stats = stats[function_name][step]
                ranks_str = ",".join(map(str, step_stats["ranks"]))
                suffix = "s" if "Throughput" not in function_name else ""
                print(f" Step {step: <6}:")
                print(
                    f"    min: {step_stats['min']:.4f}{suffix} | "
                    f"max: {step_stats['max']:.4f}{suffix} | "
                    f"avg: {step_stats['avg']:.4f}{suffix} | "
                    f"p50: {step_stats['p50']:.4f}{suffix} | "
                    f"p75: {step_stats['p75']:.4f}{suffix} | "
                    f"p90: {step_stats['p90']:.4f}{suffix} | "
                    f"p95: {step_stats['p95']:.4f}{suffix} | "
                    f"p99: {step_stats['p99']:.4f}{suffix} | "
                    f"ranks: [{ranks_str}]"
                )
            print()

    # Calculate statistics for Read and Write
    cluster_read_stats = calculate_total_throughput(raw_throughput_records["Read"])
    cluster_write_stats = calculate_total_throughput(raw_throughput_records["Write"])

    for mode, throughput_stats in [("Read", cluster_read_stats), ("Write", cluster_write_stats)]:
        if throughput_stats:
            print(f"--- Total {mode} Throughput ---")
            print(f"{'Step':<8} | {'Total Data (GB)':<15} | {'Span (s)':<10} | {'Throughput (GB/s)':<20}")
            print("-" * 65)
            all_throughput = []
            for step in sorted(throughput_stats.keys()):
                s = throughput_stats[step]
                all_throughput.append(s["throughput"])
                print(f"{step:<8} | {s['total_gb']:<15.2f} | {s['duration']:<10.3f} | {s['throughput']:<20.4f}")

            if len(all_throughput) > 1:
                # Skip the first read/write record
                avg_throughput = np.mean(all_throughput[1:])
                print(f"Total {mode} Throughput (Excluding first step): {avg_throughput:.4f} GB/s\n")
            elif all_throughput:
                # Fallback if only one step is recorded
                print(f"Total {mode} Throughput: {np.mean(all_throughput):.4f} GB/s\n")

    for mode, raw_records in [("Read", raw_throughput_records["Read"]), ("Write", raw_throughput_records["Write"])]:
        per_node_stats = calculate_per_node_throughput(raw_records, args.ranks_per_node)
        if per_node_stats:
            print(f"--- Per-Node {mode} Throughput (Ranks per Node: {args.ranks_per_node}) ---")
            print(f"{'Step':<8} | {'Node':<6} | {'Total Data (GB)':<15} | {'Span (s)':<10} | {'Throughput (GB/s)':<20}")
            print("-" * 75)

            node_averages = defaultdict(list)
            sorted_steps = sorted(per_node_stats.keys())

            for step in sorted_steps:
                for node_id in sorted(per_node_stats[step].keys()):
                    s = per_node_stats[step][node_id]
                    print(
                        f"{step:<8} | {node_id:<6} | {s['total_gb']:<15.2f} | {s['duration']:<10.3f}"
                        f"| {s['throughput']:<20.4f}"
                    )
                    if len(sorted_steps) > 1 and step != sorted_steps[0]:
                        node_averages[node_id].append(s["throughput"])

            if node_averages:
                print(f"\n Per-Node {mode} Throughput (Excluding first step):")
                for node_id in sorted(node_averages.keys()):
                    print(f"  Node {node_id}: {np.mean(node_averages[node_id]):.4f} GB/s")
            print()

    print("--- Step-to-Step Time Gap Analysis ---")
    print("Note: 'Total Gap' is the wall-clock time elapsed since the previous step finished.")
    print("      'Other Time' = Total Gap - Train Time.")
    print(
        f"{'Step':<8} | {'Timestamp (Approx)':<20} | {'Total Gap (s)':<15} | "
        f"{'Train Time (s)':<15} | {'Other Time (s)':<15}"
    )
    print("-" * 85)

    breakdown = analyze_step_time_breakdown(args.log_file)
    other_times = []
    if not breakdown:
        print("No consecutive steps or timestamps found to calculate gaps.")
    else:
        for row in breakdown:
            other_times.append(row["other_time"])
            print(
                f"{row['step']:<8} | "
                f"{str(row['timestamp']):<20} | "
                f"{row['total_gap']:<15.2f} | "
                f"{row['train_time']:<15.2f} | "
                f"{row['other_time']:<15.2f}"
            )
        print("-" * 85)
        if other_times:
            print(f"Average 'Other Time' (Overhead) per step: {np.mean(other_times):.4f}s")
            print(f"Total accumulated 'Other Time': {np.sum(other_times):.2f}s")

    print("\n" + "=" * 150 + "\n")

    total_time = calculate_total_training_time(args.log_file)
    if total_time is not None:
        print(f"Total time used in training: {total_time:.2f} seconds ({total_time / 3600:.2f} hours)")

    print("\n" + "=" * 150 + "\n")


if __name__ == "__main__":
    main()
