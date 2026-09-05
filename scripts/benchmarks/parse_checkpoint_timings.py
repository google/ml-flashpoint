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

"""Extracts checkpoint timings from Megatron Bridge / NeMo RL training logs.

Three sources are recognized, in decreasing order of preference:

``megatron-timer``
    Megatron's own timer line, emitted by ``save_checkpoint_and_time``. It brackets
    the save with barriers, so it is the number that reflects what the training
    loop actually paid::

        save-checkpoint ................................: (1234.56, 1234.56)

``mlf-timer``
    ML Flashpoint's ``log_execution_time`` output, which isolates the adapter's
    own cost::

        MLFlashpointBridgeCheckpointManager.save took 0.1234s

``nemo-rl-timer``
    NeMo RL's wrapper around the worker save, installed by the adapter::

        nemo_rl.save_checkpoint took 12.3456s

Usage::

    parse_checkpoint_timings.py --label baseline logs/baseline/*.log > baseline.json
    parse_checkpoint_timings.py --label flashpoint logs/mlf/*.log > flashpoint.json
"""

import argparse
import json
import re
import statistics
import sys
from typing import Iterable, Optional

# "save-checkpoint ....: (1234.56, 1234.56)" -- Megatron reports (min, max) in ms
# across ranks; the max is the one the slowest rank paid.
_MEGATRON_TIMER = re.compile(
    r"(?P<name>save-checkpoint(?:-non-persistent)?|load-checkpoint)\s*\.*\s*:\s*"
    r"\(\s*(?P<min>[0-9.]+)\s*,\s*(?P<max>[0-9.]+)\s*\)"
)

# ml_flashpoint.core.utils.log_execution_time: "<name> took 1.2345s"
_MLF_TIMER = re.compile(r"(?P<name>[A-Za-z_][\w.]*) took (?P<seconds>[0-9.]+)s")

_MLF_NAMES_OF_INTEREST = (
    "MLFlashpointBridgeCheckpointManager.save",
    "MLFlashpointBridgeCheckpointManager.mlf_save",
    "MLFlashpointBridgeCheckpointManager.load",
    "MLFlashpointBridgeCheckpointManager.mlf_load",
    "MLFlashpointBridgeCheckpointManager.finalize_async_saves",
    "MLFlashpointNeMoRLCheckpointer.save",
    "nemo_rl.save_checkpoint",
    "async_save",
)


def parse_lines(lines: Iterable[str]) -> dict[str, list[float]]:
    """Collects every recognized timing from a stream of log lines.

    Args:
        lines: The log lines to scan.

    Returns:
        A mapping of timer name to the observed durations, in seconds.
    """
    samples: dict[str, list[float]] = {}
    for line in lines:
        match = _MEGATRON_TIMER.search(line)
        if match:
            # Megatron timers are reported in milliseconds.
            samples.setdefault(match.group("name"), []).append(float(match.group("max")) / 1000.0)
            continue
        match = _MLF_TIMER.search(line)
        if match and match.group("name") in _MLF_NAMES_OF_INTEREST:
            samples.setdefault(match.group("name"), []).append(float(match.group("seconds")))
    return samples


def summarize(samples: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Reduces raw samples to per-timer statistics.

    Args:
        samples: Timer name to durations in seconds.

    Returns:
        Timer name to a statistics dictionary.
    """
    summary = {}
    for name, values in sorted(samples.items()):
        ordered = sorted(values)
        summary[name] = {
            "count": len(ordered),
            "total_s": sum(ordered),
            "mean_s": statistics.fmean(ordered),
            "median_s": statistics.median(ordered),
            "min_s": ordered[0],
            "max_s": ordered[-1],
            # With a handful of steps there are too few samples for a real p95,
            # so report the worst observed value alongside the mean instead.
            "stdev_s": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        }
    return summary


def _read(paths: list[str]) -> Iterable[str]:
    """Yields lines from the given files, or stdin when no file is given.

    Args:
        paths: Log file paths.

    Yields:
        Individual log lines.
    """
    if not paths:
        yield from sys.stdin
        return
    for path in paths:
        with open(path, "r", errors="replace") as handle:
            yield from handle


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Command line arguments, excluding the program name.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="*", help="Log files to parse. Reads stdin when omitted.")
    parser.add_argument("--label", required=True, help="Name for this arm of the experiment, e.g. 'baseline'.")
    parser.add_argument("--output", help="Write JSON here instead of stdout.")
    args = parser.parse_args(argv)

    samples = parse_lines(_read(args.logs))
    if not samples:
        print(
            "No checkpoint timings found. Confirm the run had logger.timing_log_level >= 0 and that "
            "ML Flashpoint logging is at INFO.",
            file=sys.stderr,
        )
    report = {"label": args.label, "timers": summarize(samples), "raw_samples": samples}

    serialized = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(serialized + "\n")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
