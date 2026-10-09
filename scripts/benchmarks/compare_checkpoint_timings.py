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

"""Compares two checkpoint-timing reports and prints the delta.

Consumes the JSON produced by ``parse_checkpoint_timings.py``::

    compare_checkpoint_timings.py --baseline baseline.json --candidate flashpoint.json

The headline number is the mean wall-clock cost of ``save-checkpoint`` (or
``nemo_rl.save_checkpoint`` for a NeMo RL run), because that is the part of the
step the training loop blocks on.
"""

import argparse
import json
import sys
from typing import Any, Optional

HEADLINE_TIMERS = (
    "save-checkpoint",
    "save-checkpoint-non-persistent",
    "nemo_rl.save_checkpoint",
)
"""Timers reported first, in this order, when present."""


def _load(path: str) -> dict[str, Any]:
    """Reads a timing report.

    Args:
        path: Path to a report produced by ``parse_checkpoint_timings.py``.

    Returns:
        The parsed report.
    """
    with open(path, "r") as handle:
        return json.load(handle)


def _fmt_delta(baseline: Optional[float], candidate: Optional[float]) -> str:
    """Formats the change between two measurements.

    Args:
        baseline: The baseline value, or None when the timer is absent.
        candidate: The candidate value, or None when the timer is absent.

    Returns:
        A human-readable delta.
    """
    if baseline is None or candidate is None:
        return "n/a"
    if baseline == 0:
        return "n/a (baseline is 0)"
    delta = candidate - baseline
    pct = delta / baseline * 100.0
    speedup = baseline / candidate if candidate > 0 else float("inf")
    return f"{delta:+.3f}s ({pct:+.1f}%, {speedup:.2f}x)"


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Builds a per-timer comparison.

    Args:
        baseline: The baseline report.
        candidate: The candidate report.

    Returns:
        One row per timer seen in either report, headline timers first.
    """
    base_timers = baseline.get("timers", {})
    cand_timers = candidate.get("timers", {})
    names = sorted(set(base_timers) | set(cand_timers))
    names.sort(key=lambda n: (HEADLINE_TIMERS.index(n) if n in HEADLINE_TIMERS else len(HEADLINE_TIMERS), n))

    rows = []
    for name in names:
        base = base_timers.get(name)
        cand = cand_timers.get(name)
        rows.append(
            {
                "timer": name,
                "baseline_mean_s": base["mean_s"] if base else None,
                "candidate_mean_s": cand["mean_s"] if cand else None,
                "baseline_max_s": base["max_s"] if base else None,
                "candidate_max_s": cand["max_s"] if cand else None,
                "baseline_count": base["count"] if base else 0,
                "candidate_count": cand["count"] if cand else 0,
                "mean_delta": _fmt_delta(base["mean_s"] if base else None, cand["mean_s"] if cand else None),
                "max_delta": _fmt_delta(base["max_s"] if base else None, cand["max_s"] if cand else None),
            }
        )
    return rows


def render(baseline_label: str, candidate_label: str, rows: list[dict[str, Any]]) -> str:
    """Renders the comparison as a plain-text table.

    Args:
        baseline_label: Name of the baseline arm.
        candidate_label: Name of the candidate arm.
        rows: Rows from :func:`compare`.

    Returns:
        The rendered table.
    """
    lines = [
        f"Checkpoint timing: {candidate_label} vs {baseline_label}",
        "",
        f"{'timer':<44} {'n':>4} {baseline_label[:12]:>13} {candidate_label[:12]:>13}  {'mean delta':<28}",
        "-" * 100,
    ]
    for row in rows:
        base = "-" if row["baseline_mean_s"] is None else f"{row['baseline_mean_s']:.3f}s"
        cand = "-" if row["candidate_mean_s"] is None else f"{row['candidate_mean_s']:.3f}s"
        count = max(row["baseline_count"], row["candidate_count"])
        lines.append(f"{row['timer']:<44} {count:>4} {base:>13} {cand:>13}  {row['mean_delta']:<28}")
    lines.append("")
    lines.append("Means are per checkpoint, over the samples found in the logs. A run of only a few steps")
    lines.append("produces few samples, so read the max column alongside the mean before concluding.")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Command line arguments, excluding the program name.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="Baseline report JSON.")
    parser.add_argument("--candidate", required=True, help="Candidate report JSON.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args(argv)

    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    rows = compare(baseline, candidate)

    if args.json:
        print(
            json.dumps(
                {
                    "baseline_label": baseline.get("label", "baseline"),
                    "candidate_label": candidate.get("label", "candidate"),
                    "rows": rows,
                },
                indent=2,
            )
        )
    else:
        print(render(baseline.get("label", "baseline"), candidate.get("label", "candidate"), rows))

    if not rows:
        print("No timers in either report.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
