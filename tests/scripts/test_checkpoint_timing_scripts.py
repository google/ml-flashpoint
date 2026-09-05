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

import importlib.util
import json
from pathlib import Path

import pytest
from assertpy import assert_that

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks"


def _load(name: str):
    """Imports a benchmark script by path, since scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def parser_module():
    return _load("parse_checkpoint_timings")


@pytest.fixture(scope="module")
def compare_module():
    return _load("compare_checkpoint_timings")


BASELINE_LOG = """
 iteration 10/100 | elapsed time per iteration (ms): 812.3
    save-checkpoint ................................: (18450.20, 18512.90)
 iteration 20/100
    save-checkpoint ................................: (17980.00, 18100.10)
"""

FLASHPOINT_LOG = """
    save-checkpoint-non-persistent .................: (612.40, 640.10)
MLFlashpointBridgeCheckpointManager.save took 0.5981s
    save-checkpoint ................................: (18300.00, 18402.10)
    load-checkpoint ................................: (2100.00, 2150.00)
nemo_rl.save_checkpoint took 12.3456s
"""


class TestParseLines:
    def test_reads_the_slowest_rank_in_seconds(self, parser_module):
        # Given/When
        samples = parser_module.parse_lines(BASELINE_LOG.splitlines())

        # Then
        assert_that(samples["save-checkpoint"]).is_length(2)
        assert_that(samples["save-checkpoint"][0]).is_close_to(18.5129, 1e-6)
        assert_that(samples["save-checkpoint"][1]).is_close_to(18.1001, 1e-6)

    def test_recognizes_non_persistent_and_load_timers(self, parser_module):
        # Given/When
        samples = parser_module.parse_lines(FLASHPOINT_LOG.splitlines())

        # Then
        assert_that(samples).contains_key("save-checkpoint-non-persistent", "load-checkpoint")

    def test_recognizes_ml_flashpoint_timers(self, parser_module):
        # Given/When
        samples = parser_module.parse_lines(FLASHPOINT_LOG.splitlines())

        # Then
        assert_that(samples["MLFlashpointBridgeCheckpointManager.save"]).is_equal_to([0.5981])
        assert_that(samples["nemo_rl.save_checkpoint"]).is_equal_to([12.3456])

    def test_ignores_unrelated_timing_lines(self, parser_module):
        # Given
        lines = ["some_unrelated_function took 3.5s", "forward-backward ...: (10.0, 11.0)"]

        # When
        samples = parser_module.parse_lines(lines)

        # Then
        assert_that(samples).is_empty()

    def test_empty_input_yields_no_samples(self, parser_module):
        # Given/When/Then
        assert_that(parser_module.parse_lines([])).is_empty()


class TestSummarize:
    def test_reports_the_expected_statistics(self, parser_module):
        # Given
        samples = {"save-checkpoint": [1.0, 3.0, 2.0]}

        # When
        summary = parser_module.summarize(samples)["save-checkpoint"]

        # Then
        assert_that(summary["count"]).is_equal_to(3)
        assert_that(summary["mean_s"]).is_equal_to(2.0)
        assert_that(summary["median_s"]).is_equal_to(2.0)
        assert_that(summary["min_s"]).is_equal_to(1.0)
        assert_that(summary["max_s"]).is_equal_to(3.0)

    def test_single_sample_has_zero_stdev(self, parser_module):
        # Given/When
        summary = parser_module.summarize({"save-checkpoint": [4.0]})["save-checkpoint"]

        # Then
        assert_that(summary["stdev_s"]).is_equal_to(0.0)


class TestParserCli:
    def test_writes_a_labelled_report(self, parser_module, tmp_path):
        # Given
        log = tmp_path / "run.log"
        log.write_text(BASELINE_LOG)
        out = tmp_path / "report.json"

        # When
        exit_code = parser_module.main(["--label", "baseline", str(log), "--output", str(out)])

        # Then
        assert_that(exit_code).is_equal_to(0)
        report = json.loads(out.read_text())
        assert_that(report["label"]).is_equal_to("baseline")
        assert_that(report["timers"]).contains_key("save-checkpoint")

    def test_reports_nothing_found_without_failing(self, parser_module, tmp_path, capsys):
        # Given
        log = tmp_path / "empty.log"
        log.write_text("nothing interesting here\n")
        out = tmp_path / "report.json"

        # When
        exit_code = parser_module.main(["--label", "baseline", str(log), "--output", str(out)])

        # Then
        assert_that(exit_code).is_equal_to(0)
        assert_that(capsys.readouterr().err).contains("No checkpoint timings found")


class TestCompare:
    def _report(self, label, timers):
        return {"label": label, "timers": timers, "raw_samples": {}}

    def test_orders_headline_timers_first(self, compare_module):
        # Given
        stats = {"count": 1, "mean_s": 1.0, "max_s": 1.0}
        baseline = self._report("baseline", {"zzz-other": stats, "save-checkpoint": stats})
        candidate = self._report("flashpoint", {"zzz-other": stats, "save-checkpoint": stats})

        # When
        rows = compare_module.compare(baseline, candidate)

        # Then
        assert_that(rows[0]["timer"]).is_equal_to("save-checkpoint")

    def test_computes_the_speedup(self, compare_module):
        # Given
        baseline = self._report("baseline", {"save-checkpoint": {"count": 2, "mean_s": 18.0, "max_s": 19.0}})
        candidate = self._report("flashpoint", {"save-checkpoint": {"count": 2, "mean_s": 0.6, "max_s": 0.7}})

        # When
        rows = compare_module.compare(baseline, candidate)

        # Then
        assert_that(rows[0]["mean_delta"]).contains("30.00x")
        assert_that(rows[0]["mean_delta"]).contains("-17.400s")

    def test_missing_timer_on_one_side_is_not_an_error(self, compare_module):
        # Given
        baseline = self._report("baseline", {})
        candidate = self._report(
            "flashpoint", {"save-checkpoint-non-persistent": {"count": 1, "mean_s": 0.6, "max_s": 0.6}}
        )

        # When
        rows = compare_module.compare(baseline, candidate)

        # Then
        assert_that(rows[0]["mean_delta"]).is_equal_to("n/a")
        assert_that(rows[0]["baseline_count"]).is_equal_to(0)

    def test_zero_baseline_is_reported_rather_than_dividing(self, compare_module):
        # Given
        baseline = self._report("baseline", {"save-checkpoint": {"count": 1, "mean_s": 0.0, "max_s": 0.0}})
        candidate = self._report("flashpoint", {"save-checkpoint": {"count": 1, "mean_s": 1.0, "max_s": 1.0}})

        # When
        rows = compare_module.compare(baseline, candidate)

        # Then
        assert_that(rows[0]["mean_delta"]).contains("baseline is 0")

    def test_render_includes_both_labels(self, compare_module):
        # Given
        rows = [
            {
                "timer": "save-checkpoint",
                "baseline_mean_s": 18.0,
                "candidate_mean_s": 0.6,
                "baseline_count": 2,
                "candidate_count": 2,
                "mean_delta": "-17.400s",
            }
        ]

        # When
        rendered = compare_module.render("baseline", "flashpoint", rows)

        # Then
        assert_that(rendered).contains("flashpoint vs baseline")
        assert_that(rendered).contains("save-checkpoint")


class TestCompareCli:
    def _write(self, path, label, timers):
        path.write_text(json.dumps({"label": label, "timers": timers, "raw_samples": {}}))

    def test_prints_a_table(self, compare_module, tmp_path, capsys):
        # Given
        stats = {"count": 1, "mean_s": 2.0, "max_s": 2.0}
        base = tmp_path / "base.json"
        cand = tmp_path / "cand.json"
        self._write(base, "baseline", {"save-checkpoint": stats})
        self._write(cand, "flashpoint", {"save-checkpoint": stats})

        # When
        exit_code = compare_module.main(["--baseline", str(base), "--candidate", str(cand)])

        # Then
        assert_that(exit_code).is_equal_to(0)
        assert_that(capsys.readouterr().out).contains("save-checkpoint")

    def test_emits_json_on_request(self, compare_module, tmp_path, capsys):
        # Given
        stats = {"count": 1, "mean_s": 2.0, "max_s": 2.0}
        base = tmp_path / "base.json"
        cand = tmp_path / "cand.json"
        self._write(base, "baseline", {"save-checkpoint": stats})
        self._write(cand, "flashpoint", {"save-checkpoint": stats})

        # When
        compare_module.main(["--baseline", str(base), "--candidate", str(cand), "--json"])

        # Then
        payload = json.loads(capsys.readouterr().out)
        assert_that(payload["candidate_label"]).is_equal_to("flashpoint")

    def test_empty_reports_exit_non_zero(self, compare_module, tmp_path):
        # Given
        base = tmp_path / "base.json"
        cand = tmp_path / "cand.json"
        self._write(base, "baseline", {})
        self._write(cand, "flashpoint", {})

        # When
        exit_code = compare_module.main(["--baseline", str(base), "--candidate", str(cand)])

        # Then
        assert_that(exit_code).is_equal_to(1)
