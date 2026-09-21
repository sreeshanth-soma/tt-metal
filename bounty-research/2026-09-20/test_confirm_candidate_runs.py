from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import confirm_candidate_runs as confirmation
from report_candidate_run import IMPLEMENTATIONS, REPEAT_CASES
from test_report_candidate_run import probe_records, write_probe
from test_summarize_repeat_capture import capture


def populate_run(directory, ordinal, host_samples=51, device_samples=5, commit="a" * 40):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "correctness.xml").write_text('<testsuites><testsuite tests="177"/></testsuites>')
    names = [case.name for case in REPEAT_CASES]
    probes = [(directory / "unprofiled.jsonl", names, False)]
    probes.extend((directory / case / "probe.jsonl", [case], True) for case in names if case != "outer_control")
    for index, (path, cases, profiling) in enumerate(probes):
        records = probe_records(cases, profiling)
        count = device_samples if profiling else host_samples
        records[0].update(
            utc=f"2026-01-01T00:{ordinal:02d}:{index:02d}+00:00",
            tracked_changes={"returncode": 0, "stdout": "", "stderr": ""},
            git_head={"returncode": 0, "stdout": commit},
        )
        records[0]["arguments"]["samples"] = count
        for record in records:
            if record.get("event") == "candidate_result":
                for implementation in IMPLEMENTATIONS:
                    original = record["timings"][implementation]["samples_us"]
                    record["timings"][implementation]["samples_us"] = [
                        original[sample % len(original)] for sample in range(count)
                    ]
        path.parent.mkdir(parents=True, exist_ok=True)
        write_probe(path, records)
        if profiling:
            rows = capture(device_samples)
            for row in rows:
                row["OP CODE"] = row["OP CODE"].replace("/aligned_w/", f"/{cases[0]}/")
            reports = path.parent / "reports/test-fixture"
            reports.mkdir(parents=True, exist_ok=True)
            with (reports / "ops_perf_results_fixture.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=sorted({key for row in rows for key in row}))
                writer.writeheader()
                writer.writerows(rows)


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tt-confirm-report-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directories = [self.root / f"run-{index}" for index in range(1, 4)]
        for index, directory in enumerate(self.directories, 1):
            populate_run(directory, index)

    def change_host_result(self, case, candidate_samples):
        path = self.directories[-1] / "unprofiled.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        record = next(entry for entry in records if entry.get("event") == "candidate_result" and entry["case"] == case)
        record["timings"]["direct_codegen"]["samples_us"] = candidate_samples
        write_probe(path, records)

    def test_complete_runs_keep_separate_ratios_and_outliers(self):
        result = confirmation.confirm(self.directories)
        self.assertTrue(result["passed"])
        self.assertEqual(result["cases"]["aligned_w"]["host_ratios"], [2, 2, 2])
        self.assertEqual(result["cases"]["aligned_w"]["device_ratios"], [2, 2, 2])
        self.assertEqual(result["runs"][0]["host"]["aligned_w"]["public_repeat"]["maximum"], 5000)
        self.assertFalse(result["automatic_subtile_routing"])
        self.assertIn("fixture_not_hardware", confirmation.render(result))
        self.assertIn("Automatic H/W routing remains disabled", confirmation.render(result))

    def test_requires_three_distinct_directories(self):
        for directories in (self.directories[:2], self.directories[:2] + self.directories[:1]):
            with self.subTest(directories=directories), self.assertRaisesRegex(ValueError, "three distinct"):
                confirmation.confirm(directories)

    def test_rejects_copied_run_artifacts(self):
        duplicate = self.root / "copied-run"
        shutil.copytree(self.directories[0], duplicate)
        with self.assertRaisesRegex(ValueError, "Duplicate probe starts"):
            confirmation.confirm(self.directories[:2] + [duplicate])

    def test_rejects_different_commits_across_complete_runs(self):
        populate_run(self.directories[-1], 3, commit="b" * 40)
        with self.assertRaisesRegex(ValueError, "differs in run 3"):
            confirmation.confirm(self.directories)

    def test_rejects_dirty_or_undated_probes(self):
        for update, message in (
            ({"tracked_changes": {"returncode": 0, "stdout": " M kernel.cpp"}}, "clean tracked"),
            ({"utc": None}, "timezone-aware"),
            ({"utc": "2026-01-01T00:03:00"}, "timezone-aware"),
            ({"utc": "not-a-date"}, "Invalid probe"),
        ):
            with self.subTest(update=update):
                populate_run(self.directories[-1], 3)
                path = self.directories[-1] / "unprofiled.jsonl"
                records = [json.loads(line) for line in path.read_text().splitlines()]
                records[0].update(update)
                write_probe(path, records)
                with self.assertRaisesRegex(ValueError, message):
                    confirmation.confirm(self.directories)

    def test_rejects_short_samples_or_skipped_tests(self):
        for host_samples, device_samples, message in ((3, 5, "51 host"), (51, 3, "five device")):
            with self.subTest(host_samples=host_samples, device_samples=device_samples):
                populate_run(self.directories[-1], 3, host_samples, device_samples)
                with self.assertRaisesRegex(ValueError, message):
                    confirmation.confirm(self.directories)
        populate_run(self.directories[-1], 3)
        (self.directories[-1] / "correctness.xml").write_text(
            '<testsuites><testsuite tests="177" skipped="1"/></testsuites>'
        )
        with self.assertRaisesRegex(ValueError, "zero skips"):
            confirmation.confirm(self.directories)

    def test_host_regression_is_not_hidden_by_other_runs(self):
        self.change_host_result("aligned_w", [400] * 51)
        result = confirmation.confirm(self.directories)
        self.assertFalse(result["passed"])
        self.assertEqual(result["cases"]["aligned_w"]["host_ratios"], [2, 2, 0.5])
        self.assertIn("run 3 host ratio", result["failures"][0])
        self.assertIn("NOT CONFIRMED", confirmation.render(result))

    def test_device_regression_is_not_hidden_by_host_improvement(self):
        path = self.directories[-1] / "gva_prefill_h/reports/test-fixture/ops_perf_results_fixture.csv"
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            fields, rows = reader.fieldnames, list(reader)
        in_candidate = False
        for row in rows:
            if row["OP TYPE"] == "signpost":
                in_candidate = "/direct_codegen/sample/" in row["OP CODE"] and row["OP CODE"].endswith("/begin")
            elif in_candidate:
                row["DEVICE KERNEL DURATION [ns]"] = "6000"
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        result = confirmation.confirm(self.directories)
        self.assertFalse(result["passed"])
        self.assertIn("run 3 device ratio", result["failures"][0])

    def test_outer_control_is_not_credited_to_new_reader(self):
        self.change_host_result("outer_control", [400] * 51)
        result = confirmation.confirm(self.directories)
        self.assertTrue(result["passed"])
        self.assertEqual(result["cases"]["outer_control"]["host_ratios"][-1], 0.5)
        self.assertIn("control; excluded", confirmation.render(result))

    def test_cli_returns_failure_and_preserves_existing_json(self):
        self.change_host_result("aligned_w", [400] * 51)
        destination = self.root / "confirmation.json"
        arguments = [str(path) for path in self.directories] + ["--json-output", str(destination)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(confirmation.main(arguments), 1)
        original = destination.read_bytes()
        self.assertFalse(json.loads(original)["passed"])
        with redirect_stderr(io.StringIO()):
            self.assertEqual(confirmation.main(arguments), 2)
        self.assertEqual(destination.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
