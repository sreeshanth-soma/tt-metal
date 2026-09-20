import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import report_candidate_run as report
from test_summarize_repeat_capture import capture


def probe_records(case_names, profiling):
    records = [
        {
            "event": "environment",
            "git_head": {"stdout": "a" * 40, "returncode": 0},
            "base_commit": report.EXPECTED_BASE,
            "revision": report.REVISION,
            "source_sha256": {path: "0" * 64 for path in report.SOURCE_PATHS},
            "probe_sha256": "1" * 64,
            "timing_helper_sha256": "2" * 64,
            "python_executable": "/candidate/python_env/bin/python",
            "arguments": {"samples": 3, "device": 0},
            "environment": {},
        },
        {
            "event": "modules",
            "device_profiling": profiling,
            "tracy_signposts": profiling,
            "ttnn_file": "/candidate/ttnn/ttnn/__init__.py",
            "torch_version": "test-fixture",
            "runtime_config": {},
        },
        {"event": "device", "architecture": "fixture_not_hardware", "compute_grid": {"x": 8, "y": 9}},
    ]
    for case in case_names:
        records.append({"event": "input_roundtrip", "case": case, "metrics": {"exact_bits": True}})
        for implementation in report.IMPLEMENTATIONS:
            for event in ("correctness", "post_timing_correctness"):
                records.append(
                    {"event": event, "case": case, "implementation": implementation, "metrics": {"exact_bits": True}}
                )
        records.append(
            {
                "event": "candidate_result",
                "case": case,
                "device_profiling": profiling,
                "tracy_signposts": profiling,
                "timings": {
                    "public_repeat": {"samples_us": [100.0, 200.0, 5000.0]},
                    "direct_codegen": {"samples_us": [50.0, 100.0, 150.0]},
                },
            }
        )
    records.extend([{"event": "suite_finished"}, {"event": "device_closed"}])
    return records


def write_probe(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


class CandidateRunReportTests(unittest.TestCase):
    def test_uses_raw_samples_and_retains_outliers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "probe.jsonl"
            write_probe(path, probe_records(["aligned_w"], False))
            identity, results = report.load_probe(path, False, ["aligned_w"])
        self.assertEqual(identity["architecture"], "fixture_not_hardware")
        self.assertEqual(results["aligned_w"]["public_repeat"]["median"], 200)
        self.assertEqual(results["aligned_w"]["public_repeat"]["maximum"], 5000)

    def test_rejects_failed_partial_profiled_or_nonfinite_baselines(self):
        baseline = probe_records(["aligned_w"], False)
        bad_records = [baseline[:-1], baseline + [{"event": "error"}], probe_records(["aligned_w"], True)]
        bad_checks = copy.deepcopy(baseline)
        next(record for record in bad_checks if record["event"] == "post_timing_correctness")["metrics"][
            "exact_bits"
        ] = False
        bad_records.append(bad_checks)
        bad_timings = copy.deepcopy(baseline)
        next(record for record in bad_timings if record["event"] == "candidate_result")["timings"]["public_repeat"][
            "samples_us"
        ][0] = float("nan")
        bad_records.append(bad_timings)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "probe.jsonl"
            for records in bad_records:
                with self.subTest(records=records):
                    write_probe(path, records)
                    with self.assertRaises(ValueError):
                        report.load_probe(path, False, ["aligned_w"])

    def test_rejects_missing_cases_or_checks(self):
        records = probe_records(["aligned_w"], False)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "probe.jsonl"
            write_probe(path, records)
            with self.assertRaisesRegex(ValueError, "cases"):
                report.load_probe(path, False, ["aligned_w", "aligned_h"])
            write_probe(path, [record for record in records if record["event"] != "correctness"])
            with self.assertRaisesRegex(ValueError, "correctness"):
                report.load_probe(path, False, ["aligned_w"])

    def test_rejects_wrong_base_and_records_candidate_commit(self):
        records = probe_records(["aligned_w"], False)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "probe.jsonl"
            write_probe(path, records)
            identity = report.load_probe(path, False, ["aligned_w"])[0]
            self.assertEqual(identity["git_commit"], "a" * 40)
            records[0]["base_commit"] = "b" * 40
            write_probe(path, records)
            with self.assertRaisesRegex(ValueError, "source base"):
                report.load_probe(path, False, ["aligned_w"])

    def test_rejects_failed_or_skipped_only_device_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "correctness.xml"
            for attributes in ('tests="2" failures="1"', 'tests="2" errors="1"', 'tests="2" skipped="2"'):
                with self.subTest(attributes=attributes):
                    path.write_text(f"<testsuites><testsuite {attributes}/></testsuites>")
                    with self.assertRaisesRegex(ValueError, "pytest"):
                        report.pytest_counts(path)

    def test_complete_report_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            names = [case.name for case in report.REPEAT_CASES]
            write_probe(directory / "unprofiled.jsonl", probe_records(names, False))
            (directory / "correctness.xml").write_text('<testsuites><testsuite tests="84" skipped="3"/></testsuites>')
            for case in names:
                if case == "outer_control":
                    continue
                case_dir = directory / case
                report_dir = case_dir / "reports" / "test-fixture"
                report_dir.mkdir(parents=True)
                write_probe(case_dir / "probe.jsonl", probe_records([case], True))
                rows = capture()
                for row in rows:
                    row["OP CODE"] = row["OP CODE"].replace("/aligned_w/", f"/{case}/")
                with (report_dir / "ops_perf_results_fixture.csv").open("w", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=sorted({key for row in rows for key in row}))
                    writer.writeheader()
                    writer.writerows(rows)
            rendered = report.render_report(directory)
            self.assertIn("81 passed, 3 skipped", rendered)
            self.assertIn("5000.000", rendered)
            self.assertIn("3.002 | 1.501 | 2.000x", rendered)
            self.assertIn("Automatic H/W routing remains disabled", rendered)
            self.assertIn(f"Commit: `{'a' * 40}`", rendered)
            changed_commit = probe_records(["aligned_w"], True)
            changed_commit[0]["git_head"]["stdout"] = "b" * 40
            write_probe(directory / "aligned_w" / "probe.jsonl", changed_commit)
            with self.assertRaisesRegex(ValueError, "changed between captures"):
                report.render_report(directory)
            changed = probe_records(["aligned_w"], True)
            changed[0]["source_sha256"][report.SOURCE_PATHS[0]] = "3" * 64
            write_probe(directory / "aligned_w" / "probe.jsonl", changed)
            with self.assertRaisesRegex(ValueError, "changed between captures"):
                report.render_report(directory)


if __name__ == "__main__":
    unittest.main()
