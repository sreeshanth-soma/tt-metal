import copy
from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import report_repeat_sweep as report
from repeat_sweep_cases import SWEEP_CASES, manifest, manifest_sha256
from test_report_candidate_run import probe_records, write_probe
from test_summarize_repeat_capture import capture


def sweep_probe(cases, profiling, index=0):
    records = probe_records([case.name for case in cases], profiling)
    environment = records[0]
    environment.update(
        {
            "utc": (datetime(2026, 9, 21, tzinfo=timezone.utc) + timedelta(seconds=index)).isoformat(),
            "tracked_changes": {"returncode": 0, "stdout": ""},
            "case_catalog_sha256": "3" * 64,
            "sweep_manifest_sha256": manifest_sha256(),
            "python": "fixture_not_hardware",
        }
    )
    sample_count = 5 if profiling else 51
    environment["arguments"].update(
        {
            "suite": "sweep",
            "case": cases[0].name if profiling else None,
            "warmup": 3 if profiling else 10,
            "samples": sample_count,
            "tracy_signposts": profiling,
        }
    )
    records[1]["ttnn_version"] = "fixture_not_hardware"
    for record in records:
        if record["event"] == "candidate_result":
            for name, median in (("public_repeat", 200.0), ("direct_codegen", 100.0)):
                record["timings"][name]["samples_us"] = [median] * (sample_count - 1) + [5000.0]
    starts = [{"event": "case_start", "case": asdict(case), "dtype": case.dtype} for case in cases]
    end = {"event": "source_state_end", **{field: copy.deepcopy(environment[field]) for field in report.STATE_FIELDS}}
    return json.loads(json.dumps(records[:3] + starts + records[3:-2] + [end] + records[-2:]))


def write_capture(path, case, candidate_duration=None, shift_indices=0, device="0"):
    rows = capture(sample_count=5)
    active = ""
    for row in rows:
        if row["OP TYPE"] == "signpost":
            row["OP CODE"] = row["OP CODE"].replace("TT_BOUNTY/aligned_w/", f"TT_BOUNTY/{case.name}/")
            parts = row["OP CODE"].split("/")
            parts[4] = str(int(parts[4]) + shift_indices)
            row["OP CODE"] = "/".join(parts)
            active = row["OP CODE"]
        elif row["OP TYPE"] == "tt_dnn_device":
            row["DEVICE ID"] = device
            if candidate_duration is not None and "/direct_codegen/sample/" in active:
                row["DEVICE KERNEL DURATION [ns]"] = str(candidate_duration)
    columns = sorted({column for row in rows for column in row})
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def create_sweep(directory):
    (directory / "manifest.json").write_text(json.dumps({**manifest(), "manifest_sha256": manifest_sha256()}))
    (directory / "correctness.xml").write_text(
        '<testsuites><testsuite tests="177" skipped="0" failures="0" errors="0"/></testsuites>'
    )
    write_probe(directory / "unprofiled.jsonl", sweep_probe(SWEEP_CASES, False))
    for index, case in enumerate(SWEEP_CASES, 1):
        case_directory = directory / case.name
        reports = case_directory / "reports/fixture"
        reports.mkdir(parents=True)
        write_probe(case_directory / "probe.jsonl", sweep_probe([case], True, index))
        write_capture(reports / "ops_perf_results_fixture.csv", case)


class SweepReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tt-sweep-report-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_complete_report_retains_every_case_and_outliers(self):
        create_sweep(self.directory)
        result = report.load_sweep(self.directory)
        self.assertTrue(result["coverage_complete"])
        self.assertTrue(result["all_cases_improve"])
        self.assertFalse(result["automatic_subtile_routing"])
        self.assertEqual(len(result["comparisons"]), 96)
        self.assertEqual(len(result["raw_artifacts"]), 195)
        for case in SWEEP_CASES:
            self.assertEqual(result["host"][case.name]["public_repeat"]["maximum"], 5000)
            self.assertIn(f"| {case.name} |", report.render(result))

    def test_losing_cases_cannot_be_hidden_by_aggregate_winners(self):
        create_sweep(self.directory)
        path = self.directory / "unprofiled.jsonl"
        records = sweep_probe(SWEEP_CASES, False)
        losing = SWEEP_CASES[-1]
        result_record = next(
            record for record in records if record["event"] == "candidate_result" and record["case"] == losing.name
        )
        result_record["timings"]["direct_codegen"]["samples_us"] = [400.0] * 51
        write_probe(path, records)
        write_capture(
            self.directory / losing.name / "reports/fixture/ops_perf_results_fixture.csv",
            losing,
            candidate_duration=6000,
        )
        result = report.load_sweep(self.directory)
        self.assertFalse(result["all_cases_improve"])
        self.assertEqual(len(result["comparisons"]), 96)
        self.assertEqual({entry["metric"] for entry in result["non_improving"]}, {"host", "device"})
        self.assertEqual({entry["case"] for entry in result["non_improving"]}, {losing.name})
        output = self.directory / "SWEEP.json"
        with mock.patch.object(report, "load_sweep", return_value=result), redirect_stdout(io.StringIO()):
            self.assertEqual(report.main([str(self.directory), "--json-output", str(output)]), 1)
        self.assertFalse(json.loads(output.read_text())["all_cases_improve"])

    def test_rejects_changed_case_specification_or_missing_cases(self):
        path = self.directory / "probe.jsonl"
        baseline = sweep_probe(SWEEP_CASES, False)
        for field, replacement in (("dtype", "fp32"), ("shape", [1, 2, 3, 4]), ("dimension", 1), ("repeats", 127)):
            records = copy.deepcopy(baseline)
            next(record for record in records if record["event"] == "case_start")["case"][field] = replacement
            write_probe(path, records)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "geometry"):
                report.load_sweep_probe(path, False, SWEEP_CASES)
        write_probe(path, sweep_probe(SWEEP_CASES[:-1], False))
        with self.assertRaisesRegex(ValueError, "cases"):
            report.load_sweep_probe(path, False, SWEEP_CASES)

    def test_rejects_dirty_changed_unlabelled_or_short_probes(self):
        path = self.directory / "probe.jsonl"
        mutations = (
            lambda records: records[0]["tracked_changes"].update(stdout=" M source.cpp"),
            lambda records: report.only(records, "source_state_end").update(probe_sha256="9" * 64),
            lambda records: records[0]["arguments"].update(warmup=1),
            lambda records: records[0].update(sweep_manifest_sha256="0" * 64),
            lambda records: records[1]["runtime_config"].update(enable_logging=True),
            lambda records: records[0].update(utc="2026-09-21T00:00:00"),
            lambda records: records[0]["arguments"].update(tracy_signposts=True),
        )
        for mutation in mutations:
            records = sweep_probe(SWEEP_CASES, False)
            mutation(records)
            write_probe(path, records)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.load_sweep_probe(path, False, SWEEP_CASES)
        records = sweep_probe(SWEEP_CASES, False)
        records[0]["arguments"]["samples"] = 50
        for record in records:
            if record["event"] == "candidate_result":
                for timings in record["timings"].values():
                    timings["samples_us"] = timings["samples_us"][:50]
        write_probe(path, records)
        with self.assertRaisesRegex(ValueError, "51/five"):
            report.load_sweep_probe(path, False, SWEEP_CASES)

    def test_rejects_saved_manifest_selection_and_skipped_device_tests(self):
        create_sweep(self.directory)
        path = self.directory / "manifest.json"
        complete_manifest = path.read_text()
        data = json.loads(complete_manifest)
        data["cases"].pop()
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "manifest"):
            report.load_sweep(self.directory)
        path.write_text(complete_manifest)
        (self.directory / "correctness.xml").write_text('<testsuite tests="177" skipped="1"/>')
        with self.assertRaisesRegex(ValueError, "zero skips"):
            report.load_sweep(self.directory)

    def test_rejects_reused_starts_and_cross_capture_source_changes(self):
        create_sweep(self.directory)
        case = SWEEP_CASES[0]
        path = self.directory / case.name / "probe.jsonl"
        write_probe(path, sweep_probe([case], True, 0))
        with self.assertRaisesRegex(ValueError, "Duplicate probe"):
            report.load_sweep(self.directory)
        records = sweep_probe([case], True, 1)
        for record in (records[0], report.only(records, "source_state_end")):
            record["source_sha256"] = {key: "9" * 64 for key in record["source_sha256"]}
        write_probe(path, records)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            report.load_sweep(self.directory)

    def test_requires_raw_single_csv_with_matching_indices_and_device(self):
        create_sweep(self.directory)
        case = SWEEP_CASES[0]
        path = self.directory / case.name / "reports/fixture/ops_perf_results_fixture.csv"
        write_capture(path, case, shift_indices=1)
        with self.assertRaisesRegex(ValueError, "sample indices"):
            report.load_sweep(self.directory)
        write_capture(path, case, device="1")
        with self.assertRaisesRegex(ValueError, "Device ID"):
            report.load_sweep(self.directory)
        duplicate = path.with_name("ops_perf_results_duplicate.csv")
        duplicate.write_text(path.read_text())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            report.load_sweep(self.directory)

    def test_report_refuses_partial_evidence_without_writing_success_json(self):
        output = self.directory / "SWEEP.json"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(report.main([str(self.directory), "--json-output", str(output)]), 2)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
