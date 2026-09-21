"""Produce one readable report from a complete, internally consistent hardware validation run."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ElementTree

from repeat_candidate_probe import EXPECTED_BASE, REPEAT_CASES, REVISION, SOURCE_PATHS
from summarize_repeat_capture import extract_samples, summarize

IMPLEMENTATIONS = ("public_repeat", "direct_codegen")


def only(records, event):
    matches = [record for record in records if record.get("event") == event]
    if len(matches) != 1:
        raise ValueError(f"Expected one {event} record, got {len(matches)}")
    return matches[0]


def load_probe(path, profiling, case_names):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(record.get("event") in ("error", "close_error") for record in records):
        raise ValueError(f"Probe failed: {path}")
    only(records, "suite_finished")
    only(records, "device_closed")
    environment = only(records, "environment")
    modules = only(records, "modules")
    device = only(records, "device")
    if environment.get("base_commit") != EXPECTED_BASE or environment["revision"] != REVISION:
        raise ValueError("Probe has the wrong source base or revision")
    commit = environment["git_head"]["stdout"].strip()
    if (
        environment["git_head"].get("returncode") != 0
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("Probe has an invalid Git commit identity")
    if set(environment["source_sha256"]) != set(SOURCE_PATHS):
        raise ValueError("Probe lacks the complete candidate source hashes")
    if modules["device_profiling"] is not profiling or modules["tracy_signposts"] is not profiling:
        raise ValueError("Profiled and unprofiled measurements must stay separate")
    result_records = [record for record in records if record.get("event") == "candidate_result"]
    if len(result_records) != len(case_names) or {record["case"] for record in result_records} != set(case_names):
        raise ValueError("Probe has missing, duplicate or unexpected cases")
    results = {}
    for record in result_records:
        case = record["case"]
        case_records = [entry for entry in records if entry.get("case") == case]
        if only(case_records, "input_roundtrip")["metrics"].get("exact_bits") is not True:
            raise ValueError(f"Input roundtrip failed for {case}")
        if record["device_profiling"] is not profiling or record["tracy_signposts"] is not profiling:
            raise ValueError("Profiled and unprofiled measurements must stay separate")
        results[case] = {}
        for implementation in IMPLEMENTATIONS:
            checks = [entry for entry in case_records if entry.get("implementation") == implementation]
            for event in ("correctness", "post_timing_correctness"):
                if only(checks, event)["metrics"].get("exact_bits") is not True:
                    raise ValueError(f"Exact correctness failed for {case}/{implementation}/{event}")
            samples = record["timings"][implementation]["samples_us"]
            if len(samples) != environment["arguments"]["samples"] or len(samples) < 3:
                raise ValueError("Timing sample count does not match the probe arguments")
            if any(not math.isfinite(sample) or sample <= 0 for sample in samples):
                raise ValueError("Timing samples must be finite and positive")
            results[case][implementation] = {
                "median": statistics.median(samples),
                "minimum": min(samples),
                "maximum": max(samples),
                "count": len(samples),
            }
    identity = {
        "git_commit": commit,
        "source_sha256": environment["source_sha256"],
        "probe_sha256": environment["probe_sha256"],
        "timing_helper_sha256": environment["timing_helper_sha256"],
        "python_executable": environment["python_executable"],
        "device_index": environment["arguments"]["device"],
        "visible_devices": environment["environment"].get("TT_VISIBLE_DEVICES"),
        "architecture": device["architecture"],
        "compute_grid": device["compute_grid"],
        "ttnn_file": modules["ttnn_file"],
        "torch_version": modules["torch_version"],
        "runtime_config": modules["runtime_config"],
    }
    return identity, results


def pytest_counts(path):
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {
        name: sum(int(suite.get(name, "0")) for suite in suites) for name in ("tests", "skipped", "failures", "errors")
    }
    if counts["failures"] or counts["errors"] or counts["tests"] <= counts["skipped"]:
        raise ValueError("Device pytest did not complete successfully with executed tests")
    return counts


def probe_metadata(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    environment = only(records, "environment")
    return {name: environment.get(name) for name in ("utc", "tracked_changes")}


def load_run(directory):
    counts = pytest_counts(directory / "correctness.xml")
    case_names = [case.name for case in REPEAT_CASES]
    identity, host = load_probe(directory / "unprofiled.jsonl", False, case_names)
    metadata = [probe_metadata(directory / "unprofiled.jsonl")]
    device = {}
    for case in case_names:
        if case == "outer_control":
            continue
        capture_identity, capture_timings = load_probe(directory / case / "probe.jsonl", True, [case])
        if capture_identity != identity:
            raise ValueError(f"Source, runtime or device changed between captures: {case}")
        paths = list((directory / case / "reports").glob("*/ops_perf_results_*.csv"))
        if len(paths) != 1:
            raise ValueError(f"Expected exactly one device CSV for {case}")
        with paths[0].open(newline="", encoding="utf-8") as source:
            samples = extract_samples(csv.DictReader(source), require_candidate=True)
        measured = summarize(samples, require_candidate=True)
        if set(measured) != {case}:
            raise ValueError(f"CSV case does not match its probe: {case}")
        if any(summary["samples"] != capture_timings[case][name]["count"] for name, summary in measured[case].items()):
            raise ValueError(f"CSV sample count does not match its probe: {case}")
        device[case] = measured[case]
        metadata.append(probe_metadata(directory / case / "probe.jsonl"))
    return {
        "directory": str(directory.resolve()),
        "identity": identity,
        "pytest": counts,
        "host": host,
        "device": device,
        "probes": metadata,
    }


def render_report(directory):
    run = load_run(directory)
    counts, identity, host = run["pytest"], run["identity"], run["host"]
    case_names = [case.name for case in REPEAT_CASES]
    device_rows = []
    for case, implementations in run["device"].items():
        public = implementations["public_repeat"]["median_kernel_sum_us"]
        candidate = implementations["direct_codegen"]["median_kernel_sum_us"]
        device_rows.append(f"| {case} | {public:.3f} | {candidate:.3f} | {public / candidate:.3f}x |")
    lines = [
        "# Repeat candidate hardware report",
        "",
        f"Base: `{EXPECTED_BASE}`. Commit: `{identity['git_commit']}`.",
        f"Device: `{identity['architecture']}`, grid `{identity['compute_grid']}`.",
        f"Device pytest: {counts['tests'] - counts['skipped']} passed, {counts['skipped']} skipped; no failures/errors.",
        "All benchmark input, pre-timing and post-timing bitwise checks passed. The new reader appears in every candidate capture.",
        "",
        "## Unprofiled host-to-completion latency (microseconds)",
        "",
        "| Case | Samples/leg | Public median [min, max] | Candidate median [min, max] | Public/candidate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for case in case_names:
        public, candidate = (host[case][name] for name in IMPLEMENTATIONS)
        lines.append(
            f"| {case} | {public['count']} | {public['median']:.3f} [{public['minimum']:.3f}, {public['maximum']:.3f}] | "
            f"{candidate['median']:.3f} [{candidate['minimum']:.3f}, {candidate['maximum']:.3f}] | "
            f"{public['median'] / candidate['median']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "## Profiled device time: median per-call kernel sums (microseconds)",
            "",
            "| Case | Public | Candidate | Public/candidate |",
            "| --- | ---: | ---: | ---: |",
            *device_rows,
            "",
            "Ratios above 1 mean a lower candidate median; below 1 mean a regression on that metric. Outliers are retained.",
            "The outer control does not exercise the new H/W reader. Profiling measurements are not unprofiled latency.",
            "Automatic H/W routing remains disabled. Review repeat runs and other required architectures before promoting any case.",
            "This report does not establish model-level improvement, maintainer acceptance or bounty eligibility.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args(argv)
    print(render_report(arguments.directory), end="")


if __name__ == "__main__":
    main()
