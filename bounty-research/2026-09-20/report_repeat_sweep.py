"""Audit the full fixed sweep from raw artifacts and expose every non-improving case."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ElementTree

from repeat_candidate_probe import require_clean_state
from repeat_sweep_cases import DTYPES, SWEEP_CASES, describe_case, manifest, manifest_sha256
from report_candidate_run import IMPLEMENTATIONS, load_probe, only, pytest_counts
from summarize_repeat_capture import extract_samples, summarize


STATE_FIELDS = (
    "git_head",
    "tracked_changes",
    "source_sha256",
    "probe_sha256",
    "timing_helper_sha256",
    "case_catalog_sha256",
)


def load_sweep_probe(path, profiling, cases):
    identity, timings = load_probe(path, profiling, [case.name for case in cases])
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    environment = only(records, "environment")
    arguments = environment["arguments"]
    if arguments.get("suite") != "sweep" or environment.get("sweep_manifest_sha256") != manifest_sha256():
        raise ValueError("Probe does not match the fixed sweep manifest")
    expected_case = cases[0].name if profiling else None
    if arguments.get("case") != expected_case:
        raise ValueError("Sweep requires all host cases and one case per device capture")
    if arguments.get("tracy_signposts") is not profiling:
        raise ValueError("Declared profiling mode differs from the sweep capture")
    if arguments.get("warmup", 0) < (3 if profiling else 10) or arguments["samples"] < (5 if profiling else 51):
        raise ValueError("Sweep requires ten/three warmups and at least 51/five host/device samples")
    end_state = only(records, "source_state_end")
    for record in (environment, end_state):
        try:
            require_clean_state(record)
        except RuntimeError as error:
            raise ValueError(str(error)) from error
    if any(environment[field] != end_state[field] for field in STATE_FIELDS):
        raise ValueError("Source or benchmark tools changed during a probe")
    hashes = [
        *environment["source_sha256"].values(),
        *(environment[field] for field in ("probe_sha256", "timing_helper_sha256", "case_catalog_sha256")),
    ]
    if any(not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef") for value in hashes):
        raise ValueError("Invalid source/tool SHA-256 identity")
    starts = [record for record in records if record.get("event") == "case_start"]
    expected = {case.name: json.loads(json.dumps(asdict(case))) for case in cases}
    if len(starts) != len(cases) or {record["case"]["name"] for record in starts} != set(expected):
        raise ValueError("Missing, duplicate or unexpected case specifications")
    for record in starts:
        specification = expected[record["case"]["name"]]
        if record["case"] != specification or record["dtype"] != specification["dtype"]:
            raise ValueError("Case geometry, repeat count or dtype differs from the manifest")
    configuration = only(records, "modules")["runtime_config"]
    if any(
        configuration.get(flag)
        for flag in ("enable_comparison_mode", "enable_logging", "enable_graph_report", "enable_tensor_report")
    ):
        raise ValueError("Instrumented TTNN runtime would contaminate sweep timings")
    started = datetime.fromisoformat(environment["utc"].replace("Z", "+00:00"))
    if started.tzinfo is None:
        raise ValueError("Probe start must be timezone-aware")
    identity["case_catalog_sha256"] = environment["case_catalog_sha256"]
    identity["sweep_manifest_sha256"] = environment["sweep_manifest_sha256"]
    identity["python"] = environment["python"]
    identity["ttnn_version"] = only(records, "modules")["ttnn_version"]
    return identity, timings, started.astimezone(timezone.utc)


def file_evidence(path, directory):
    return {"path": str(path.relative_to(directory)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_sweep(directory):
    directory = directory.resolve()
    recorded_manifest = json.loads((directory / "manifest.json").read_text())
    if recorded_manifest != {**manifest(), "manifest_sha256": manifest_sha256()}:
        raise ValueError("Saved manifest differs from the complete fixed sweep")
    counts = pytest_counts(directory / "correctness.xml")
    if counts["skipped"] or counts["tests"] < 177:
        raise ValueError("Sweep requires at least 177 executed device tests with zero skips")
    host_path = directory / "unprofiled.jsonl"
    identity, host, host_start = load_sweep_probe(host_path, False, SWEEP_CASES)
    seen_starts = {host_start}
    device_ids = set()
    device = {}
    evidence = [
        file_evidence(directory / name, directory) for name in ("manifest.json", "correctness.xml", "unprofiled.jsonl")
    ]
    for case in SWEEP_CASES:
        case_directory = directory / case.name
        probe_path = case_directory / "probe.jsonl"
        capture_identity, capture_timings, started = load_sweep_probe(probe_path, True, [case])
        if capture_identity != identity:
            raise ValueError(f"Source, runtime or reported device identity differs for {case.name}")
        if started in seen_starts:
            raise ValueError("Duplicate probe starts; do not reuse captures")
        seen_starts.add(started)
        paths = list((case_directory / "reports").glob("*/ops_perf_results_*.csv"))
        if len(paths) != 1:
            raise ValueError(f"Expected exactly one raw device CSV for {case.name}")
        with paths[0].open(newline="", encoding="utf-8") as source:
            samples = extract_samples(csv.DictReader(source), require_candidate=True)
        measured = summarize(samples, require_candidate=True)
        if set(measured) != {case.name} or set(measured[case.name]) != set(IMPLEMENTATIONS):
            raise ValueError(f"CSV case/implementations differ from the probe: {case.name}")
        for implementation in IMPLEMENTATIONS:
            expected_count = capture_timings[case.name][implementation]["count"]
            indices = sorted(sample["sample"] for sample in samples if sample["implementation"] == implementation)
            if indices != list(range(expected_count)):
                raise ValueError(f"CSV sample indices/count differ from the probe: {case.name}")
            if any(
                not math.isfinite(value) or value <= 0
                for value in measured[case.name][implementation]["kernel_sums_us"]
            ):
                raise ValueError(f"Device sample sums must be finite and positive: {case.name}")
        device_ids.update(sample["device"] for sample in samples)
        device[case.name] = measured[case.name]
        evidence.extend(file_evidence(path, directory) for path in (probe_path, paths[0]))
    if len(device_ids) != 1:
        raise ValueError("Device ID differs between sweep captures")
    comparisons = {}
    non_improving = []
    for case in SWEEP_CASES:
        host_ratio = host[case.name]["public_repeat"]["median"] / host[case.name]["direct_codegen"]["median"]
        device_ratio = (
            device[case.name]["public_repeat"]["median_kernel_sum_us"]
            / device[case.name]["direct_codegen"]["median_kernel_sum_us"]
        )
        if not math.isfinite(host_ratio) or not math.isfinite(device_ratio):
            raise ValueError(f"Nonfinite comparison ratio for {case.name}")
        comparisons[case.name] = {
            "specification": describe_case(case),
            "host_ratio": host_ratio,
            "device_ratio": device_ratio,
        }
        for metric, ratio in (("host", host_ratio), ("device", device_ratio)):
            if ratio <= 1:
                non_improving.append({"case": case.name, "metric": metric, "ratio": ratio})
    return {
        "coverage_complete": True,
        "all_cases_improve": not non_improving,
        "non_improving": non_improving,
        "directory": str(directory),
        "identity": identity,
        "pytest": counts,
        "comparisons": comparisons,
        "host": host,
        "device": device,
        "raw_artifacts": evidence,
        "automatic_subtile_routing": False,
        "scope": "One bounded performance screen, not repeatability or statistical proof for the expanded matrix",
    }


def render(result):
    identity = result["identity"]
    lines = [
        "# Repeat candidate broader performance screen",
        "",
        f"Coverage: **COMPLETE**, {len(result['comparisons'])} cases; {result['pytest']['tests']} device tests passed.",
        f"All cases improve both medians: **{'YES' if result['all_cases_improve'] else 'NO'}**.",
        f"Commit: `{identity['git_commit']}`. Device: `{identity['architecture']}`, grid `{identity['compute_grid']}`.",
        f"Artifacts: `{result['directory']}`. Manifest SHA-256: `{identity['sweep_manifest_sha256']}`.",
        "",
        "Host latency is unprofiled and synchronized; device time is the sum of profiled kernel durations per call.",
        "Ratios are public/candidate medians. Above 1 means a lower candidate median. No cases or outliers are dropped.",
        "",
        "| Dtype | Cases | Both medians improve | Host non-improvements | Device non-improvements |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dtype in DTYPES:
        values = [value for value in result["comparisons"].values() if value["specification"]["dtype"] == dtype]
        wins = sum(value["host_ratio"] > 1 and value["device_ratio"] > 1 for value in values)
        host_losses = sum(value["host_ratio"] <= 1 for value in values)
        device_losses = sum(value["device_ratio"] <= 1 for value in values)
        lines.append(f"| {dtype} | {len(values)} | {wins} | {host_losses} | {device_losses} |")
    lines.extend(
        [
            "",
            "## Every measured case",
            "",
            "Times are microseconds, public / candidate. Exact shape, axis, repeats, output pages, sample counts and min/max ranges are retained in SWEEP.json; raw samples remain in the hashed artifacts.",
            "",
            "| Case | Host medians | Host ratio | Device kernel-sum medians | Device ratio | Both improve |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for case, comparison in result["comparisons"].items():
        host = " / ".join(f"{result['host'][case][name]['median']:.3f}" for name in IMPLEMENTATIONS)
        device = " / ".join(f"{result['device'][case][name]['median_kernel_sum_us']:.3f}" for name in IMPLEMENTATIONS)
        host_ratio, device_ratio = comparison["host_ratio"], comparison["device_ratio"]
        status = "yes" if host_ratio > 1 and device_ratio > 1 else "NO"
        lines.append(f"| {case} | {host} | {host_ratio:.3f}x | {device} | {device_ratio:.3f}x | {status} |")
    lines.extend(
        [
            "",
            "Automatic public H/W routing remains disabled; this report does not authorize a global route change.",
            "This is one screen of the fixed matrix, not repeated expanded-matrix evidence or proof for unmeasured inputs.",
            "Investigate non-improvements and repeat proposed route coverage before considering a narrowly scoped gate.",
            "Blackhole, installed-wheel packaging, project CI and model-level performance remain separate gates.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json-output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        result = load_sweep(arguments.directory)
        if arguments.json_output:
            with arguments.json_output.open("x") as output:
                json.dump(result, output, indent=2, allow_nan=False)
                output.write("\n")
        print(render(result), end="")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, ElementTree.ParseError) as error:
        print(f"Sweep report refused: {error}", file=sys.stderr)
        return 2
    return 0 if result["all_cases_improve"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
