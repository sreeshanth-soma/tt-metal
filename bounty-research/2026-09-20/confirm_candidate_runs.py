"""Check repeatability across complete hardware runs without pooling samples or enabling routing."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ElementTree

from report_candidate_run import IMPLEMENTATIONS, REPEAT_CASES, load_run


def confirm(directories):
    paths = [directory.resolve() for directory in directories]
    if len(paths) < 3 or len(set(paths)) != len(paths):
        raise ValueError("Need at least three distinct run directories, not reused paths")
    runs = [load_run(directory) for directory in paths]
    identity = runs[0]["identity"]
    counts = runs[0]["pytest"]
    seen_starts = set()
    for index, run in enumerate(runs, 1):
        if run["identity"] != identity:
            raise ValueError(f"Commit, source, runtime or device differs in run {index}")
        if run["pytest"] != counts or run["pytest"]["skipped"]:
            raise ValueError("Confirmation requires the same executed device tests and zero skips in every run")
        for probe in run["probes"]:
            changes = probe["tracked_changes"]
            if not isinstance(changes, dict) or changes.get("returncode") != 0 or changes.get("stdout") != "":
                raise ValueError("Every probe must report a clean tracked Git checkout")
            stamp = probe["utc"]
            if not isinstance(stamp, str):
                raise ValueError("Every probe needs a timezone-aware UTC start timestamp")
            try:
                started = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError("Invalid probe start timestamp") from error
            if started.tzinfo is None:
                raise ValueError("Every probe needs a timezone-aware UTC start timestamp")
            started = started.astimezone(timezone.utc)
            if started in seen_starts:
                raise ValueError("Duplicate probe starts: copied artifacts are not independent runs")
            seen_starts.add(started)
        for case, timings in run["host"].items():
            for implementation in IMPLEMENTATIONS:
                count = timings[implementation]["count"]
                if count < 51 or count != runs[0]["host"][case][implementation]["count"]:
                    raise ValueError("Use at least 51 host samples per leg and equal counts across runs")
        for case, timings in run["device"].items():
            for implementation in IMPLEMENTATIONS:
                count = timings[implementation]["samples"]
                if count < 5 or count != runs[0]["device"][case][implementation]["samples"]:
                    raise ValueError("Use at least five device samples per leg and equal counts across runs")
    cases = {}
    failures = []
    for case in REPEAT_CASES:
        host_ratios = [
            run["host"][case.name]["public_repeat"]["median"] / run["host"][case.name]["direct_codegen"]["median"]
            for run in runs
        ]
        device_ratios = (
            []
            if case.name == "outer_control"
            else [
                run["device"][case.name]["public_repeat"]["median_kernel_sum_us"]
                / run["device"][case.name]["direct_codegen"]["median_kernel_sum_us"]
                for run in runs
            ]
        )
        cases[case.name] = {"host_ratios": host_ratios, "device_ratios": device_ratios}
        if case.name == "outer_control":
            continue
        for metric, ratios in (("host", host_ratios), ("device", device_ratios)):
            for index, ratio in enumerate(ratios, 1):
                if ratio <= 1:
                    failures.append(f"{case.name}: run {index} {metric} ratio {ratio:.3f}x is not an improvement")
    return {
        "passed": not failures,
        "failures": failures,
        "cases": cases,
        "runs": runs,
        "automatic_subtile_routing": False,
        "scope": "Observed repeatability for these inputs only; not statistical proof, routing approval or bounty eligibility",
    }


def render(result):
    runs = result["runs"]
    identity = runs[0]["identity"]
    lines = [
        "# Repeat candidate confirmation",
        "",
        f"Result: **{'PASS' if result['passed'] else 'NOT CONFIRMED'}** across {len(runs)} complete runs.",
        f"Commit: `{identity['git_commit']}`. Device: `{identity['architecture']}`, grid `{identity['compute_grid']}`.",
        "",
    ]
    for index, run in enumerate(runs, 1):
        lines.append(f"- Run {index}: {run['pytest']['tests']} device tests passed; artifacts: `{run['directory']}`.")
    lines.extend(
        [
            "",
            "Ratios are public/candidate medians, in run order; above 1 means a lower candidate median.",
            "Host and device metrics remain separate. Samples are not pooled and no run or outlier is discarded.",
            "",
            "| Case | Unprofiled host ratios | Profiled kernel-sum ratios | Every target comparison improves |",
            "| --- | --- | --- | --- |",
        ]
    )
    for case, ratios in result["cases"].items():
        host = ", ".join(f"{ratio:.3f}x" for ratio in ratios["host_ratios"])
        device = ", ".join(f"{ratio:.3f}x" for ratio in ratios["device_ratios"]) or "not captured"
        wins = all(ratio > 1 for ratio in ratios["host_ratios"] + ratios["device_ratios"])
        status = "control; excluded" if case == "outer_control" else "yes" if wins else "NO"
        lines.append(f"| {case} | {host} | {device} | {status} |")
    if result["failures"]:
        lines.extend(["", "## Comparisons needing investigation", ""])
        lines.extend(f"- {failure}" for failure in result["failures"])
    lines.extend(
        [
            "",
            "Every probe has a distinct start, a clean tracked checkout and matching commit/source/runtime/device identity.",
            "Each leg has at least 51 host samples or five device samples, with equal counts across runs.",
            "See each run's SUMMARY.md and raw artifacts for medians, ranges and retained outliers; CONFIRMATION.json retains all validated run summaries.",
            "Automatic H/W routing remains disabled. This is repeatability evidence for these cases, not proof for other shapes/dtypes/architectures.",
            "Broader performance coverage, routing validation, project CI and human review still precede a production claim.",
            "This report does not establish model-level improvement, maintainer acceptance or bounty eligibility.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--json-output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        result = confirm(arguments.directories)
        if arguments.json_output is not None:
            with arguments.json_output.open("x") as output:
                json.dump(result, output, indent=2, allow_nan=False)
                output.write("\n")
    except (OSError, ValueError, KeyError, TypeError, ElementTree.ParseError) as error:
        print(f"Confirmation refused: {error}", file=sys.stderr)
        return 2
    print(render(result), end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
