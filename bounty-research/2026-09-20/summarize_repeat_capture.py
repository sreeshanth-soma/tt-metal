"""Summarize measured repeat signpost regions, never warmups or individual core durations."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def extract_samples(rows, require_candidate=False):
    samples = []
    active = None
    operations = []
    for row in rows:
        code = row.get("OP CODE", "")
        if row.get("OP TYPE") == "signpost" and code.startswith("TT_BOUNTY/"):
            parts = code.split("/")
            if len(parts) != 6:
                raise ValueError(f"Malformed repeat signpost: {code}")
            case, implementation, phase, index, boundary = parts[1:]
            key = (case, implementation, phase, int(index))
            if phase not in ("sample", "warmup"):
                raise ValueError(f"Unknown repeat phase: {phase}")
            if boundary == "begin":
                if active is not None:
                    raise ValueError("Nested or unclosed repeat signposts")
                active, operations = key, []
            elif boundary == "end":
                if key != active:
                    raise ValueError("Repeat signpost end has no matching begin")
                if phase == "sample":
                    if not operations:
                        raise ValueError("Measured region contains no device operations")
                    device_ids = {operation["device"] for operation in operations}
                    if len(device_ids) != 1:
                        raise ValueError("Cannot sum multiple devices into one operation latency")
                    if require_candidate and implementation == "direct_codegen":
                        if len(operations) != 1:
                            raise ValueError("Direct candidate must emit exactly one device operation")
                        if (
                            case != "outer_control"
                            and "reader_repeat_interleave_subtile.cpp" not in operations[0]["source"]
                        ):
                            raise ValueError("Candidate trace does not contain the new tiled-copy reader")
                    samples.append(
                        {
                            "case": case,
                            "implementation": implementation,
                            "sample": int(index),
                            "device": next(iter(device_ids)),
                            "kernel_sum_ns": sum(operation["duration_ns"] for operation in operations),
                            "operations": operations,
                        }
                    )
                active, operations = None, []
            else:
                raise ValueError(f"Capture contains an aborted or unknown boundary: {code}")
        elif active is not None and active[2] == "sample" and row.get("OP TYPE") == "tt_dnn_device":
            duration = float(row.get("DEVICE KERNEL DURATION [ns]", ""))
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("Missing or invalid device kernel duration")
            device_id = row.get("DEVICE ID", "")
            if not device_id:
                raise ValueError("Missing device ID in measured region")
            operations.append(
                {
                    "code": code,
                    "device": device_id,
                    "duration_ns": duration,
                    "cores": row.get("CORE COUNT", ""),
                    "source": row.get("DATA MOVEMENT KERNEL SOURCE", ""),
                }
            )
    if active is not None:
        raise ValueError("Capture ends inside a repeat region")
    if not samples:
        raise ValueError("No measured repeat sample regions found")
    return samples


def summarize(samples, require_candidate=False):
    grouped = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        grouped[sample["case"]][sample["implementation"]].append(sample)
    result = {}
    for case, implementations in grouped.items():
        if len({sample["device"] for records in implementations.values() for sample in records}) != 1:
            raise ValueError("Implementations were measured on different devices")
        if "public_repeat" not in implementations:
            raise ValueError(f"Missing public baseline for {case}")
        if require_candidate and "direct_codegen" not in implementations:
            raise ValueError(f"Missing direct candidate for {case}")
        baseline_indices = sorted(sample["sample"] for sample in implementations["public_repeat"])
        if len(baseline_indices) < 3 or len(set(baseline_indices)) != len(baseline_indices):
            raise ValueError("Need at least three distinct measured samples per implementation")
        summaries = {}
        for implementation, records in implementations.items():
            if sorted(sample["sample"] for sample in records) != baseline_indices:
                raise ValueError("Implementations have mismatched sample indices")
            totals = [sample["kernel_sum_ns"] for sample in records]
            median_ns = statistics.median(totals)
            if median_ns <= 0:
                raise ValueError("Zero device duration is not a usable performance baseline")
            summaries[implementation] = {
                "samples": len(records),
                "kernel_sums_us": [duration / 1000 for duration in totals],
                "median_kernel_sum_us": median_ns / 1000,
                "operations_per_sample": [len(sample["operations"]) for sample in records],
                "operation_sequence": [operation["code"] for operation in records[0]["operations"]],
                "first_sample_core_counts": [operation["cores"] for operation in records[0]["operations"]],
            }
        public_median = summaries["public_repeat"]["median_kernel_sum_us"]
        for implementation, summary in summaries.items():
            if implementation != "public_repeat":
                summary["public_over_alternative_kernel_sum_ratio"] = public_median / summary["median_kernel_sum_us"]
        result[case] = summaries
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--require-candidate", action="store_true")
    arguments = parser.parse_args(argv)
    with arguments.csv.open(newline="", encoding="utf-8") as source:
        samples = extract_samples(csv.DictReader(source), require_candidate=arguments.require_candidate)
    print(
        json.dumps(
            {
                "source": str(arguments.csv.resolve()),
                "source_sha256": hashlib.sha256(arguments.csv.read_bytes()).hexdigest(),
                "measurement": "median of per-call sums of DEVICE KERNEL DURATION [ns]; not end-to-end latency",
                "warmups_and_unlabelled_operations_excluded": True,
                "results": summarize(samples, require_candidate=arguments.require_candidate),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
