"""Compare the forced tiled-copy candidate with public repeat on the same current checkout."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

from hardware_probe import REPEAT_CASES, Reporter, Runtime, command_result
from repeat_sweep_cases import SWEEP_CASES, manifest, manifest_sha256

EXPECTED_BASE = "9173350554b616022b3aa7c4fbad33f13cb36aee"
REVISION = "repeat-direct-tile-candidate-v3-packed-rows"
SOURCE_PATHS = (
    "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/repeat_interleave.cpp",
    "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.cpp",
    "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.hpp",
    "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_program_factory.cpp",
    "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/codegen/kernels/reader_repeat_interleave_subtile.cpp",
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("focused", "sweep"), default="focused")
    parser.add_argument("--case", choices=[case.name for case in (*REPEAT_CASES, *SWEEP_CASES)])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=51)
    parser.add_argument("--tracy-signposts", action="store_true")
    parser.add_argument("--list", action="store_true", dest="list_only")
    parser.add_argument("--output")
    arguments = parser.parse_args(argv)
    available = SWEEP_CASES if arguments.suite == "sweep" else REPEAT_CASES
    if arguments.case is not None and arguments.case not in {case.name for case in available}:
        parser.error("--case must belong to the selected --suite")
    if arguments.warmup < 1 or arguments.samples < 3:
        parser.error("Use at least one warmup and three samples")
    if arguments.device < 0:
        parser.error("Device index must be nonnegative")
    profiling = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    if arguments.tracy_signposts and not profiling and not arguments.list_only:
        parser.error("--tracy-signposts requires python -m tracy -p -r")
    if (profiling or arguments.tracy_signposts) and (
        arguments.case is None or arguments.warmup > 10 or arguments.samples > 16
    ):
        parser.error("Device profiling requires one --case, at most ten warmups and sixteen samples")
    if profiling and not arguments.tracy_signposts:
        parser.error("Use --tracy-signposts so device measurements can be assigned to samples")
    if arguments.suite == "sweep" and not arguments.list_only:
        minimum_warmup, minimum_samples = (3, 5) if profiling else (10, 51)
        if arguments.warmup < minimum_warmup or arguments.samples < minimum_samples:
            parser.error(f"Sweep requires at least {minimum_warmup} warmups and {minimum_samples} samples")
    return arguments


def preflight():
    head = command_result(["git", "rev-parse", "HEAD"])
    if head["returncode"] != 0:
        raise RuntimeError("Run from the isolated candidate checkout, not an exported source directory")
    commit = head["stdout"].strip()
    if (
        commit != EXPECTED_BASE
        and command_result(["git", "merge-base", "--is-ancestor", EXPECTED_BASE, commit])["returncode"] != 0
    ):
        raise RuntimeError(
            f"Run from the isolated candidate checkout based on {EXPECTED_BASE}, with its Git history available; "
            "the earlier hardware checkout does not contain the codegen operation. Do not overwrite that installation."
        )
    for name in ("TT_METAL_SIMULATOR", "TT_METAL_EMULE_MODE", "TT_METAL_MOCK_DEVICE"):
        if os.environ.get(name) not in (None, "", "0"):
            raise RuntimeError(f"{name} is enabled; this probe records hardware measurements only")
    for name in ("TT_METAL_WATCHER", "TT_METAL_DPRINT_CORES", "TT_METAL_SLOW_DISPATCH_MODE"):
        if os.environ.get(name) not in (None, "", "0"):
            raise RuntimeError(f"{name} would change the performance configuration; run correctness checks separately")
    missing = [path for path in SOURCE_PATHS if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"Candidate sources missing: {missing}")
    return head


def bit_metrics(torch, actual, expected):
    shape_match = tuple(actual.shape) == tuple(expected.shape)
    dtype_match = actual.dtype == expected.dtype
    if not shape_match or not dtype_match:
        return {"shape_match": shape_match, "dtype_match": dtype_match, "exact_bits": False}
    bits_dtype = torch.int16 if expected.dtype == torch.bfloat16 else torch.int32
    actual_bits = actual.contiguous().view(bits_dtype)
    expected_bits = expected.contiguous().view(bits_dtype)
    mismatches = int((actual_bits != expected_bits).sum().item())
    return {
        "shape_match": True,
        "dtype_match": True,
        "exact_bits": mismatches == 0,
        "mismatched_elements": mismatches,
        "elements": expected.numel(),
    }


def make_input(torch, case):
    generator = torch.Generator().manual_seed(20260920)
    dtype = getattr(case, "dtype", "bf16")
    if dtype == "int32":
        return torch.randint(-(2**31), 2**31, case.shape, generator=generator, dtype=torch.int32)
    if dtype == "fp32":
        return torch.randint(-(2**23), 2**23, case.shape, generator=generator).to(torch.float32) / 2**23
    return torch.randint(-64, 65, case.shape, generator=generator).to(torch.bfloat16) / 16


def source_state():
    return {
        "git_head": command_result(["git", "rev-parse", "HEAD"]),
        "tracked_changes": command_result(["git", "status", "--short", "--untracked-files=no"]),
        "source_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in SOURCE_PATHS},
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing_helper_sha256": hashlib.sha256(Path(__file__).with_name("hardware_probe.py").read_bytes()).hexdigest(),
        "case_catalog_sha256": hashlib.sha256(
            Path(__file__).with_name("repeat_sweep_cases.py").read_bytes()
        ).hexdigest(),
    }


def require_clean_state(state):
    changes = state["tracked_changes"]
    if changes.get("returncode") != 0 or changes.get("stdout") != "":
        raise RuntimeError("Sweep requires a clean tracked checkout; preserve changes rather than resetting them")


def run_case(runtime, case, arguments):
    torch, ttnn = runtime.torch, runtime.ttnn
    reporter = runtime.reporter
    dtype = getattr(case, "dtype", "bf16")
    reporter.emit("case_start", case=asdict(case), dtype=dtype)
    source = make_input(torch, case)
    expected = torch.repeat_interleave(source, case.repeats, case.dimension)
    device_dtype = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32, "int32": ttnn.int32}[dtype]
    device_input = runtime.upload(source, device_dtype)
    private_ops = ttnn._ttnn.operations.data_movement
    forced = getattr(private_ops, "repeat_interleave_force_codegen", None)
    if not callable(forced):
        raise RuntimeError("This TTNN build lacks the forced codegen entry; build the candidate checkout first")

    def public_repeat():
        return ttnn.repeat_interleave(device_input, case.repeats, case.dimension, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def direct_codegen():
        return forced(device_input, case.repeats, case.dimension, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    operations = {"public_repeat": public_repeat, "direct_codegen": direct_codegen}
    roundtrip = bit_metrics(torch, ttnn.to_torch(device_input), source)
    reporter.emit("input_roundtrip", case=case.name, metrics=roundtrip)
    if not roundtrip["exact_bits"]:
        raise RuntimeError("Input transfer differs bitwise; operation timing is not a valid comparison")
    for name, operation in operations.items():
        output = operation()
        metrics = bit_metrics(torch, ttnn.to_torch(output), expected)
        runtime.release(output)
        reporter.emit("correctness", case=case.name, implementation=name, metrics=metrics)
        if not metrics["exact_bits"]:
            raise RuntimeError(f"{case.name}/{name} failed exact correctness; refusing to benchmark")
    timings = runtime.benchmark(operations, arguments.warmup, arguments.samples, case_name=case.name)
    for name, operation in operations.items():
        output = operation()
        metrics = bit_metrics(torch, ttnn.to_torch(output), expected)
        runtime.release(output)
        reporter.emit("post_timing_correctness", case=case.name, implementation=name, metrics=metrics)
        if not metrics["exact_bits"]:
            raise RuntimeError(f"{case.name}/{name} failed post-timing correctness; discard timings")
    reporter.emit(
        "candidate_result",
        case=case.name,
        timings=timings,
        public_over_candidate_ratio=timings["public_repeat"]["median_us"] / timings["direct_codegen"]["median_us"],
        device_profiling=runtime.device_profiling,
        tracy_signposts=runtime.tracy_signposts,
        timing_scope="synchronized host-to-completion latency, not isolated device time",
        comparison="same input/output on current source; no selector setup or dense matmul",
        automatic_subtile_routing="disabled pending device correctness and performance validation",
        control_case=case.name == "outer_control",
    )
    runtime.release(device_input)


def main(argv=None):
    arguments = parse_arguments(argv)
    available = SWEEP_CASES if arguments.suite == "sweep" else REPEAT_CASES
    cases = [case for case in available if arguments.case is None or case.name == arguments.case]
    if arguments.list_only:
        if arguments.suite == "sweep":
            print(json.dumps({**manifest(), "manifest_sha256": manifest_sha256()}, indent=2))
            return 0
        print(
            json.dumps(
                {"revision": REVISION, "base": EXPECTED_BASE, "cases": [asdict(case) for case in cases]}, indent=2
            )
        )
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reporter = Reporter(arguments.output or f"repeat-candidate-{stamp}-{os.getpid()}.jsonl")
    runtime = None
    status = 0
    try:
        preflight()
        state = source_state()
        if arguments.suite == "sweep":
            require_clean_state(state)
        reporter.emit(
            "environment",
            utc=datetime.now(timezone.utc).isoformat(),
            revision=REVISION,
            base_commit=EXPECTED_BASE,
            arguments=vars(arguments),
            **state,
            sweep_manifest_sha256=manifest_sha256() if arguments.suite == "sweep" else None,
            python=sys.version,
            python_executable=sys.executable,
            environment={
                name: os.environ.get(name)
                for name in (
                    "TT_VISIBLE_DEVICES",
                    "ARCH_NAME",
                    "TRACY_PORT",
                    "TT_METAL_DEVICE_PROFILER",
                    "TT_METAL_PROFILER_DIR",
                    "TT_METAL_SLOW_DISPATCH_MODE",
                    "TT_METAL_WATCHER",
                    "TT_METAL_DPRINT_CORES",
                    "TT_METAL_SIMULATOR",
                    "TT_METAL_EMULE_MODE",
                    "TT_METAL_MOCK_DEVICE",
                )
            },
        )
        ttnn_module = importlib.import_module("ttnn")
        if not Path(ttnn_module.__file__).resolve().is_relative_to(Path.cwd().resolve()):
            raise RuntimeError("TTNN was imported outside this checkout; activate its Python environment and rebuild")
        runtime = Runtime(arguments.device, reporter, tracy_signposts=arguments.tracy_signposts)
        if arguments.suite == "sweep":
            flags = ("enable_comparison_mode", "enable_logging", "enable_graph_report", "enable_tensor_report")
            if any(getattr(runtime.ttnn.CONFIG, flag, False) for flag in flags):
                raise RuntimeError("Disable TTNN comparison/logging/reporting before collecting sweep timings")
        runtime.describe_device()
        for case in cases:
            run_case(runtime, case, arguments)
        if arguments.suite == "sweep":
            final_state = source_state()
            require_clean_state(final_state)
            if final_state != state:
                raise RuntimeError("Source or benchmark tools changed during the sweep; discard this probe")
            reporter.emit("source_state_end", **final_state)
        reporter.emit("suite_finished", message="Candidate evidence only; not merge or bounty approval")
    except Exception as error:
        reporter.emit("error", error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        status = 1
    finally:
        if runtime is not None:
            try:
                runtime.ttnn.close_device(runtime.device)
                reporter.emit("device_closed")
            except Exception as error:
                reporter.emit("close_error", message=str(error))
                status = 1
        reporter.close()
        print(f"Results: {reporter.path}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
