"""Checkpoint-free screening of two unconfirmed TTNN performance opportunities."""

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


RESEARCH_COMMIT = "9173350554b616022b3aa7c4fbad33f13cb36aee"
PROBE_REVISION = "repeat-selector-batches-v3-tracy"
SOURCE_PATHS = (
    "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/repeat_interleave.cpp",
    "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.cpp",
    "ttnn/cpp/ttnn/operations/conv/conv_transpose2d/conv_transpose2d.cpp",
)


@dataclass(frozen=True)
class RepeatCase:
    name: str
    shape: tuple
    dimension: int
    repeats: int


REPEAT_CASES = (
    RepeatCase("aligned_h", (1, 4, 64, 128), 2, 2),
    RepeatCase("aligned_w", (1, 4, 64, 128), 3, 2),
    RepeatCase("gva_decode_h", (1, 1, 8, 64), 2, 4),
    RepeatCase("gva_prefill_h", (1, 128, 8, 64), 2, 4),
    RepeatCase("ragged_w", (1, 2, 37, 65), 3, 3),
    RepeatCase("outer_control", (1, 8, 128, 64), 1, 4),
)


def selector_coordinates(length, repeats, left_multiply):
    if length < 1 or repeats < 1:
        raise ValueError("length and repeats must be positive")
    output_indices = list(range(length * repeats))
    input_indices = [output_index // repeats for output_index in output_indices]
    if left_multiply:
        return output_indices, input_indices
    return input_indices, output_indices


def build_selector(torch, case, dtype):
    if len(case.shape) < 2 or case.dimension not in (len(case.shape) - 2, len(case.shape) - 1):
        raise ValueError("a selector requires one of the last two input dimensions")
    left_multiply = case.dimension == len(case.shape) - 2
    length = case.shape[case.dimension]
    row_indices, column_indices = selector_coordinates(length, case.repeats, left_multiply)
    matrix_shape = (length * case.repeats, length) if left_multiply else (length, length * case.repeats)
    selector = torch.zeros((*case.shape[:-2], *matrix_shape), dtype=dtype)
    selector[..., row_indices, column_indices] = 1
    return selector


def timing_summary(samples):
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("timings must be finite positive microseconds")
    median = statistics.median(samples)
    return {
        "samples_us": samples,
        "median_us": median,
        "min_us": min(samples),
        "max_us": max(samples),
        "spread_over_median": (max(samples) - min(samples)) / median,
    }


def output_extent(input_extent, kernel, stride, padding, output_padding, dilation=1):
    return (input_extent - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1


def command_result(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip()[:4000],
            "stderr": result.stderr.strip()[:1000],
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def environment_metadata():
    repository = command_result(["git", "rev-parse", "--show-toplevel"])
    source_hashes = {}
    if repository.get("returncode") == 0:
        root = Path(repository["stdout"])
        for relative_path in SOURCE_PATHS:
            source_path = root / relative_path
            source_hashes[relative_path] = (
                hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path.is_file() else "missing"
            )
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "git_head": command_result(["git", "rev-parse", "HEAD"]),
        "tracked_changes": command_result(["git", "diff", "HEAD", "--stat"]),
        "research_commit": RESEARCH_COMMIT,
        "probe_revision": PROBE_REVISION,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "tt_smi_version": command_result(["tt-smi", "--version"]),
        "execution_environment": {
            name: os.environ.get(name)
            for name in (
                "TT_VISIBLE_DEVICES",
                "ARCH_NAME",
                "TT_METAL_SLOW_DISPATCH_MODE",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_PROFILER_DIR",
                "TT_METAL_DPRINT_CORES",
                "TT_METAL_WATCHER",
                "TT_METAL_SIMULATOR",
                "TTNN_CONFIG_OVERRIDES",
                "TRACY_PORT",
            )
        },
    }


class Reporter:
    def __init__(self, output):
        self.path = Path(output).expanduser().resolve()
        self.handle = self.path.open("x", encoding="utf-8")

    def emit(self, event, **fields):
        record = {"event": event, **fields}
        encoded = json.dumps(record, allow_nan=False)
        self.handle.write(encoded + "\n")
        self.handle.flush()
        print(encoded, flush=True)

    def close(self):
        self.handle.close()


class Runtime:
    def __init__(self, device_id, reporter, tracy_signposts=False):
        self.reporter = reporter
        self.tracy_signposts = tracy_signposts
        self.device_profiling = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
        if tracy_signposts and not self.device_profiling:
            raise RuntimeError("--tracy-signposts requires a device-profiled run; launch with python -m tracy -p -r")
        if self.device_profiling:
            conflicts = [name for name in ("TT_METAL_DPRINT_CORES", "TT_METAL_WATCHER") if os.environ.get(name)]
            if conflicts:
                raise RuntimeError(f"Device profiling conflicts with {', '.join(conflicts)}; leave the device unopened")
        self.torch = importlib.import_module("torch")
        self.ttnn = importlib.import_module("ttnn")
        if tracy_signposts and not callable(getattr(self.ttnn, "tracy_message", None)):
            raise RuntimeError("This TTNN installation does not expose tracy_message; cannot label the capture")
        runtime_config = {
            name: getattr(self.ttnn.CONFIG, name, None)
            for name in (
                "enable_comparison_mode",
                "enable_logging",
                "enable_graph_report",
                "enable_tensor_report",
                "enable_fast_runtime_mode",
            )
        }
        reporter.emit(
            "modules",
            torch_version=str(self.torch.__version__),
            ttnn_version=str(getattr(self.ttnn, "__version__", "not exposed")),
            ttnn_file=str(self.ttnn.__file__),
            runtime_config=runtime_config,
            device_profiling=self.device_profiling,
            tracy_signposts=self.tracy_signposts,
        )
        if runtime_config["enable_comparison_mode"]:
            raise RuntimeError("TTNN comparison mode is enabled; CPU golden work would contaminate timings")
        self.device = self.ttnn.open_device(device_id=device_id, l1_small_size=64 * 1024)

    def describe_device(self):
        self.device.enable_program_cache()
        grid = self.device.compute_with_storage_grid_size()
        self.reporter.emit(
            "device",
            architecture=str(self.device.arch()),
            compute_grid={"x": grid.x, "y": grid.y},
            l1_small_size=64 * 1024,
            program_cache_enabled=True,
            hardware_execution="user-run; simulator environment must be checked separately",
        )

    def upload(self, host_tensor, dtype, layout=None):
        return self.ttnn.from_torch(
            host_tensor,
            dtype=dtype,
            layout=self.ttnn.TILE_LAYOUT if layout is None else layout,
            device=self.device,
            memory_config=self.ttnn.DRAM_MEMORY_CONFIG,
        )

    def release(self, tensor):
        self.ttnn.deallocate(tensor)

    def metrics(self, actual, expected, absolute_tolerance=0.0, relative_tolerance=0.0):
        torch = self.torch
        if tuple(actual.shape) != tuple(expected.shape):
            return {
                "shape_match": False,
                "actual_shape": list(actual.shape),
                "expected_shape": list(expected.shape),
                "allclose": False,
                "exact": False,
            }
        actual_values = actual.to(torch.float64).reshape(-1)
        expected_values = expected.to(torch.float64).reshape(-1)
        if not bool(torch.isfinite(expected_values).all()):
            raise ValueError("These probes require a finite reference")
        finite = torch.isfinite(actual_values)
        difference = torch.abs(actual_values - expected_values)
        threshold = absolute_tolerance + relative_tolerance * torch.abs(expected_values)
        bad = (~finite) | (difference > threshold)
        ranking = torch.where(finite, difference, torch.full_like(difference, float("inf")))
        worst_index = int(torch.argmax(ranking).item())
        all_finite = bool(finite.all())
        denominator = float(torch.linalg.vector_norm(expected_values).item())
        error_norm = float(torch.linalg.vector_norm(difference).item()) if all_finite else None
        relative_l2 = error_norm / denominator if all_finite and denominator > 0 else None
        actual_worst = float(actual_values[worst_index].item())
        return {
            "shape_match": True,
            "elements": actual_values.numel(),
            "exact": bool(torch.equal(actual_values, expected_values)),
            "allclose": not bool(bad.any()),
            "mismatched_elements": int((actual_values != expected_values).sum().item()),
            "outside_tolerance": int(bad.sum().item()),
            "nonfinite_outputs": int((~finite).sum().item()),
            "max_abs_error": float(difference.max().item()) if all_finite else None,
            "relative_l2_error": relative_l2,
            "atol": absolute_tolerance,
            "rtol": relative_tolerance,
            "worst_flat_index": worst_index,
            "worst_expected": float(expected_values[worst_index].item()),
            "worst_actual": actual_worst if math.isfinite(actual_worst) else str(actual_worst),
        }

    @contextmanager
    def profile_region(self, case_name, implementation, phase, iteration):
        if not self.tracy_signposts:
            yield
            return
        label = f"TT_BOUNTY/{case_name}/{implementation}/{phase}/{iteration}"
        self.ttnn.tracy_message(f"`TT_SIGNPOST: {label}/begin`")
        try:
            yield
        except BaseException:
            self.ttnn.tracy_message(f"`TT_SIGNPOST: {label}/aborted`")
            raise
        else:
            self.ttnn.tracy_message(f"`TT_SIGNPOST: {label}/end`")

    def benchmark(self, operations, warmup, samples, case_name="unspecified"):
        names = list(operations)
        measurements = {name: [] for name in names}
        for warmup_index in range(warmup):
            for name, operation in operations.items():
                if self.tracy_signposts:
                    self.ttnn.synchronize_device(self.device)
                with self.profile_region(case_name, name, "warmup", warmup_index):
                    result = operation()
                    self.ttnn.synchronize_device(self.device)
                self.release(result)
        for sample_index in range(samples):
            order = names if sample_index % 2 == 0 else list(reversed(names))
            for name in order:
                self.ttnn.synchronize_device(self.device)
                with self.profile_region(case_name, name, "sample", sample_index):
                    started = time.perf_counter_ns()
                    result = operations[name]()
                    self.ttnn.synchronize_device(self.device)
                    measurements[name].append((time.perf_counter_ns() - started) / 1000.0)
                self.release(result)
        return {name: timing_summary(values) for name, values in measurements.items()}


def run_repeat_case(runtime, case, arguments):
    torch, ttnn = runtime.torch, runtime.ttnn
    reporter = runtime.reporter
    reporter.emit("case_start", suite="repeat", case=asdict(case), dtype=arguments.dtype)
    generator = torch.Generator().manual_seed(20260920)
    host_dtype = torch.bfloat16 if arguments.dtype == "bf16" else torch.float32
    device_dtype = ttnn.bfloat16 if arguments.dtype == "bf16" else ttnn.float32
    host_input = torch.randint(-64, 65, case.shape, generator=generator).to(host_dtype) / 16.0
    expected = torch.repeat_interleave(host_input, case.repeats, dim=case.dimension)
    device_input = runtime.upload(host_input, device_dtype)
    transfer_metrics = runtime.metrics(ttnn.to_torch(device_input), host_input)
    reporter.emit("input_roundtrip", case=case.name, metrics=transfer_metrics)
    if not transfer_metrics["exact"]:
        raise RuntimeError("Input transfer differs; do not attribute this result to repeat_interleave")

    def native_repeat():
        return ttnn.repeat_interleave(
            device_input, case.repeats, dim=case.dimension, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    native_output = native_repeat()
    native_metrics = runtime.metrics(ttnn.to_torch(native_output), expected)
    runtime.release(native_output)
    reporter.emit("correctness", case=case.name, implementation="public_repeat", metrics=native_metrics)
    operations = {"public_repeat": native_repeat} if native_metrics["exact"] else {}
    device_selector = None
    selector_elements = 0
    if case.dimension >= len(case.shape) - 2 and not arguments.only_native:
        left_multiply = case.dimension == len(case.shape) - 2
        selector = build_selector(torch, case, host_dtype)
        selector_elements = selector.numel()
        device_selector = runtime.upload(selector, device_dtype)
        reporter.emit(
            "selector_setup",
            case=case.name,
            logical_shape=list(selector.shape),
            logical_elements=selector_elements,
            batch_strategy="materialized selectors matching every input batch dimension",
            setup_excluded_from_warm_timings=True,
        )
        fp32 = arguments.dtype == "fp32"
        fidelity = ttnn.MathFidelity.HiFi3 if fp32 and ttnn.is_wormhole_b0(runtime.device) else ttnn.MathFidelity.HiFi4
        compute_config = ttnn.init_device_compute_kernel_config(
            runtime.device.arch(), math_fidelity=fidelity, fp32_dest_acc_en=fp32, packer_l1_acc=False
        )

        def selector_matmul():
            first, second = (device_selector, device_input) if left_multiply else (device_input, device_selector)
            return ttnn.matmul(
                first, second, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=compute_config
            )

        selector_output = selector_matmul()
        selector_metrics = runtime.metrics(ttnn.to_torch(selector_output), expected)
        runtime.release(selector_output)
        reporter.emit("correctness", case=case.name, implementation="selector_matmul", metrics=selector_metrics)
        if selector_metrics["exact"]:
            operations["selector_matmul"] = selector_matmul
    timings = runtime.benchmark(operations, arguments.warmup, arguments.samples, case_name=case.name) if operations else {}
    post_timing_checks = {}
    for name, operation in operations.items():
        checked_output = operation()
        post_timing_checks[name] = runtime.metrics(ttnn.to_torch(checked_output), expected)
        runtime.release(checked_output)
    speed_ratio = None
    if (
        "public_repeat" in timings
        and "selector_matmul" in timings
        and all(check["exact"] for check in post_timing_checks.values())
    ):
        speed_ratio = timings["public_repeat"]["median_us"] / timings["selector_matmul"]["median_us"]
    reporter.emit(
        "repeat_result",
        case=case.name,
        timings=timings,
        post_timing_checks=post_timing_checks,
        public_over_selector_latency_ratio=speed_ratio,
        selector_logical_elements=selector_elements,
        selector_setup_excluded_from_warm_timings=True,
        device_profiling=runtime.device_profiling,
        tracy_signposts=runtime.tracy_signposts,
        profiling_caution="When device_profiling is true, these host timings are not an unprofiled baseline",
        comparison="same output; dense-selector diagnostic, not a production kernel or speedup guarantee",
        timing_scope="synchronized host-to-completion operation latency; not isolated device cycles",
    )
    runtime.release(device_input)
    if device_selector is not None:
        runtime.release(device_selector)


def run_grouped_conv(runtime, arguments):
    torch, ttnn = runtime.torch, runtime.ttnn
    channels, kernel, stride, padding, output_padding = 64, 3, 2, 1, 1
    spatial, groups = arguments.spatial, arguments.groups
    runtime.reporter.emit("case_start", suite="conv", groups=groups, channels=channels, spatial=spatial)
    generator = torch.Generator().manual_seed(20260920)
    host_input = torch.randn((1, channels, spatial, spatial), generator=generator).to(torch.bfloat16)
    host_weight = (
        torch.randn((channels, channels // groups, kernel, kernel), generator=generator)
        / math.sqrt((channels // groups) * kernel * kernel)
    ).to(torch.bfloat16)
    expected = torch.nn.functional.conv_transpose2d(
        host_input.double(), host_weight.double(), stride=stride, padding=padding,
        output_padding=output_padding, groups=groups
    )
    device_input = runtime.upload(host_input.permute(0, 2, 3, 1).contiguous(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    host_tt_weight = ttnn.from_torch(host_weight, dtype=ttnn.bfloat16)
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        deallocate_activation=False,
        enable_act_double_buffer=False,
        output_layout=ttnn.TILE_LAYOUT,
        config_tensors_in_dram=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        runtime.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False, packer_l1_acc=False
    )
    parameters = {
        "input_tensor": device_input,
        "device": runtime.device,
        "in_channels": channels,
        "out_channels": channels,
        "batch_size": 1,
        "input_height": spatial,
        "input_width": spatial,
        "kernel_size": (kernel, kernel),
        "stride": (stride, stride),
        "padding": (padding, padding),
        "output_padding": (output_padding, output_padding),
        "dilation": (1, 1),
        "groups": groups,
        "dtype": ttnn.bfloat16,
        "conv_config": conv_config,
        "compute_config": compute_config,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "mirror_kernel": True,
    }
    started = time.perf_counter_ns()
    first_output, dimensions, prepared = ttnn.conv_transpose2d(
        weight_tensor=host_tt_weight, return_output_dim=True, return_weights_and_bias=True, **parameters
    )
    ttnn.synchronize_device(runtime.device)
    cold_us = (time.perf_counter_ns() - started) / 1000.0
    output_height, output_width = [int(value) for value in dimensions]
    expected_extent = output_extent(spatial, kernel, stride, padding, output_padding)
    if (output_height, output_width) != (expected_extent, expected_extent):
        raise RuntimeError(f"Unexpected output dimensions: {dimensions}; expected {expected_extent} squared")

    def read_output(tensor):
        return ttnn.to_torch(tensor).reshape(1, output_height, output_width, channels).permute(0, 3, 1, 2)

    first_metrics = runtime.metrics(read_output(first_output), expected, 0.03, 0.03)
    runtime.release(first_output)
    prepared_weight, prepared_bias = prepared
    runtime.reporter.emit("correctness", implementation="conv_first_call", groups=groups, metrics=first_metrics)

    def warm_conv():
        return ttnn.conv_transpose2d(weight_tensor=prepared_weight, bias_tensor=prepared_bias, **parameters)

    warm_output = warm_conv()
    warm_metrics = runtime.metrics(read_output(warm_output), expected, 0.03, 0.03)
    runtime.release(warm_output)
    runtime.reporter.emit("correctness", implementation="conv_reused_weights", groups=groups, metrics=warm_metrics)
    operations = (
        {"grouped_conv_transpose2d": warm_conv} if first_metrics["allclose"] and warm_metrics["allclose"] else {}
    )
    timings = runtime.benchmark(operations, arguments.warmup, arguments.samples) if operations else {}
    post_timing_metrics = None
    if operations:
        checked_output = warm_conv()
        post_timing_metrics = runtime.metrics(read_output(checked_output), expected, 0.03, 0.03)
        runtime.release(checked_output)
    prepared_shape = [int(value) for value in prepared_weight.padded_shape]
    runtime.reporter.emit(
        "conv_result",
        groups=groups,
        channels=channels,
        spatial=spatial,
        raw_weight_elements=host_weight.numel(),
        expanded_dense_elements=channels * channels * kernel * kernel,
        prepared_weight_logical_shape=[int(value) for value in prepared_weight.shape],
        prepared_weight_padded_shape=prepared_shape,
        prepared_weight_dtype=str(prepared_weight.dtype),
        prepared_weight_padded_elements=math.prod(prepared_shape),
        cold_preparation_and_first_call_us=cold_us,
        timings=timings,
        post_timing_metrics=post_timing_metrics,
        acceptance="screening tolerances only, not a bounty acceptance contract",
        config_tensors_in_dram=True,
        timing_scope="synchronized host-to-completion latency, prepared weights reused, no CPU golden in timed region",
        caution="groups=1 is a scaling control, not the same mathematical operator or a proposed fix",
    )
    runtime.release(prepared_weight)
    if prepared_bias is not None:
        runtime.release(prepared_bias)
    runtime.release(device_input)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("repeat", "conv"), default="repeat")
    parser.add_argument("--case", choices=tuple(case.name for case in REPEAT_CASES))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--groups", type=int, choices=(1, 4, 16, 64), default=1)
    parser.add_argument("--spatial", type=int, choices=(16, 32), default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--only-native", action="store_true")
    parser.add_argument("--tracy-signposts", action="store_true", help="Label warmups and samples in a Tracy device capture")
    parser.add_argument("--list", action="store_true", dest="list_only")
    parser.add_argument("--output")
    arguments = parser.parse_args(argv)
    if arguments.samples < 3 or arguments.warmup < 1:
        parser.error("Use at least 3 samples and 1 warm-up")
    if arguments.suite == "conv" and (arguments.case or arguments.dtype != "bf16" or arguments.only_native):
        parser.error("--case, --dtype fp32, and --only-native apply only to the repeat suite")
    if arguments.tracy_signposts and (
        arguments.suite != "repeat" or arguments.case is None or arguments.warmup > 10 or arguments.samples > 16
    ):
        parser.error("--tracy-signposts requires one repeat --case, at most 10 warmups, and at most 16 samples")
    return arguments


def main(argv=None):
    arguments = parse_arguments(argv)
    cases = [case for case in REPEAT_CASES if arguments.case is None or case.name == arguments.case]
    if arguments.list_only:
        print(
            json.dumps(
                {"suite": arguments.suite, "repeat_cases": [asdict(case) for case in cases], "args": vars(arguments)},
                indent=2,
            )
        )
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = arguments.output or f"tt-probe-{arguments.suite}-{stamp}-{os.getpid()}.jsonl"
    reporter = Reporter(output)
    runtime = None
    status = 0
    try:
        reporter.emit("environment", arguments=vars(arguments), **environment_metadata())
        runtime = Runtime(arguments.device, reporter, tracy_signposts=arguments.tracy_signposts)
        runtime.describe_device()
        if arguments.suite == "repeat":
            for case in cases:
                run_repeat_case(runtime, case, arguments)
        else:
            run_grouped_conv(runtime, arguments)
        reporter.emit("suite_finished", message="Read correctness and timing records; this is not bounty approval")
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
        print(f"Results: {reporter.path}", flush=True)
        reporter.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
