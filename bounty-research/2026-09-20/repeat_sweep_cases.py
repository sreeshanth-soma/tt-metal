"""Fixed, bounded H/W coverage; no case selection based on observed performance."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from hardware_probe import REPEAT_CASES, RepeatCase


SCHEMA = "repeat-hw-performance-sweep-v1"
DTYPES = ("bf16", "fp32", "int32")
REPEATS = (2, 3, 4, 7, 8, 16, 33, 127)
MAX_OUTPUT_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class SweepCase:
    name: str
    shape: tuple
    dimension: int
    repeats: int
    dtype: str
    category: str


GEOMETRIES = (
    *((case, "original_target") for case in REPEAT_CASES if case.name != "outer_control"),
    *(
        (RepeatCase(f"rank2_{axis}_r{repeats}", shape, dimension, repeats), "ragged_repeat")
        for axis, shape, dimension in (("h", (17, 33), 0), ("w", (33, 17), 1))
        for repeats in REPEATS
    ),
    (RepeatCase("aligned_w_r4", (1, 4, 64, 128), 3, 4), "packed_width"),
    (RepeatCase("aligned_w_r8", (1, 4, 64, 128), 3, 8), "packed_width"),
    (RepeatCase("rank3_h", (3, 65, 37), 1, 7), "rank3"),
    (RepeatCase("rank3_w", (3, 37, 65), -1, 7), "rank3"),
    (RepeatCase("single_h", (1, 1), 0, 8), "single_element"),
    (RepeatCase("single_w", (1, 1), 1, 8), "single_element"),
    *((RepeatCase(f"pages_{pages}_h", (1, pages, 8, 32), 2, 4), "core_partition") for pages in (71, 72, 73)),
    (RepeatCase("scale_h_r4", (1, 16, 256, 256), 2, 4), "larger_output"),
    (RepeatCase("scale_w_r8", (1, 8, 256, 256), 3, 8), "larger_output"),
)
SWEEP_CASES = tuple(
    SweepCase(f"{dtype}_{case.name}", case.shape, case.dimension, case.repeats, dtype, category)
    for dtype in DTYPES
    for case, category in GEOMETRIES
)


def describe_case(case):
    description = asdict(case)
    description["shape"] = list(case.shape)
    output_shape = list(case.shape)
    output_shape[case.dimension] *= case.repeats
    output_pages = math.prod(output_shape[:-2]) * math.ceil(output_shape[-2] / 32) * math.ceil(output_shape[-1] / 32)
    output_bytes = output_pages * 1024 * (2 if case.dtype == "bf16" else 4)
    if output_bytes > MAX_OUTPUT_BYTES:
        raise ValueError(f"Sweep output exceeds the fixed memory bound: {case.name}")
    return {**description, "output_shape": output_shape, "output_pages": output_pages, "output_bytes": output_bytes}


def manifest():
    return {
        "schema": SCHEMA,
        "layout": "default 32x32 TILE",
        "memory": "interleaved DRAM input and output",
        "maximum_padded_output_bytes": MAX_OUTPUT_BYTES,
        "cases": [describe_case(case) for case in SWEEP_CASES],
    }


def manifest_sha256():
    return hashlib.sha256(json.dumps(manifest(), sort_keys=True).encode()).hexdigest()
