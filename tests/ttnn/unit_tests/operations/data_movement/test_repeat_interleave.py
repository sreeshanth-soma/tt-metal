# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("repeats", [1, 2, 3, 58])
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.uint16])
def test_repeat_interleave(device, repeats, dim, dtype):
    if dtype == ttnn.uint16:
        torch_dtype = torch.int16
        torch_input_tensor = torch.randint(0, 100, (1, 1, 32, 32), dtype=torch_dtype)
    else:
        torch_dtype = torch.bfloat16
        torch_input_tensor = torch.rand(1, 1, 32, 32, dtype=torch_dtype)

    torch_result = torch.repeat_interleave(torch_input_tensor, repeats, dim=dim)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output = ttnn.repeat_interleave(input_tensor, repeats, dim=dim)
    output = ttnn.to_torch(output)
    assert_equal(torch_result, output)


# --- Codegen-path coverage ---
#
# ttnn.repeat_interleave routes gate-supported cases to codegen and the rest to native, and offers
# no way to ask for one: the verification-only entries below live in the private module for that
# reason (see repeat_interleave_force.hpp). These pin the codegen path so the suite exercises it
# regardless of the gate's verdict.
#
_force_native = ttnn._ttnn.operations.data_movement.repeat_interleave_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.repeat_interleave_force_codegen

_CODEGEN_CASES = [
    # (shape, repeats, dim, layout)
    ((2, 3, 32, 64), 2, 1, ttnn.TILE_LAYOUT),
    ((2, 3, 32, 64), 3, 0, ttnn.TILE_LAYOUT),
    ((2, 3, 32, 64), 2, -3, ttnn.TILE_LAYOUT),
    ((2, 3, 32, 64), 2, 1, ttnn.ROW_MAJOR_LAYOUT),
    ((2, 3, 32, 64), 3, 0, ttnn.ROW_MAJOR_LAYOUT),
    ((4, 6, 8), 2, 1, ttnn.ROW_MAJOR_LAYOUT),
]
_CODEGEN_CASE_IDS = [
    "[2, 3, 32, 64]|dim=1&repeats=2|tile",
    "[2, 3, 32, 64]|dim=0&repeats=3|tile",
    "[2, 3, 32, 64]|dim=-3&repeats=2|tile",
    "[2, 3, 32, 64]|dim=1&repeats=2|row_major",
    "[2, 3, 32, 64]|dim=0&repeats=3|row_major",
    "[4, 6, 8]|dim=1&repeats=2|row_major",
]
_CODEGEN_DTYPES = [ttnn.bfloat16, ttnn.float32, ttnn.int32]
_CODEGEN_DTYPE_IDS = ["bfloat16", "float32", "int32"]


def _codegen_input(shape, dtype):
    if dtype == ttnn.int32:
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


@pytest.mark.parametrize("dtype", _CODEGEN_DTYPES, ids=_CODEGEN_DTYPE_IDS)
@pytest.mark.parametrize("shape,repeats,dim,layout", _CODEGEN_CASES, ids=_CODEGEN_CASE_IDS)
def test_repeat_interleave_codegen(device, shape, repeats, dim, layout, dtype):
    """Bit-exactness against the implementation codegen replaces, on the same input."""
    input_tensor = ttnn.from_torch(_codegen_input(shape, dtype), layout=layout, dtype=dtype, device=device)
    golden = ttnn.to_torch(_force_native(input_tensor, repeats, dim))
    assert_equal(golden, ttnn.to_torch(_force_codegen(input_tensor, repeats, dim)))


@pytest.mark.parametrize("shape,repeats,dim,layout", _CODEGEN_CASES, ids=_CODEGEN_CASE_IDS)
def test_pc_repeat_interleave_codegen(device, shape, repeats, dim, layout):
    """A second dispatch of the same spec must reuse the cached program with the new buffers.

    The descriptor factory hands raw Buffer*s to emplace_runtime_args, so a cache hit relies on the
    framework re-resolving those bindings; a stale binding would read or write the first
    invocation's allocation. Cache identity does not vary with dtype, so one is enough here.
    """
    dtype = ttnn.bfloat16
    first = ttnn.from_torch(_codegen_input(shape, dtype), layout=layout, dtype=dtype, device=device)
    first_golden = ttnn.to_torch(_force_native(first, repeats, dim))
    assert_equal(first_golden, ttnn.to_torch(_force_codegen(first, repeats, dim)))
    entries_after_miss = device.num_program_cache_entries()

    # A distinct allocation with the same spec: same program hash, different Buffer*.
    second = ttnn.from_torch(_codegen_input(shape, dtype), layout=layout, dtype=dtype, device=device)
    second_golden = ttnn.to_torch(_force_native(second, repeats, dim))
    assert_equal(second_golden, ttnn.to_torch(_force_codegen(second, repeats, dim)))
    msg = "second codegen dispatch missed the program cache"
    assert device.num_program_cache_entries() == entries_after_miss, msg


_SUBTILE_CASES = [
    pytest.param((1, 4, 64, 128), 2, -2, id="aligned_h"),
    pytest.param((1, 4, 64, 128), 2, -1, id="aligned_w"),
    pytest.param((1, 1, 8, 64), 4, 2, id="decode_h"),
    pytest.param((1, 128, 8, 64), 4, 2, id="prefill_h"),
    pytest.param((1, 2, 37, 65), 3, 3, id="ragged_w"),
    pytest.param((2, 3, 37, 65), 3, 2, id="ragged_h"),
    pytest.param((3, 5), 33, 0, id="rank2_h_large_repeat"),
    pytest.param((5, 3), 33, 1, id="rank2_w_large_repeat"),
    pytest.param((3, 65, 37), 7, 1, id="rank3_h"),
    pytest.param((3, 37, 65), 7, -1, id="rank3_w"),
    pytest.param((1, 2, 17, 33), 2, 2, id="face_boundary_h"),
    pytest.param((2, 3, 1, 1), 5, -1, id="single_element_w"),
]
_SUBTILE_TORCH_DTYPES = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.int32: torch.int32}


def _subtile_input(shape, dtype, seed=0):
    values = torch.arange(math.prod(shape), dtype=torch.int64) * 2654435761 + seed * 2246822519
    if dtype == ttnn.bfloat16:
        bits = ((values & 0x807F) | 0x3F00).to(torch.int16)
    elif dtype == ttnn.float32:
        bits = ((values & 0x807FFFFF) | 0x3F000000).to(torch.int32)
    else:
        bits = values.to(torch.int32)
    return bits.view(_SUBTILE_TORCH_DTYPES[dtype]).reshape(shape)


def _assert_subtile_bits(expected, actual):
    assert expected.shape == actual.shape
    assert expected.dtype == actual.dtype
    bits_dtype = torch.int16 if expected.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(expected.contiguous().view(bits_dtype), actual.contiguous().view(bits_dtype))


@pytest.mark.parametrize("dtype", _CODEGEN_DTYPES, ids=_CODEGEN_DTYPE_IDS)
@pytest.mark.parametrize("shape,repeats,dim", _SUBTILE_CASES)
def test_repeat_interleave_codegen_subtile(device, shape, repeats, dim, dtype):
    source = _subtile_input(shape, dtype)
    input_tensor = ttnn.from_torch(
        source, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    _assert_subtile_bits(source, ttnn.to_torch(input_tensor))
    output = _force_codegen(input_tensor, repeats, dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert output.layout == ttnn.TILE_LAYOUT
    assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    _assert_subtile_bits(torch.repeat_interleave(source, repeats, dim), ttnn.to_torch(output))


@pytest.mark.parametrize("dtype", _CODEGEN_DTYPES, ids=_CODEGEN_DTYPE_IDS)
@pytest.mark.parametrize("shape,repeats,dim", _SUBTILE_CASES)
def test_pc_repeat_interleave_codegen_subtile(device, shape, repeats, dim, dtype):
    sources = [_subtile_input(shape, dtype, seed=seed) for seed in (1, 2)]
    inputs = [ttnn.from_torch(source, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device) for source in sources]
    assert inputs[0].buffer_address() != inputs[1].buffer_address()
    first_output = _force_codegen(inputs[0], repeats, dim)
    _assert_subtile_bits(torch.repeat_interleave(sources[0], repeats, dim), ttnn.to_torch(first_output))
    entries_after_miss = device.num_program_cache_entries()
    second_output = _force_codegen(inputs[1], repeats, dim)
    assert first_output.buffer_address() != second_output.buffer_address()
    _assert_subtile_bits(torch.repeat_interleave(sources[1], repeats, dim), ttnn.to_torch(second_output))
    _assert_subtile_bits(torch.repeat_interleave(sources[0], repeats, dim), ttnn.to_torch(first_output))
    assert device.num_program_cache_entries() == entries_after_miss


@pytest.mark.parametrize("dim", [-2, -1])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bfloat16", "float32"])
def test_repeat_interleave_codegen_subtile_special_bits(device, dim, dtype):
    if dtype == ttnn.bfloat16:
        patterns = [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x7F7F, 0x7F80, 0xFF80, 0x7FC1, 0xFFC2, 0x7F81]
        bits = torch.tensor(patterns, dtype=torch.int64).to(torch.int16)
    else:
        patterns = [
            0x00000000,
            0x80000000,
            0x00000001,
            0x80000001,
            0x007FFFFF,
            0x00800000,
            0x7F7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC12345,
            0xFFC23456,
            0x7F812345,
        ]
        bits = torch.tensor(patterns, dtype=torch.int64).to(torch.int32)
    shape = (2, 3, 37, 65)
    indices = torch.arange(math.prod(shape)) % len(patterns)
    source = bits[indices].view(_SUBTILE_TORCH_DTYPES[dtype]).reshape(shape)
    input_tensor = ttnn.from_torch(source, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    _assert_subtile_bits(source, ttnn.to_torch(input_tensor))
    output = _force_codegen(input_tensor, 3, dim)
    _assert_subtile_bits(torch.repeat_interleave(source, 3, dim), ttnn.to_torch(output))


@pytest.mark.skip(reason="ttnn.repeat_interleave only supports `repeats` as int")
def test_repeat_interleave_with_repeat_tensor(device):
    torch_input_tensor = torch.rand(1, 2, 32, 32, dtype=torch.bfloat16)
    torch_repeats = torch.tensor([1, 2])
    torch_result = torch.repeat_interleave(torch_input_tensor, torch_repeats, dim=1)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    repeats = ttnn.from_torch(torch_repeats)
    output = ttnn.repeat_interleave(input_tensor, repeats, dim=1)
    output = ttnn.to_torch(output)

    assert_equal(torch_result, output)
