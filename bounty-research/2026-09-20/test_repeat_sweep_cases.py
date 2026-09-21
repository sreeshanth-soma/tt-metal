from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import repeat_candidate_probe as probe
import repeat_sweep_cases as sweep


class SweepCaseTests(unittest.TestCase):
    def test_fixed_matrix_covers_axes_ranks_repeats_and_dtypes(self):
        self.assertEqual(len(sweep.GEOMETRIES), 32)
        self.assertEqual(len(sweep.SWEEP_CASES), 96)
        self.assertEqual(len({case.name for case in sweep.SWEEP_CASES}), 96)
        for dtype in sweep.DTYPES:
            cases = [case for case in sweep.SWEEP_CASES if case.dtype == dtype]
            self.assertEqual({len(case.shape) for case in cases}, {2, 3, 4})
            for offset in (1, 2):
                repeats = {
                    case.repeats for case in cases if case.dimension % len(case.shape) == len(case.shape) - offset
                }
                self.assertEqual(repeats, set(sweep.REPEATS))
            self.assertTrue(all(case.dimension % len(case.shape) >= len(case.shape) - 2 for case in cases))
        for case in sweep.SWEEP_CASES:
            self.assertRegex(case.name, r"^[a-z0-9_]+$")

    def test_padded_memory_bound_and_grid_edge_outputs(self):
        records = sweep.manifest()["cases"]
        self.assertEqual(max(record["output_bytes"] for record in records), 16 * 1024 * 1024)
        self.assertTrue(all(record["output_bytes"] > 0 for record in records))
        for pages in (71, 72, 73):
            case = next(record for record in records if record["name"] == f"bf16_pages_{pages}_h")
            self.assertEqual(case["output_pages"], pages)
        oversized = sweep.SweepCase("too_large", (1, 16, 256, 256), 2, 127, "fp32", "fixture")
        with self.assertRaisesRegex(ValueError, "memory bound"):
            sweep.describe_case(oversized)

    def test_manifest_is_deterministic_and_list_needs_no_runtime(self):
        output = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(probe, "Runtime") as runtime:
            with mock.patch.object(probe, "preflight") as preflight, redirect_stdout(output):
                self.assertEqual(probe.main(["--suite", "sweep", "--list"]), 0)
        runtime.assert_not_called()
        preflight.assert_not_called()
        self.assertEqual(
            json.loads(output.getvalue()), {**sweep.manifest(), "manifest_sha256": sweep.manifest_sha256()}
        )

    def test_case_selection_and_minimum_samples_are_enforced(self):
        with mock.patch.dict(os.environ, {}, clear=True), redirect_stderr(io.StringIO()):
            for arguments in (
                ["--suite", "sweep", "--case", "aligned_h"],
                ["--case", "bf16_aligned_h"],
                ["--suite", "sweep", "--samples", "50"],
                ["--suite", "sweep", "--warmup", "9"],
            ):
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                    probe.parse_arguments(arguments)
        with mock.patch.dict(os.environ, {"TT_METAL_DEVICE_PROFILER": "1"}, clear=True):
            arguments = probe.parse_arguments(
                ["--suite", "sweep", "--case", "int32_rank3_w", "--warmup", "3", "--samples", "5", "--tracy-signposts"]
            )
        self.assertEqual(arguments.case, "int32_rank3_w")

    def test_dtype_specific_inputs_do_not_convert_integer_values_to_float(self):
        for dtype in sweep.DTYPES:
            torch = mock.MagicMock()
            case = next(case for case in sweep.SWEEP_CASES if case.dtype == dtype)
            source = probe.make_input(torch, case)
            if dtype == "int32":
                self.assertIs(source, torch.randint.return_value)
                self.assertEqual(torch.randint.call_args.args[:2], (-(2**31), 2**31))
                self.assertIs(torch.randint.call_args.kwargs["dtype"], torch.int32)
                torch.randint.return_value.to.assert_not_called()
            else:
                torch.randint.return_value.to.assert_called_once_with(
                    torch.bfloat16 if dtype == "bf16" else torch.float32
                )
                divisor = 16 if dtype == "bf16" else 2**23
                torch.randint.return_value.to.return_value.__truediv__.assert_called_once_with(divisor)

    def test_bit_metrics_use_one_integer_word_per_element(self):
        torch = SimpleNamespace(bfloat16="bf16", int16="int16", int32="int32")
        for dtype in sweep.DTYPES:
            expected, actual = mock.MagicMock(), mock.MagicMock()
            expected.shape = actual.shape = (2, 3)
            expected.dtype = actual.dtype = dtype
            expected.numel.return_value = 6
            differences = mock.MagicMock()
            differences.sum.return_value.item.return_value = 2
            actual.contiguous.return_value.view.return_value.__ne__.return_value = differences
            result = probe.bit_metrics(torch, actual, expected)
            bits_dtype = "int16" if dtype == "bf16" else "int32"
            actual.contiguous.return_value.view.assert_called_once_with(bits_dtype)
            expected.contiguous.return_value.view.assert_called_once_with(bits_dtype)
            self.assertEqual(result["mismatched_elements"], 2)
            self.assertEqual(result["elements"], 6)
            self.assertFalse(result["exact_bits"])

    def test_dirty_checkout_is_rejected_before_importing_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "probe.jsonl"
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(probe, "preflight"):
                with mock.patch.object(
                    probe,
                    "source_state",
                    return_value={"tracked_changes": {"returncode": 0, "stdout": " M source.cpp"}},
                ):
                    with mock.patch.object(probe, "Runtime") as runtime, redirect_stdout(io.StringIO()):
                        self.assertEqual(probe.main(["--suite", "sweep", "--output", str(output)]), 1)
            runtime.assert_not_called()
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([record["event"] for record in records], ["error"])
            self.assertIn("clean tracked checkout", records[0]["message"])

    def test_probe_checks_final_sources_and_closes_the_device(self):
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "probe.jsonl"
                state = {"tracked_changes": {"returncode": 0, "stdout": ""}, "probe_sha256": "0" * 64}
                end_state = {**state, "probe_sha256": "1" * 64} if changed else state
                module = SimpleNamespace(
                    __file__=str(Path.cwd() / "ttnn/fixture.py"), CONFIG=SimpleNamespace(), close_device=mock.Mock()
                )
                runtime = mock.Mock(ttnn=module)
                with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(probe, "preflight"):
                    with mock.patch.object(probe, "source_state", side_effect=[state, end_state]), mock.patch.object(
                        probe.importlib, "import_module", return_value=module
                    ):
                        with mock.patch.object(probe, "Runtime", return_value=runtime), mock.patch.object(
                            probe, "run_case"
                        ) as run_case, redirect_stdout(io.StringIO()):
                            status = probe.main(
                                ["--suite", "sweep", "--case", "bf16_aligned_h", "--output", str(output)]
                            )
                self.assertEqual(status, int(changed))
                run_case.assert_called_once()
                module.close_device.assert_called_once_with(runtime.device)
                events = [json.loads(line)["event"] for line in output.read_text().splitlines()]
                self.assertEqual("source_state_end" in events, not changed)
                self.assertEqual("suite_finished" in events, not changed)
                self.assertEqual("error" in events, changed)

    def test_instrumented_runtime_is_rejected_before_benchmarking(self):
        with tempfile.TemporaryDirectory() as temporary:
            module = SimpleNamespace(
                __file__=str(Path.cwd() / "ttnn/fixture.py"),
                CONFIG=SimpleNamespace(enable_logging=True),
                close_device=mock.Mock(),
            )
            runtime = mock.Mock(ttnn=module)
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(probe, "preflight"):
                with mock.patch.object(
                    probe, "source_state", return_value={"tracked_changes": {"returncode": 0, "stdout": ""}}
                ), mock.patch.object(probe.importlib, "import_module", return_value=module):
                    with mock.patch.object(probe, "Runtime", return_value=runtime), mock.patch.object(
                        probe, "run_case"
                    ) as run_case, redirect_stdout(io.StringIO()):
                        status = probe.main(["--suite", "sweep", "--output", str(Path(temporary) / "probe.jsonl")])
            self.assertEqual(status, 1)
            run_case.assert_not_called()
            module.close_device.assert_called_once_with(runtime.device)


if __name__ == "__main__":
    unittest.main()
