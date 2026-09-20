"""Host-only checks; these do not validate TTNN calls or device behavior."""

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import hardware_probe as probe


class ProbeHostTests(unittest.TestCase):
    def test_left_selector_repeats_rows(self):
        rows, columns = probe.selector_coordinates(3, 4, True)
        source = [11, 29, 47]
        output = [0] * 12
        for row, column in zip(rows, columns):
            output[row] = source[column]
        self.assertEqual(output, [11] * 4 + [29] * 4 + [47] * 4)

    def test_right_selector_repeats_columns(self):
        rows, columns = probe.selector_coordinates(3, 2, False)
        source = [7, 19, 31]
        output = [0] * 6
        for row, column in zip(rows, columns):
            output[column] = source[row]
        self.assertEqual(output, [7, 7, 19, 19, 31, 31])

    def test_selector_rejects_empty_inputs(self):
        for length, repeats in ((0, 1), (1, 0), (-1, 2)):
            with self.assertRaises(ValueError):
                probe.selector_coordinates(length, repeats, True)

    def test_selector_materializes_every_case_batch(self):
        expected_shapes = {
            "aligned_h": (1, 4, 128, 64),
            "aligned_w": (1, 4, 128, 256),
            "gva_decode_h": (1, 1, 32, 8),
            "gva_prefill_h": (1, 128, 32, 8),
            "ragged_w": (1, 2, 65, 195),
        }
        for case in probe.REPEAT_CASES:
            if case.name not in expected_shapes:
                continue
            with self.subTest(case=case.name):
                torch = MagicMock()
                selector = probe.build_selector(torch, case, "bf16")
                expected_shape = expected_shapes[case.name]
                torch.zeros.assert_called_once_with(expected_shape, dtype="bf16")
                self.assertIs(selector, torch.zeros.return_value)
                self.assertEqual(expected_shape[:-2], case.shape[:-2])
                self.assertLessEqual(math.prod(expected_shape), 262144)
                rows, columns = probe.selector_coordinates(
                    case.shape[case.dimension], case.repeats, case.dimension == len(case.shape) - 2
                )
                selector.__setitem__.assert_called_once_with((Ellipsis, rows, columns), 1)

    def test_selector_preserves_multiple_nonunit_batch_dimensions(self):
        for dimension, expected_shape in ((2, (2, 3, 8, 4)), (3, (2, 3, 5, 10))):
            with self.subTest(dimension=dimension):
                torch = MagicMock()
                case = probe.RepeatCase("multi_batch", (2, 3, 4, 5), dimension, 2)
                probe.build_selector(torch, case, "fp32")
                torch.zeros.assert_called_once_with(expected_shape, dtype="fp32")

    def test_selector_rejects_outer_axis(self):
        torch = MagicMock()
        case = probe.RepeatCase("outer", (1, 8, 128, 64), 1, 4)
        with self.assertRaises(ValueError):
            probe.build_selector(torch, case, "bf16")
        torch.zeros.assert_not_called()

    def test_timing_summary(self):
        summary = probe.timing_summary([9.0, 11.0, 10.0])
        self.assertEqual(summary["median_us"], 10.0)
        self.assertAlmostEqual(summary["spread_over_median"], 0.2)

    def test_timing_summary_rejects_invalid_samples(self):
        for values in ([], [0], [-1], [math.nan], [math.inf]):
            with self.assertRaises(ValueError):
                probe.timing_summary(values)

    def test_conv_dimensions(self):
        self.assertEqual(probe.output_extent(16, 3, 2, 1, 1), 32)
        self.assertEqual(probe.output_extent(32, 3, 2, 1, 1), 64)

    def test_case_matrix_is_bounded(self):
        self.assertEqual(len({case.name for case in probe.REPEAT_CASES}), 6)
        for case in probe.REPEAT_CASES:
            self.assertLessEqual(math.prod(case.shape) * case.repeats, 262144)
            self.assertGreaterEqual(case.dimension, 0)
            self.assertLess(case.dimension, len(case.shape))

    def test_list_needs_no_torch_or_device(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(probe.main(["--list", "--case", "gva_decode_h"]), 0)
        document = json.loads(output.getvalue())
        self.assertEqual(document["repeat_cases"][0]["shape"], [1, 1, 8, 64])

    def test_invalid_suite_options_are_rejected(self):
        for arguments in (["--samples", "0"], ["--suite", "conv", "--dtype", "fp32"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                probe.parse_arguments(arguments)

    def test_profile_options_require_a_bounded_single_case(self):
        rejected = (
            ["--tracy-signposts"],
            ["--tracy-signposts", "--suite", "conv"],
            ["--tracy-signposts", "--case", "aligned_w", "--samples", "17"],
            ["--tracy-signposts", "--case", "aligned_w", "--warmup", "11"],
        )
        for arguments in rejected:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                probe.parse_arguments(arguments)
        arguments = probe.parse_arguments(["--tracy-signposts", "--case", "aligned_w", "--samples", "5"])
        self.assertTrue(arguments.tracy_signposts)

    def test_profile_checks_run_before_imports_or_device_open(self):
        environments = (
            {},
            {"TT_METAL_DEVICE_PROFILER": "1", "TT_METAL_DPRINT_CORES": "all"},
            {"TT_METAL_DEVICE_PROFILER": "1", "TT_METAL_WATCHER": "120"},
        )
        for environment in environments:
            with self.subTest(environment=environment), patch.dict(probe.os.environ, environment, clear=True):
                with patch.object(probe.importlib, "import_module") as import_module:
                    with self.assertRaises(RuntimeError):
                        probe.Runtime(0, MagicMock(), tracy_signposts=True)
                    import_module.assert_not_called()

    def test_profile_regions_mark_failures_without_success_end(self):
        runtime = probe.Runtime.__new__(probe.Runtime)
        runtime.tracy_signposts = True
        runtime.ttnn = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "operation failed"):
            with runtime.profile_region("aligned_w", "public_repeat", "sample", 0):
                raise RuntimeError("operation failed")
        self.assertEqual(
            [call.args[0] for call in runtime.ttnn.tracy_message.call_args_list],
            [
                "`TT_SIGNPOST: TT_BOUNTY/aligned_w/public_repeat/sample/0/begin`",
                "`TT_SIGNPOST: TT_BOUNTY/aligned_w/public_repeat/sample/0/aborted`",
            ],
        )

    def test_benchmark_labels_samples_and_keeps_alternating_order(self):
        runtime = probe.Runtime.__new__(probe.Runtime)
        runtime.tracy_signposts = True
        runtime.ttnn = MagicMock()
        runtime.device = object()
        runtime.release = MagicMock()
        timeline = []
        runtime.ttnn.tracy_message.side_effect = lambda message: timeline.append(message)
        runtime.ttnn.synchronize_device.side_effect = lambda device: timeline.append("sync")
        operations = {
            name: MagicMock(side_effect=lambda name=name: timeline.append(f"operation:{name}"))
            for name in ("public_repeat", "selector_matmul")
        }
        with patch.object(probe.time, "perf_counter_ns", side_effect=range(1000, 13000, 1000)):
            timings = runtime.benchmark(operations, warmup=1, samples=3, case_name="aligned_w")
        expected_calls = [
            ("warmup", 0, "public_repeat"), ("warmup", 0, "selector_matmul"),
            ("sample", 0, "public_repeat"), ("sample", 0, "selector_matmul"),
            ("sample", 1, "selector_matmul"), ("sample", 1, "public_repeat"),
            ("sample", 2, "public_repeat"), ("sample", 2, "selector_matmul"),
        ]
        expected_timeline = []
        for phase, iteration, name in expected_calls:
            label = f"TT_BOUNTY/aligned_w/{name}/{phase}/{iteration}"
            expected_timeline.extend([
                "sync", f"`TT_SIGNPOST: {label}/begin`", f"operation:{name}", "sync",
                f"`TT_SIGNPOST: {label}/end`",
            ])
        self.assertEqual(timeline, expected_timeline)
        for result in timings.values():
            self.assertEqual(result["samples_us"], [1.0, 1.0, 1.0])
        self.assertEqual(runtime.release.call_count, 8)

    def test_disabled_profile_regions_emit_no_markers(self):
        runtime = probe.Runtime.__new__(probe.Runtime)
        runtime.tracy_signposts = False
        runtime.ttnn = MagicMock()
        with runtime.profile_region("aligned_w", "public_repeat", "sample", 0):
            pass
        runtime.ttnn.tracy_message.assert_not_called()

    def test_reporter_flushes_and_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result.jsonl"
            reporter = probe.Reporter(destination)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    reporter.emit("sample", value=3)
                self.assertEqual(json.loads(destination.read_text())["value"], 3)
                with self.assertRaises(FileExistsError):
                    probe.Reporter(destination)
            finally:
                reporter.close()


if __name__ == "__main__":
    unittest.main()
