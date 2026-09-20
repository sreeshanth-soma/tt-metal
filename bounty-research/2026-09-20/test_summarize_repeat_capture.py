import unittest

from summarize_repeat_capture import extract_samples, summarize


def marker(implementation, phase, index, boundary):
    return {"OP CODE": f"TT_BOUNTY/aligned_w/{implementation}/{phase}/{index}/{boundary}", "OP TYPE": "signpost"}


def operation(duration, code="RepeatInterleaveCodegenDeviceOperation", device="0"):
    return {
        "OP CODE": code,
        "OP TYPE": "tt_dnn_device",
        "DEVICE ID": device,
        "DEVICE KERNEL DURATION [ns]": str(duration),
        "CORE COUNT": "64",
        "DATA MOVEMENT KERNEL SOURCE": "reader_repeat_interleave_subtile.cpp",
    }


def capture():
    rows = [operation(999999)]
    rows.extend(
        [marker("public_repeat", "warmup", 0, "begin"), operation(999999), marker("public_repeat", "warmup", 0, "end")]
    )
    for index in range(3):
        rows.extend(
            [
                marker("public_repeat", "sample", index, "begin"),
                operation(1000 + index),
                operation(2000 + index),
                marker("public_repeat", "sample", index, "end"),
                marker("direct_codegen", "sample", index, "begin"),
                operation(1500 + index),
                marker("direct_codegen", "sample", index, "end"),
            ]
        )
    return rows


class CaptureSummaryTests(unittest.TestCase):
    def test_sums_each_measured_call_before_taking_median(self):
        samples = extract_samples(capture(), require_candidate=True)
        self.assertEqual(len(samples), 6)
        summary = summarize(samples, require_candidate=True)["aligned_w"]
        self.assertEqual(summary["public_repeat"]["median_kernel_sum_us"], 3.002)
        self.assertEqual(summary["direct_codegen"]["median_kernel_sum_us"], 1.501)
        self.assertEqual(summary["direct_codegen"]["public_over_alternative_kernel_sum_ratio"], 2)

    def test_rejects_unclosed_or_aborted_regions(self):
        with self.assertRaisesRegex(ValueError, "inside"):
            extract_samples(capture()[:-1])
        with self.assertRaisesRegex(ValueError, "aborted"):
            extract_samples([marker("direct_codegen", "sample", 0, "aborted")])

    def test_rejects_multiple_devices_in_a_region(self):
        rows = [
            marker("public_repeat", "sample", 0, "begin"),
            operation(1),
            operation(2, device="1"),
            marker("public_repeat", "sample", 0, "end"),
        ]
        with self.assertRaisesRegex(ValueError, "multiple devices"):
            extract_samples(rows)

    def test_requires_the_actual_new_kernel(self):
        rows = capture()
        rows[-2]["DATA MOVEMENT KERNEL SOURCE"] = "reader_bmm_tile_layout.cpp"
        with self.assertRaisesRegex(ValueError, "new tiled-copy reader"):
            extract_samples(rows, require_candidate=True)

    def test_rejects_invalid_duration(self):
        rows = [marker("public_repeat", "sample", 0, "begin"), operation("nan")]
        with self.assertRaisesRegex(ValueError, "invalid"):
            extract_samples(rows)

    def test_rejects_mismatched_sample_indices(self):
        samples = extract_samples(capture())
        samples[-1]["sample"] = 9
        with self.assertRaisesRegex(ValueError, "mismatched"):
            summarize(samples)

    def test_rejects_comparison_between_devices(self):
        samples = extract_samples(capture())
        for sample in samples:
            if sample["implementation"] == "direct_codegen":
                sample["device"] = "1"
        with self.assertRaisesRegex(ValueError, "different devices"):
            summarize(samples)


if __name__ == "__main__":
    unittest.main()
