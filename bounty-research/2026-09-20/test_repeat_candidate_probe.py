from contextlib import redirect_stdout
import io
import json
import os
import unittest
from unittest import mock

import repeat_candidate_probe as probe


class CandidateProbeTests(unittest.TestCase):
    def test_list_does_not_open_a_device(self):
        output = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(probe, "Runtime") as runtime:
            with redirect_stdout(output):
                self.assertEqual(probe.main(["--list"]), 0)
        runtime.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertEqual(result["base"], probe.EXPECTED_BASE)
        self.assertEqual(len(result["cases"]), 6)

    def test_profile_requires_bounded_labelled_single_case(self):
        with mock.patch.dict(os.environ, {"TT_METAL_DEVICE_PROFILER": "1"}, clear=True):
            for arguments in ([], ["--case", "aligned_w"], ["--tracy-signposts", "--samples", "5"]):
                with self.subTest(arguments=arguments), redirect_stdout(io.StringIO()):
                    with mock.patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
                        probe.parse_arguments(arguments)

    def test_profile_accepts_explicit_small_capture(self):
        with mock.patch.dict(os.environ, {"TT_METAL_DEVICE_PROFILER": "1"}, clear=True):
            arguments = probe.parse_arguments(
                ["--case", "gva_prefill_h", "--warmup", "3", "--samples", "5", "--tracy-signposts"]
            )
        self.assertEqual(arguments.case, "gva_prefill_h")
        self.assertEqual(arguments.samples, 5)

    def test_rejects_old_hardware_checkout_before_runtime(self):
        old_head = {"returncode": 0, "stdout": "9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9", "stderr": ""}
        unrelated = {"returncode": 1, "stdout": "", "stderr": ""}
        with mock.patch.object(probe, "command_result", side_effect=[old_head, unrelated]):
            with self.assertRaisesRegex(RuntimeError, "isolated candidate checkout"):
                probe.preflight()

    def test_accepts_committed_candidate_descendant(self):
        candidate_head = {"returncode": 0, "stdout": "a" * 40, "stderr": ""}
        ancestor = {"returncode": 0, "stdout": "", "stderr": ""}
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(probe.Path, "is_file", return_value=True):
            with mock.patch.object(probe, "command_result", side_effect=[candidate_head, ancestor]) as command:
                self.assertEqual(probe.preflight(), candidate_head)
        self.assertEqual(
            command.call_args_list[-1], mock.call(["git", "merge-base", "--is-ancestor", probe.EXPECTED_BASE, "a" * 40])
        )

    def test_rejects_missing_candidate_sources_on_descendant(self):
        candidate_head = {"returncode": 0, "stdout": "a" * 40, "stderr": ""}
        ancestor = {"returncode": 0, "stdout": "", "stderr": ""}
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(probe.Path, "is_file", return_value=False):
            with mock.patch.object(probe, "command_result", side_effect=[candidate_head, ancestor]):
                with self.assertRaisesRegex(RuntimeError, "Candidate sources missing"):
                    probe.preflight()

    def test_rejects_emulation_and_debug_timings(self):
        current_head = {"returncode": 0, "stdout": probe.EXPECTED_BASE, "stderr": ""}
        for variable in (
            "TT_METAL_EMULE_MODE",
            "TT_METAL_SIMULATOR",
            "TT_METAL_WATCHER",
            "TT_METAL_SLOW_DISPATCH_MODE",
        ):
            with self.subTest(variable=variable), mock.patch.object(probe, "command_result", return_value=current_head):
                with mock.patch.dict(os.environ, {variable: "1"}, clear=True):
                    with self.assertRaisesRegex(RuntimeError, variable):
                        probe.preflight()


if __name__ == "__main__":
    unittest.main()
