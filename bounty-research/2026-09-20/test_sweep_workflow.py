import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_candidate_workflow import executable


HERE = Path(__file__).resolve().parent


class SweepWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tt-sweep workflow-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        scripts = self.root / "bounty-research/2026-09-20"
        scripts.mkdir(parents=True)
        self.runner = scripts / "run_repeat_performance_sweep.sh"
        shutil.copyfile(HERE / self.runner.name, self.runner)
        commands = self.root / "mock-bin"
        commands.mkdir()
        executable(commands / "uname", "printf 'Linux\n'\n")
        binaries = self.root / "python_env/bin"
        binaries.mkdir(parents=True)
        (binaries / "activate").write_text(
            'export VIRTUAL_ENV="$PYTHON_ENV_DIR"\nexport PATH="$VIRTUAL_ENV/bin:$PATH"\n'
        )
        executable(
            binaries / "python",
            """printf "%s\n" "$*" >> "$CALL_LOG"
printf "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n" "$VIRTUAL_ENV" "$TT_METAL_HOME" "$TT_METAL_RUNTIME_ROOT" "$LD_LIBRARY_PATH" "$PYTHONPATH" "$PYTHONNOUSERSITE" "${PYTHONHOME-unset}" "${TT_METAL_DEVICE_PROFILER-unset}" "${TT_METAL_PROFILER_DIR-unset}" "${TRACY_PORT-unset}" >> "$ENV_LOG"
if [[ "$1" == "-c" ]]; then
    if [[ "$2" == *"preflight"* ]]; then exit "$PREFLIGHT_STATUS"; fi
    printf 'bf16_aligned_h\nfp32_aligned_h\nint32_aligned_h\n'
elif [[ "$1" == "-m" && "$2" == "pytest" ]]; then
    echo 'fixture correctness log'
    exit "$PYTEST_STATUS"
elif [[ "$1" == "-m" && "$2" == "tracy" ]]; then
    if [[ "$*" == *"--case $FAIL_CASE "* ]]; then echo 'fixture capture failure'; exit 37; fi
elif [[ "$1" == *"repeat_candidate_probe.py" ]]; then
    if [[ "$*" == *"--list"* ]]; then printf '{"cases": []}\n'; else exit "$HOST_STATUS"; fi
elif [[ "$1" == *"report_repeat_sweep.py" ]]; then
    echo 'fixture sweep report'
    exit "$REPORT_STATUS"
else
    exit 99
fi
""",
        )
        executable(self.root / "build_metal.sh", 'echo "fixture build log"\nexit "$BUILD_STATUS"\n')
        self.environment = {
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "TMPDIR": str(self.root),
            "VIRTUAL_ENV": "/old/python_env",
            "PYTHON_ENV_DIR": "/old/python_env",
            "LD_LIBRARY_PATH": "/old/build/lib",
            "PYTHONPATH": "/old/ttnn",
            "PYTHONHOME": "/old",
            "TT_METAL_DEVICE_PROFILER": "1",
            "TT_METAL_PROFILER_DIR": "/old/profiler",
            "TRACY_PORT": "1234",
            "CALL_LOG": str(self.root / "calls.log"),
            "ENV_LOG": str(self.root / "environment.log"),
            "PREFLIGHT_STATUS": "0",
            "PYTEST_STATUS": "0",
            "HOST_STATUS": "0",
            "REPORT_STATUS": "0",
            "BUILD_STATUS": "0",
            "FAIL_CASE": "none",
        }

    def run_workflow(self, *arguments):
        return subprocess.run(
            ["bash", str(self.runner), *arguments], env=self.environment, capture_output=True, text=True, timeout=20
        )

    def calls(self):
        return (self.root / "calls.log").read_text().splitlines()

    def test_runs_all_listed_cases_in_isolated_environment(self):
        result = self.run_workflow()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        captures = [call for call in calls if call.startswith("-m tracy ")]
        self.assertEqual(len(captures), 3)
        for dtype, capture in zip(("bf16", "fp32", "int32"), captures):
            self.assertIn("-p -r --check-exit-code", capture)
            self.assertIn(f"--suite sweep --case {dtype}_aligned_h --warmup 3 --samples 5 --tracy-signposts", capture)
        self.assertIn("--suite sweep --warmup 10 --samples 51", calls[4])
        self.assertIn("report_repeat_sweep.py", calls[-1])
        values = (self.root / "environment.log").read_text().splitlines()[0].split("|")
        self.assertEqual(
            values[:4], [str(self.root / "python_env"), str(self.root), str(self.root), str(self.root / "build/lib")]
        )
        self.assertEqual(values[5:], ["1", "unset", "unset", "unset", "unset"])
        self.assertNotIn("/old", values[4])
        self.assertIn("fixture sweep report", result.stdout)
        self.assertIn("Sweep artifacts retained:", result.stdout)

    def test_refuses_non_linux_and_invalid_arguments(self):
        self.assertEqual(self.run_workflow("--unknown").returncode, 2)
        executable(self.root / "mock-bin/uname", "printf 'Darwin\n'\n")
        self.assertEqual(self.run_workflow().returncode, 2)
        self.assertFalse((self.root / "calls.log").exists())

    def test_refuses_missing_own_environment(self):
        (self.root / "python_env/bin/activate").rename(self.root / "unused-activate")
        result = self.run_workflow()
        self.assertEqual(result.returncode, 2)
        self.assertIn("own prepared python_env", result.stderr)
        self.assertFalse((self.root / "calls.log").exists())

    def test_preflight_failure_stops_before_build_or_device(self):
        self.environment["PREFLIGHT_STATUS"] = "73"
        result = self.run_workflow("--build")
        self.assertEqual(result.returncode, 73)
        self.assertEqual(len(self.calls()), 1)
        self.assertFalse(list(self.root.glob("tt-repeat-sweep.*/build.log")))

    def test_build_failure_preserves_log_and_stops(self):
        self.environment["BUILD_STATUS"] = "42"
        result = self.run_workflow("--build")
        self.assertEqual(result.returncode, 42)
        self.assertEqual(len(self.calls()), 3)
        self.assertIn("fixture build log", result.stderr)
        self.assertEqual(len(list(self.root.glob("tt-repeat-sweep.*/build.log"))), 1)

    def test_correctness_failure_stops_before_any_benchmark(self):
        self.environment["PYTEST_STATUS"] = "43"
        result = self.run_workflow()
        self.assertEqual(result.returncode, 43)
        self.assertTrue(self.calls()[-1].startswith("-m pytest -q"))
        self.assertIn("fixture correctness log", result.stderr)

    def test_host_failure_stops_before_captures(self):
        self.environment["HOST_STATUS"] = "44"
        result = self.run_workflow()
        self.assertEqual(result.returncode, 44)
        self.assertFalse(any(call.startswith("-m tracy") for call in self.calls()))

    def test_capture_failure_stops_without_success_report(self):
        self.environment["FAIL_CASE"] = "fp32_aligned_h"
        result = self.run_workflow()
        self.assertEqual(result.returncode, 37)
        self.assertEqual(len([call for call in self.calls() if call.startswith("-m tracy")]), 2)
        self.assertFalse(any("report_repeat_sweep.py" in call for call in self.calls()))
        self.assertIn("fixture capture failure", result.stderr)

    def test_non_improvement_and_invalid_report_statuses_are_preserved(self):
        for status in ("1", "2"):
            self.environment["REPORT_STATUS"] = status
            result = self.run_workflow()
            self.assertEqual(result.returncode, int(status))
            self.assertIn("fixture sweep report", result.stdout)
            self.assertIn(f"Sweep report returned {status}", result.stderr)
        reports = list(self.root.glob("tt-repeat-sweep.*/SWEEP.md"))
        self.assertEqual(len(reports), 2)
        self.assertTrue(all("fixture sweep report" in path.read_text() for path in reports))


if __name__ == "__main__":
    unittest.main()
