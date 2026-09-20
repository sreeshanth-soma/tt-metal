import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def executable(path, content):
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + content)
    path.chmod(0o755)


class CandidateWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tt-workflow-test-")
        self.addCleanup(self.temporary.cleanup)
        self.repository = Path(self.temporary.name).resolve()
        scripts = self.repository / "bounty-research/2026-09-20"
        scripts.mkdir(parents=True)
        self.runner = scripts / "run_candidate_validation.sh"
        shutil.copyfile(HERE / self.runner.name, self.runner)
        commands = self.repository / "mock-bin"
        commands.mkdir()
        executable(commands / "uname", "printf 'Linux\\n'\n")
        self.environment = {
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "TMPDIR": str(self.repository),
            "VIRTUAL_ENV": "/old-checkout/python_env",
            "PYTHON_ENV_DIR": "/old-checkout/python_env",
            "PYTHONPATH": "/old-checkout:/old-checkout/ttnn",
            "LD_LIBRARY_PATH": "/old-checkout/build/lib",
            "WORKFLOW_ENV_LOG": str(self.repository / "environment.log"),
            "WORKFLOW_CALL_LOG": str(self.repository / "calls.log"),
            "PREFLIGHT_STATUS": "73",
            "PYTEST_STATUS": "43",
            "BUILD_STATUS": "42",
        }

    def prepare_environment(self):
        binaries = self.repository / "python_env/bin"
        binaries.mkdir(parents=True)
        (binaries / "activate").write_text(
            'export VIRTUAL_ENV="$PYTHON_ENV_DIR"\nexport PATH="$VIRTUAL_ENV/bin:$PATH"\n'
        )
        executable(
            binaries / "python",
            'printf "%s\\n" "$VIRTUAL_ENV" "$PYTHON_ENV_DIR" "$TT_METAL_HOME" '
            '"$TT_METAL_RUNTIME_ROOT" "$LD_LIBRARY_PATH" "$PYTHONPATH" "$PYTHONNOUSERSITE" '
            '"$PATH" >> "$WORKFLOW_ENV_LOG"\n'
            'printf "%s\\n" "$*" >> "$WORKFLOW_CALL_LOG"\n'
            'if [[ "$1" == "-c" ]]; then exit "$PREFLIGHT_STATUS"; fi\n'
            'if [[ "$1" == "-m" && "$2" == "pytest" ]]; then\n'
            '    echo "fixture pytest failure" >&2\n'
            '    exit "$PYTEST_STATUS"\n'
            "fi\nexit 99\n",
        )

    def run_workflow(self, *arguments):
        return subprocess.run(
            ["bash", str(self.runner), *arguments],
            env=self.environment,
            text=True,
            capture_output=True,
            timeout=20,
        )

    def test_rejects_missing_candidate_environment(self):
        result = self.run_workflow()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("own prepared python_env", result.stderr)
        self.assertFalse((self.repository / "calls.log").exists())

    def test_rejects_non_linux_before_opening_runtime(self):
        executable(self.repository / "mock-bin/uname", "printf 'Darwin\\n'\n")
        result = self.run_workflow()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Linux hardware host", result.stderr)
        self.assertFalse((self.repository / "calls.log").exists())

    def test_selects_candidate_environment_not_inherited_checkout(self):
        self.prepare_environment()
        result = self.run_workflow()
        self.assertEqual(result.returncode, 73, result.stdout + result.stderr)
        values = (self.repository / "environment.log").read_text().splitlines()
        self.assertEqual(values[:2], [str(self.repository / "python_env")] * 2)
        self.assertEqual(values[2:4], [str(self.repository)] * 2)
        self.assertEqual(values[4], str(self.repository / "build/lib"))
        self.assertEqual(
            values[5],
            f"{self.runner.parent}:{self.repository}:{self.repository / 'ttnn'}",
        )
        self.assertEqual(values[6], "1")
        self.assertTrue(values[7].startswith(str(self.repository / "python_env/bin") + ":"))
        self.assertEqual(len((self.repository / "calls.log").read_text().splitlines()), 1)
        self.assertIn("Artifacts retained:", result.stdout)

    def test_build_failure_preserves_log_and_stops_before_hardware(self):
        self.prepare_environment()
        self.environment["PREFLIGHT_STATUS"] = "0"
        executable(self.repository / "build_metal.sh", 'echo "fixture build failure" >&2\nexit "$BUILD_STATUS"\n')
        result = self.run_workflow("--build")
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        self.assertIn("fixture build failure", result.stderr)
        self.assertEqual(len((self.repository / "calls.log").read_text().splitlines()), 1)
        logs = list(self.repository.glob("tt-repeat-direct.*/build.log"))
        self.assertEqual(len(logs), 1)
        self.assertIn("fixture build failure", logs[0].read_text())

    def test_correctness_failure_stops_before_benchmark(self):
        self.prepare_environment()
        self.environment["PREFLIGHT_STATUS"] = "0"
        result = self.run_workflow()
        self.assertEqual(result.returncode, 43, result.stdout + result.stderr)
        self.assertIn("fixture pytest failure", result.stderr)
        calls = (self.repository / "calls.log").read_text().splitlines()
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[1].startswith("-m pytest -q "))
        self.assertNotIn("Benchmarking six cases", result.stdout)
        self.assertEqual(len(list(self.repository.glob("tt-repeat-direct.*/correctness.log"))), 1)


if __name__ == "__main__":
    unittest.main()
