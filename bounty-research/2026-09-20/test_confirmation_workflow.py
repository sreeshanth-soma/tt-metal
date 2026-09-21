import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_candidate_workflow import executable


HERE = Path(__file__).resolve().parent


class ConfirmationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tt-confirm workflow-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.scripts = self.root / "bounty-research/2026-09-20"
        self.scripts.mkdir(parents=True)
        self.runner = self.scripts / "confirm_candidate_validation.sh"
        shutil.copyfile(HERE / self.runner.name, self.runner)
        commands = self.root / "mock-bin"
        commands.mkdir()
        executable(commands / "uname", "printf 'Linux\\n'\n")
        binaries = self.root / "python_env/bin"
        binaries.mkdir(parents=True)
        (binaries / "activate").write_text("")
        executable(
            binaries / "python",
            'printf "%s\\n" "$@" > "$CONFIRM_ARGUMENTS"\n' 'printf "fixture confirmation\\n"\nexit "$CONFIRM_STATUS"\n',
        )
        executable(
            self.scripts / "run_candidate_validation.sh",
            'printf "%s|%s\\n" "$TMPDIR" "$*" >> "$RUN_ARGUMENTS"\n'
            'if [[ "$TMPDIR" == *"/run-$FAIL_RUN" ]]; then echo "fixture validation failure" >&2; exit 37; fi\n'
            'if [[ "$NO_ARTIFACTS" == "0" ]]; then mkdir "$TMPDIR/tt-repeat-direct.fixture"; fi\n',
        )
        self.environment = {
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "TMPDIR": str(self.root),
            "RUN_ARGUMENTS": str(self.root / "run-arguments.log"),
            "CONFIRM_ARGUMENTS": str(self.root / "confirm-arguments.log"),
            "FAIL_RUN": "0",
            "NO_ARTIFACTS": "0",
            "CONFIRM_STATUS": "0",
        }

    def run_workflow(self, *arguments):
        return subprocess.run(
            ["bash", str(self.runner), *arguments], env=self.environment, capture_output=True, text=True, timeout=20
        )

    def test_runs_three_processes_and_builds_only_once(self):
        result = self.run_workflow("--build")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = (self.root / "run-arguments.log").read_text().splitlines()
        self.assertEqual(len(calls), 3)
        self.assertEqual([call.split("|")[1] for call in calls], ["--build", "", ""])
        self.assertEqual(len({call.split("|")[0] for call in calls}), 3)
        confirmation = (self.root / "confirm-arguments.log").read_text().splitlines()
        self.assertEqual(len(confirmation), 6)
        self.assertTrue(confirmation[0].endswith("confirm_candidate_runs.py"))
        self.assertTrue(all(path.endswith("tt-repeat-direct.fixture") for path in confirmation[1:4]))
        self.assertEqual(confirmation[4], "--json-output")
        self.assertIn("fixture confirmation", result.stdout)
        self.assertEqual(len(list(self.root.glob("tt-repeat-confirm.*/CONFIRMATION.md"))), 1)

    def test_stops_on_failed_run_without_claiming_confirmation(self):
        self.environment["FAIL_RUN"] = "2"
        result = self.run_workflow()
        self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
        self.assertEqual(len((self.root / "run-arguments.log").read_text().splitlines()), 2)
        self.assertFalse((self.root / "confirm-arguments.log").exists())
        self.assertIn("fixture validation failure", result.stderr)
        self.assertIn("artifacts retained", result.stdout)

    def test_rejects_missing_artifacts(self):
        self.environment["NO_ARTIFACTS"] = "1"
        result = self.run_workflow()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("exactly one artifact", result.stderr)
        self.assertFalse((self.root / "confirm-arguments.log").exists())

    def test_propagates_regression_exit_code_and_keeps_report(self):
        self.environment["CONFIRM_STATUS"] = "1"
        result = self.run_workflow()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Confirmation did not pass", result.stderr)
        reports = list(self.root.glob("tt-repeat-confirm.*/CONFIRMATION.md"))
        self.assertEqual(len(reports), 1)
        self.assertIn("fixture confirmation", reports[0].read_text())

    def test_refuses_non_linux(self):
        executable(self.root / "mock-bin/uname", "printf 'Darwin\\n'\n")
        result = self.run_workflow()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Linux hardware host", result.stderr)
        self.assertFalse((self.root / "run-arguments.log").exists())


if __name__ == "__main__":
    unittest.main()
