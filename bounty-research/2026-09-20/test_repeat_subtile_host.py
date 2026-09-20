import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


class RepeatSubtileHostTest(unittest.TestCase):
    def test_actual_reader_with_guarded_host_mocks(self):
        compiler = shutil.which("clang++")
        if compiler is None:
            self.skipTest("clang++ is required for the actual-reader host test")
        kernel = (
            ROOT
            / "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/codegen/kernels/reader_repeat_interleave_subtile.cpp"
        )
        digest = hashlib.sha256(kernel.read_bytes()).hexdigest()
        results = []
        with tempfile.TemporaryDirectory(prefix="tt-repeat-subtile-") as temporary:
            executable = Path(temporary) / "reader_test"
            for element_bytes, repeat_height, repeats in itertools.product(
                (2, 4), (0, 1), (2, 3, 4, 7, 8, 16, 33, 127)
            ):
                with self.subTest(element_bytes=element_bytes, repeat_height=repeat_height, repeats=repeats):
                    command = [
                        compiler,
                        "-std=c++17",
                        "-O1",
                        "-g",
                        "-Wall",
                        "-Wextra",
                        "-Werror",
                        "-fsanitize=address,undefined",
                        "-fno-omit-frame-pointer",
                        f"-DTEST_ELEMENT_BYTES={element_bytes}",
                        f"-DTEST_REPEAT_HEIGHT={repeat_height}",
                        f"-DTEST_REPEATS={repeats}",
                        f"-I{HERE / 'subtile_host'}",
                        f"-I{ROOT}",
                        str(HERE / "subtile_host" / "reader_test.cpp"),
                        "-o",
                        str(executable),
                    ]
                    built = subprocess.run(command, capture_output=True, text=True, timeout=120)
                    self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                    tested = subprocess.run([str(executable)], capture_output=True, text=True, timeout=120)
                    self.assertEqual(tested.returncode, 0, tested.stdout + tested.stderr)
                    result = json.loads(tested.stdout)
                    self.assertEqual(result["cases"], 251)
                    self.assertEqual(
                        result["output_l1_word_writes"], result["checked_padded_elements"] * element_bytes // 4
                    )
                    results.append(result)
                    print(
                        json.dumps(
                            {
                                "element_bytes": element_bytes,
                                "repeat_height": repeat_height,
                                "repeats": repeats,
                                **result,
                            }
                        ),
                        flush=True,
                    )
        self.assertEqual(len(results), 32)
        self.assertEqual(hashlib.sha256(kernel.read_bytes()).hexdigest(), digest)
        print(
            json.dumps(
                {
                    "event": "host_validation_summary",
                    "kernel_sha256": digest,
                    "variants": len(results),
                    "cases": sum(result["cases"] for result in results),
                    "core_invocations": sum(result["core_invocations"] for result in results),
                    "checked_padded_elements": sum(result["checked_padded_elements"] for result in results),
                    "source_l1_halfword_reads": sum(result["source_l1_halfword_reads"] for result in results),
                    "source_l1_word_reads": sum(result["source_l1_word_reads"] for result in results),
                    "output_l1_word_writes": sum(result["output_l1_word_writes"] for result in results),
                    "validation": "actual reader with guarded host mocks, ASan and UBSan; not device execution",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
