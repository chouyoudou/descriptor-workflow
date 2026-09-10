import json
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import unittest

from restricted_exec import MAX_TIMEOUT_SECONDS, run_bounded

ROOT = Path(__file__).resolve().parents[1]


class RestrictedExecutionTests(unittest.TestCase):
    def test_runtime_marker_is_captured_but_not_in_public_status(self):
        marker = "runtime-" + secrets.token_hex(24)
        result = run_bounded(
            [sys.executable, "-c", f"import sys; sys.stdout.write({marker!r}); sys.stderr.write({marker!r})"],
            timeout_seconds=2,
            max_output_bytes=4096,
        )
        self.assertEqual(result.child_exit_code, 0)
        self.assertEqual(result.stdout.decode(), marker)
        self.assertEqual(result.stderr.decode(), marker)
        self.assertNotIn(marker, json.dumps(result.public_status(), sort_keys=True))

    def test_output_limit_is_enforced(self):
        result = run_bounded(
            [sys.executable, "-c", "print('x' * 20000)"],
            timeout_seconds=2,
            max_output_bytes=4096,
        )
        self.assertTrue(result.output_limit_exceeded)
        self.assertLessEqual(len(result.stdout), 4096)
        self.assertLessEqual(len(result.stderr), 4096)

    def test_timeout_is_enforced(self):
        result = run_bounded(
            [sys.executable, "-c", "import time; time.sleep(3)"],
            timeout_seconds=0.2,
            max_output_bytes=4096,
        )
        self.assertTrue(result.timed_out)

    def test_engineering_timeout_budget_accepts_30_minutes_without_waiting(self):
        result = run_bounded(
            [sys.executable, "-c", "pass"],
            timeout_seconds=MAX_TIMEOUT_SECONDS,
            max_output_bytes=4096,
        )
        self.assertEqual(MAX_TIMEOUT_SECONDS, 1800)
        self.assertEqual(result.child_exit_code, 0)
        self.assertFalse(result.timed_out)

    def test_timeout_above_engineering_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            run_bounded(
                [sys.executable, "-c", "pass"],
                timeout_seconds=MAX_TIMEOUT_SECONDS + 0.001,
                max_output_bytes=4096,
            )

    def test_geometry_output_remains_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            stdout_file = Path(tmp) / "result.json"
            status_file = Path(tmp) / "status.json"
            proc = subprocess.run([
                sys.executable, str(ROOT / "restricted_exec.py"),
                "--timeout", "3", "--max-output", "8192",
                "--stdout-file", str(stdout_file), "--status-file", str(status_file), "--",
                sys.executable, str(ROOT / "crystal_geometry.py"), str(ROOT / "examples" / "cells.json"),
            ], capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(proc.stdout, "")
            self.assertEqual(proc.stderr, "")
            payload = json.loads(stdout_file.read_text())
            self.assertEqual([row["id"] for row in payload], ["cubic", "orthorhombic", "skew"])
            status = json.loads(status_file.read_text())
            self.assertEqual(status["child_exit_code"], 0)
            self.assertFalse(status["timed_out"])
            self.assertFalse(status["output_limit_exceeded"])

    def test_geometry_failure_is_nonzero_without_raw_error_echo(self):
        marker = "runtime-" + secrets.token_hex(24)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text('{"broken":"' + marker, encoding="utf-8")
            stdout_file = Path(tmp) / "result.json"
            status_file = Path(tmp) / "status.json"
            proc = subprocess.run([
                sys.executable, str(ROOT / "restricted_exec.py"),
                "--timeout", "3", "--max-output", "8192",
                "--stdout-file", str(stdout_file), "--status-file", str(status_file), "--",
                sys.executable, str(ROOT / "crystal_geometry.py"), str(bad),
            ], capture_output=True, text=True, check=False)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(proc.stdout, "")
            self.assertEqual(proc.stderr, "")
            self.assertFalse(stdout_file.exists())
            public = status_file.read_text()
            self.assertNotIn(marker, public)
            self.assertNotIn("Traceback", public)
            status = json.loads(public)
            self.assertNotEqual(status["child_exit_code"], 0)
            self.assertGreater(status["stderr_bytes"], 0)

    def test_unhandled_candidate_exception_is_nonzero_without_traceback_echo(self):
        with tempfile.TemporaryDirectory() as tmp:
            stdout_file = Path(tmp) / "result.txt"
            status_file = Path(tmp) / "status.json"
            proc = subprocess.run([
                sys.executable, str(ROOT / "restricted_exec.py"),
                "--timeout", "3", "--max-output", "8192",
                "--stdout-file", str(stdout_file), "--status-file", str(status_file), "--",
                sys.executable, "-c", "raise RuntimeError('synthetic failure detail')",
            ], capture_output=True, text=True, check=False)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(proc.stdout, "")
            self.assertEqual(proc.stderr, "")
            self.assertFalse(stdout_file.exists())
            public = status_file.read_text()
            self.assertNotIn("Traceback", public)
            self.assertNotIn("synthetic failure detail", public)


if __name__ == "__main__":
    unittest.main()
