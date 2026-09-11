"""Tests for the public, non-sensitive trigger envelope."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import request_contract as rc


class RequestContractTests(unittest.TestCase):
    def test_legacy_request_keeps_single_pointer_compatibility(self):
        contract = rc.resolve_request({"id": "direct-20260911-03"})
        self.assertEqual(contract["task_path"], "transport/active_task.json")
        self.assertFalse(contract["parallel_preparation_safe"])

    def test_v2_request_derives_create_only_per_trigger_task_path(self):
        contract = rc.resolve_request({"version": 2, "id": "agent-a-0042"})
        self.assertEqual(
            contract["task_path"],
            "transport/task_requests/agent-a-0042.json",
        )
        self.assertTrue(contract["parallel_preparation_safe"])

    def test_request_cannot_publish_private_paths_or_extra_metadata(self):
        with self.assertRaisesRegex(rc.RequestContractError, "unsupported_request_shape"):
            rc.resolve_request(
                {
                    "version": 2,
                    "id": "agent-a-0042",
                    "private_path": "secret/source.py",
                }
            )

    def test_invalid_trigger_id_is_rejected(self):
        with self.assertRaisesRegex(rc.RequestContractError, "invalid_trigger_id"):
            rc.resolve_request({"version": 2, "id": "../escape"})

    def test_cli_writes_only_bounded_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = root / "request.json"
            output = root / "github-output.txt"
            request.write_text(
                json.dumps({"version": 2, "id": "agent-b-0007"}) + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(rc.__file__).resolve()),
                    str(request),
                    "--github-output",
                    str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0)
            self.assertEqual(completed.stderr, "")
            self.assertNotIn(
                "transport/task_requests/agent-b-0007.json",
                completed.stdout,
            )
            self.assertEqual(
                output.read_text(encoding="utf-8").splitlines(),
                [
                    "trigger_id=agent-b-0007",
                    "task_path=transport/task_requests/agent-b-0007.json",
                    "parallel_preparation_safe=true",
                ],
            )


if __name__ == "__main__":
    unittest.main()
