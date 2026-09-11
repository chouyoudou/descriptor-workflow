"""Tests for the public, non-sensitive trigger envelope and event discovery."""
from __future__ import annotations

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
        self.assertFalse(contract["parallel_execution_safe"])

    def test_v2_keeps_parallel_preparation_but_serial_execution(self):
        contract = rc.resolve_request({"version": 2, "id": "agent-a-0042"})
        self.assertEqual(
            contract["task_path"], "transport/task_requests/agent-a-0042.json"
        )
        self.assertTrue(contract["parallel_preparation_safe"])
        self.assertFalse(contract["parallel_execution_safe"])

    def test_v3_binds_lane_slot_and_public_slot_path(self):
        path = "private_job/slots/descriptor-19.json"
        contract = rc.resolve_request(
            {"version": 3, "id": "agent-t-0042", "lane": "descriptor", "slot": 19},
            request_path=path,
        )
        self.assertEqual(contract["task_path"], "transport/task_requests/agent-t-0042.json")
        self.assertEqual(contract["request_path"], path)
        self.assertTrue(contract["parallel_execution_safe"])

    def test_lane_slot_limits_reserve_headroom(self):
        self.assertEqual(rc.LANE_SLOTS, {"descriptor": 20, "ml": 8, "infrastructure": 2})
        with self.assertRaisesRegex(rc.RequestContractError, "execution_slot_out_of_range"):
            rc.resolve_request(
                {"version": 3, "id": "bad", "lane": "descriptor", "slot": 20}
            )

    def test_slot_path_mismatch_is_rejected(self):
        with self.assertRaisesRegex(rc.RequestContractError, "slot_request_path_mismatch"):
            rc.resolve_request(
                {"version": 3, "id": "a", "lane": "ml", "slot": 2},
                request_path="private_job/slots/ml-03.json",
            )

    def test_request_cannot_publish_private_paths_or_extra_metadata(self):
        with self.assertRaisesRegex(rc.RequestContractError, "unsupported_request_shape"):
            rc.resolve_request(
                {
                    "version": 3,
                    "id": "agent-a-0042",
                    "lane": "descriptor",
                    "slot": 0,
                    "private_path": "secret/source.py",
                }
            )

    def test_push_discovers_one_isolated_slot_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slot = root / "private_job/slots/infrastructure-01.json"
            slot.parent.mkdir(parents=True)
            slot.write_text(
                json.dumps(
                    {
                        "version": 3,
                        "id": "parallel-probe-b",
                        "lane": "infrastructure",
                        "slot": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            event = {
                "commits": [
                    {
                        "added": ["private_job/slots/infrastructure-01.json"],
                        "modified": [],
                        "removed": [],
                    }
                ]
            }
            path, contract = rc.discover_request(
                event, event_name="push", repository_root=root
            )
            self.assertEqual(path, slot)
            self.assertEqual(contract["trigger_id"], "parallel-probe-b")

    def test_trigger_commit_cannot_mix_code_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slot = root / "private_job/slots/descriptor-00.json"
            slot.parent.mkdir(parents=True)
            slot.write_text(
                '{"version":3,"id":"a","lane":"descriptor","slot":0}\n',
                encoding="utf-8",
            )
            event = {
                "commits": [
                    {
                        "added": ["private_job/slots/descriptor-00.json"],
                        "modified": ["private_io.py"],
                        "removed": [],
                    }
                ]
            }
            with self.assertRaisesRegex(
                rc.RequestContractError, "trigger_commit_must_be_isolated"
            ):
                rc.discover_request(event, event_name="push", repository_root=root)

    def test_multi_commit_push_is_rejected(self):
        with self.assertRaisesRegex(
            rc.RequestContractError, "push_must_contain_one_commit"
        ):
            rc.discover_request(
                {"commits": [{}, {}]}, event_name="push", repository_root=Path(".")
            )

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
                    "--request",
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
            self.assertNotIn("transport/task_requests/agent-b-0007.json", completed.stdout)
            self.assertEqual(
                output.read_text(encoding="utf-8").splitlines(),
                [
                    "trigger_id=agent-b-0007",
                    "task_path=transport/task_requests/agent-b-0007.json",
                    "request_path=private_job/request.json",
                    "lane=descriptor",
                    "slot=0",
                    "parallel_preparation_safe=true",
                    "parallel_execution_safe=false",
                ],
            )

    def test_cli_validates_actual_slot_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / "slot.json"
            request.write_text(
                '{"version":3,"id":"probe","lane":"infrastructure","slot":1}\n',
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(rc.__file__).resolve()),
                    "--request",
                    str(request),
                    "--repository-path",
                    "private_job/slots/infrastructure-00.json",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("slot_request_path_mismatch", completed.stdout)
            self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
