import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import bundle_control as bc


class InputPolicyTests(unittest.TestCase):
    def test_v3_accepts_more_than_old_sixteen_input_quota(self):
        inputs = {
            f"input_{i}.json": {
                "path": f"inputs/fixture/{i}.json",
                "ref": "a" * 40,
                "blob": "b" * 40,
            }
            for i in range(40)
        }
        bundle = {
            "schema": bc.TEXT_SCHEMA,
            "bundle_id": "bt-many-inputs-unit",
            "files": {"run_task.py": "print('ok')\n"},
            "inputs": inputs,
            "timeout_seconds": 30,
        }
        _, _, normalized, _ = bc.validate_bundle(
            json.dumps(bundle), bundle["bundle_id"]
        )
        self.assertEqual(len(normalized), 40)

    def test_pinned_reader_has_no_project_byte_quota(self):
        raw = b"x" * 1024
        with mock.patch.object(
            bc, "read_repository_file", return_value=(raw, "b" * 40)
        ) as read:
            got = bc.read_pinned_file(
                "owner/private", "inputs/x.bin", "a" * 40, "b" * 40
            )
        self.assertEqual(got, raw)
        self.assertIsNone(read.call_args.kwargs["limit"])
        self.assertEqual(read.call_args.kwargs["expected_blob"], "b" * 40)


class OutputPolicyTests(unittest.TestCase):
    def _success_output(self, root):
        (root / "summary.json").write_text(
            json.dumps({"stage": "materialized", "rows": 1}) + "\n"
        )
        (root / "execution.json").write_text(
            json.dumps({
                "exit_code": 0,
                "timed_out": False,
                "output_limit_exceeded": False,
            }) + "\n"
        )
        (root / "result.jsonl").write_text('{"id":"a"}\n')

    def test_output_larger_than_old_32mib_quota_is_collected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._success_output(out)
            large = out / "diagnostic_payload.bin"
            with large.open("wb") as f:
                f.seek(33 * 1024 * 1024)
                f.write(b"x")
            captured = {}
            def put(repo, files, message, recovery_ref=None):
                captured.update(files)
                return "c" * 40
            with mock.patch.object(bc, "put_create_only", side_effect=put), \
                 mock.patch.dict(os.environ, {
                     "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"
                 }):
                bc.publish(
                    "owner/private", "bt-output-unit", "b" * 40, "a" * 40, out
                )
            key = (
                "transport/bundle-executions/bt-output-unit/123-1/"
                "diagnostic_payload.bin"
            )
            self.assertIn(key, captured)
            self.assertGreater(len(captured[key]), 32 * 1024 * 1024)

    def test_reserved_or_nonregular_output_still_fails(self):
        for name in ("receipt.json", "nested"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                self._success_output(out)
                if name == "nested":
                    (out / name).mkdir()
                else:
                    (out / name).write_text("{}\n")
                with mock.patch.object(bc, "put_create_only", return_value="c" * 40) as put, \
                     mock.patch.dict(os.environ, {
                         "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"
                     }):
                    with self.assertRaisesRegex(bc.BundleError, "failed_execution_preserved"):
                        bc.publish("owner/private", "bt-output-unit", "b" * 40, "a" * 40, out)
                put.assert_called_once()
                saved = put.call_args.args[1]
                prefix = "transport/bundle-executions/bt-output-unit/123-1/"
                receipt = json.loads(saved[prefix + "receipt.json"])
                self.assertEqual(receipt["status"], "failed_or_partial")
                self.assertEqual(receipt["collection_errors"][0]["name"], name)
                self.assertNotIn(name, receipt["files"])
                self.assertIn(prefix + "result.jsonl", saved)
                self.assertNotIn("transport/bundle-executions/bt-output-unit/completed.json", saved)


if __name__ == "__main__":
    unittest.main()
