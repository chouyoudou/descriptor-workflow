"""Synthetic publication fixtures; no private data or remote writes."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import bundle_control as bc


class PublicationPreservationTests(unittest.TestCase):
    bundle_id = "bt-output-preservation-unit"
    prefix = "transport/bundle-executions/bt-output-preservation-unit/123-1/"

    def fixture(self, root):
        raw = {
            "result.jsonl": b'{"id":"case-a","value":7}\n',
            "summary.json": b'{"stage":"materialized","rows":1}\n',
            "execution.json": b'{"exit_code":0,"timed_out":false,"output_limit_exceeded":false}\n',
            "checkpoint.bin": bytes(range(256)),
        }
        for name, data in raw.items():
            (root / name).write_bytes(data)
        return raw

    def publish(self, root, put_error=None):
        captured = {}
        def put(repo, files, message, recovery_ref=None):
            if put_error:
                raise bc.BundleError(put_error)
            captured.update(files)
            return "c" * 40
        failure = None
        logs = io.StringIO()
        with mock.patch.object(bc, "put_create_only", side_effect=put) as writer, \
             mock.patch.dict(os.environ, {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}), \
             contextlib.redirect_stdout(logs):
            try:
                bc.publish("owner/private", self.bundle_id, "a" * 40, "b" * 40, root)
            except bc.BundleError as exc:
                failure = exc
        return captured, failure, writer.call_count, logs.getvalue()

    def test_clean_output_keeps_existing_success_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self.fixture(root)
            saved, failure, calls, _ = self.publish(root)
            self.assertIsNone(failure)
            self.assertEqual(calls, 1)
            for name, raw in original.items():
                self.assertEqual(saved[self.prefix + name], raw)
            receipt = json.loads(saved[self.prefix + "receipt.json"])
            self.assertEqual(receipt["status"], "materialized")
            self.assertEqual(receipt["publication_status"], "published")
            self.assertEqual(receipt["scientific_completion"], "not_claimed")
            self.assertEqual(receipt["scientific_review_status"], "pending")
            self.assertIn("transport/bundle-executions/" + self.bundle_id + "/completed.json", saved)

    def test_one_bad_entry_does_not_discard_regular_siblings(self):
        for kind in ("directory", "reserved", "symlink", "hardlink", "fifo", "backslash"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                root = base / "output"
                root.mkdir()
                original = self.fixture(root)
                outside = base / "outside.bin"
                outside.write_bytes(b"must-not-be-collected")
                name = "receipt.json" if kind == "reserved" else "bad-entry"
                if kind == "backslash":
                    name = "bad\\entry"
                path = root / name
                if kind == "directory":
                    path.mkdir()
                    (path / "nested.txt").write_bytes(b"not-a-supported-root-output")
                elif kind == "symlink":
                    path.symlink_to(outside)
                elif kind == "hardlink":
                    os.link(outside, path)
                elif kind == "fifo":
                    os.mkfifo(path)
                else:
                    path.write_bytes(b"not-a-valid-output-entry")
                saved, failure, calls, logs = self.publish(root)
                self.assertIsNone(failure)
                self.assertEqual(calls, 1, "Valid siblings must reach the private transaction before failure returns")
                for filename, raw in original.items():
                    self.assertEqual(saved[self.prefix + filename], raw)
                receipt = json.loads(saved[self.prefix + "receipt.json"])
                self.assertEqual(receipt["status"], "failed_or_partial")
                self.assertEqual(receipt["publication_status"], "published")
                self.assertEqual(receipt["scientific_completion"], "not_claimed")
                self.assertTrue(receipt["collection_errors"])
                self.assertEqual(receipt["result_validation"], {"status": "valid", "records": 1})
                self.assertNotIn(name, receipt["files"])
                self.assertEqual(receipt["files"]["checkpoint.bin"], hashlib.sha256(original["checkpoint.bin"]).hexdigest())
                self.assertNotIn("transport/bundle-executions/" + self.bundle_id + "/completed.json", saved)
                review = json.loads(saved["transport/reviews/pending/" + self.bundle_id + "/123-1.json"])
                self.assertEqual(review["execution_status"], "failed_or_partial")
                self.assertEqual(review["next_action"], "independent_review")
                self.assertEqual(review["scientific_review_status"], "pending")
                self.assertNotIn(name, logs)
                self.assertEqual(outside.read_bytes(), b"must-not-be-collected")

    def test_nonstandard_report_is_preserved_for_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "report.txt").write_bytes(b"human-readable report only\n")
            saved, failure, calls, _ = self.publish(root)
            self.assertIsNone(failure)
            self.assertEqual(calls, 1)
            prefix = self.prefix
            self.assertEqual(saved[prefix + "report.txt"], b"human-readable report only\n")
            receipt = json.loads(saved[prefix + "receipt.json"])
            self.assertEqual(receipt["status"], "failed_or_partial")
            self.assertEqual(receipt["publication_status"], "published")
            self.assertEqual(receipt["scientific_completion"], "not_claimed")
            self.assertNotIn(
                "transport/bundle-executions/" + self.bundle_id + "/completed.json",
                saved,
            )
            review = json.loads(
                saved["transport/reviews/pending/" + self.bundle_id + "/123-1.json"]
            )
            self.assertEqual(review["next_action"], "independent_review")
            self.assertEqual(review["scientific_review_status"], "pending")

    def test_execution_failure_and_timeout_are_saved_for_review(self):
        for name, execution in (
            ("failure", {"exit_code": 7, "timed_out": False, "output_limit_exceeded": False}),
            ("timeout", {"exit_code": 0, "timed_out": True, "output_limit_exceeded": False}),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root)
                (root / "execution.json").write_text(
                    json.dumps(execution) + chr(10), encoding="utf-8"
                )
                saved, failure, calls, _ = self.publish(root)
                self.assertIsNone(failure)
                self.assertEqual(calls, 1)
                receipt = json.loads(saved[self.prefix + "receipt.json"])
                self.assertEqual(receipt["status"], "failed_or_partial")
                self.assertEqual(receipt["publication_status"], "published")
                self.assertEqual(receipt["scientific_completion"], "not_claimed")
                self.assertEqual(receipt["execution_status"], execution)
                self.assertNotIn(
                    "transport/bundle-executions/" + self.bundle_id + "/completed.json",
                    saved,
                )

    def test_real_private_put_failure_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "report.txt").write_bytes(b"preserve before API failure" + bytes([10]))
            saved, failure, calls, _ = self.publish(root, put_error="api_failure")
            self.assertIsNotNone(failure)
            self.assertIn("api_failure", str(failure))
            self.assertEqual(calls, 1)
            self.assertEqual(saved, {})


if __name__ == "__main__":
    unittest.main()
