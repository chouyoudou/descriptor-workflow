"""Source reuse is tested as execution of the same bytes with independent inputs."""
import base64
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

import bundle_ingress as ingress

BUNDLE = "bt-run-unit"
REF = "a" * 40
SOURCE = b'''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--output",required=True)
out=Path(p.parse_args().output); out.mkdir(exist_ok=True)
root=Path(__file__).parent
x=json.loads((root/"input.json").read_text())["x"] if (root/"input.json").exists() else 2
cfg=json.loads((root/"run_config.json").read_text()) if (root/"run_config.json").exists() else {}
value=cfg.get("scale",1)*x*x
(out/"result.jsonl").write_text(json.dumps({"id":"synthetic","value":value})+"\\n")
(out/"summary.json").write_text(json.dumps({"stage":"materialized","rows":1})+"\\n")
(out/"focused-and-batch.log").write_text("arithmetic fixture completed\\n")
'''


def blob(raw):
    return hashlib.sha1(("blob %d" % len(raw)).encode() + b"\0" + raw).hexdigest()


def missing():
    return urllib.error.HTTPError("fixture", 404, "not found", {}, None)


class Control:
    """Mock transport only; actual candidate arithmetic is run in one test."""
    FILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
    ALLOWED_INPUT_PREFIXES = ("inputs/", "transport/tasks/")

    def __init__(self, request=None):
        self.request = request or {"schema": ingress.RUN_SCHEMA, "bundle_id": BUNDLE,
                                  "source_ref": REF, "inputs": {}, "timeout_seconds": 30}
        self.raw = (json.dumps(self.request) + "\n").encode()
        self.blob = blob(self.raw)
        self.completed = None
        self.sources = {"run_task.py": SOURCE, "example.json": b'{"x":3}\n'}
        self.input_data = b'{"x":3}\n'
        self.calls = []
        self.published = {}
        self.bad_source = False
        self.bad_input = False

    @staticmethod
    def safe_repo(repo):
        if repo != "owner/private":
            raise ValueError("repo")
        return repo

    @staticmethod
    def safe_path(path):
        if not isinstance(path, str) or path.startswith("/") or ".." in path.split("/") or "\\" in path:
            raise ValueError("invalid_path")
        return path

    def read_repository_file(self, repo, path, ref):
        self.calls.append(("read", path, ref))
        if path.endswith("/completed.json"):
            if self.completed is None:
                raise missing()
            return json.dumps(self.completed).encode(), "c" * 40
        if path.endswith("BUNDLE_SOURCE.json"):
            return json.dumps({"bundle_blob": self.blob}).encode(), "d" * 40
        return self.raw, self.blob

    def _load_base_source(self, repo, ref):
        self.calls.append(("source", ref))
        if ref != REF or self.bad_source:
            raise ValueError("base_source_hash_mismatch")
        return {"bundle_id": "bt-old-source", "sources": dict(self.sources),
                "manifest": {"inputs": {"old_input.json": "must_not_be_inherited"}}}

    def api(self, path, **kwargs):
        self.calls.append(("api", path))
        return {"sha": self.blob, "encoding": "base64",
                "content": base64.b64encode(self.raw).decode()}

    @staticmethod
    def decode_contents(item):
        raw = base64.b64decode(item["content"])
        if item["sha"] != blob(raw):
            raise ValueError("blob_identity_mismatch")
        return raw

    def read_pinned_file(self, repo, path, ref, expected_blob):
        self.calls.append(("input", path, ref, expected_blob))
        if self.bad_input or expected_blob != blob(self.input_data):
            raise ValueError("pinned_file_identity_mismatch")
        return self.input_data

    def materialize(self, *args):
        raise AssertionError("run request must not materialize source")

    def fetch_task(self, *args):
        self.calls.append(("legacy_fetch", args))

    def put_create_only(self, repo, files, message, recovery_ref=None):
        self.published.update(files)
        return "f" * 40

    @staticmethod
    def recovery_branch(*args):
        return "recovery-unit"


class RunRequestTests(unittest.TestCase):
    def request(self, **changes):
        return {"schema": ingress.RUN_SCHEMA, "bundle_id": BUNDLE,
                "source_ref": REF, "inputs": {}, "timeout_seconds": 30, **changes}

    def spec(self):
        return {"path": "transport/tasks/fixture/example.json", "ref": REF,
                "blob": blob(b'{"x":3}\n')}

    def fetch(self, c, target):
        return ingress.fetch_task("owner/private", BUNDLE, REF, c.blob, target, c)

    def test_many_inputs_are_not_rejected_by_a_project_count_quota(self):
        inputs = {f"input_{i}.json": self.spec() for i in range(40)}
        c = Control(self.request(inputs=inputs))
        got = ingress.materialize_outputs(c, "owner/private", BUNDLE)
        self.assertEqual(got["source_ref"], REF)
        self.assertEqual(len(ingress.validate_run_request(c, c.request, BUNDLE)[1]), 40)

    def test_no_edit_fields_required_and_no_source_write(self):
        c = Control()
        got = ingress.materialize_outputs(c, "owner/private", BUNDLE)
        self.assertEqual(got["source_ref"], REF)
        self.assertEqual(got["source_mode"], "reused")
        self.assertEqual(got["skip"], "false")
        self.assertFalse(c.published)

    def test_identical_request_can_have_independent_run_identity(self):
        c = Control(self.request(bundle_id="bt-another-run"))
        got = ingress.materialize_outputs(c, "owner/private", "bt-another-run")
        self.assertEqual(got["source_ref"], REF)
        self.assertEqual(got["skip"], "false")

    def test_completed_replay_skips_parent_reads(self):
        c = Control()
        c.completed = {"bundle_blob": c.blob, "source_ref": REF, "status": "materialized"}
        got = ingress.materialize_outputs(c, "owner/private", BUNDLE)
        self.assertEqual(got["skip"], "true")
        self.assertFalse(any(call[0] == "source" for call in c.calls))
        c.completed["bundle_blob"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "completed_run_identity_changed"):
            ingress.materialize_outputs(c, "owner/private", BUNDLE)

    def test_invalid_source_and_accidental_code_fields_rejected(self):
        for changes in ({"source_ref": "main"}, {"files": {"run_task.py": "x"}},
                        {"changed_files": {}}, {"config": {"x": float("nan")}}):
            c = Control(self.request(**changes))
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ingress.materialize_outputs(c, "owner/private", BUNDLE)
        c = Control(); c.bad_source = True
        with self.assertRaisesRegex(ValueError, "base_source_hash_mismatch"):
            ingress.materialize_outputs(c, "owner/private", BUNDLE)

    def test_source_input_and_config_collisions_rejected(self):
        for request in (self.request(inputs={"run_task.py": self.spec()}),
                        self.request(inputs={"../x": self.spec()}),
                        self.request(inputs={"run_config.json": self.spec()}, config={})): 
            with self.subTest(request=request), self.assertRaises(ValueError):
                ingress.materialize_outputs(Control(request), "owner/private", BUNDLE)
        c = Control(self.request(config={})); c.sources["run_config.json"] = b"{}"
        with self.assertRaisesRegex(ValueError, "run_config_source_collision"):
            ingress.materialize_outputs(c, "owner/private", BUNDLE)

    def test_fetch_preserves_source_and_uses_new_inputs_config(self):
        c = Control(self.request(inputs={"input.json": self.spec()}, config={"scale": 2}))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "task"
            with contextlib.redirect_stdout(io.StringIO()):
                self.fetch(c, target)
            for name, data in c.sources.items():
                self.assertEqual((target / name).read_bytes(), data)
            self.assertEqual(json.loads((target / "input.json").read_text()), {"x": 3})
            self.assertEqual(json.loads((target / "run_config.json").read_text()), {"scale": 2})
            self.assertFalse((target / "old_input.json").exists())
            self.assertFalse(c.published)
            self.assertFalse(any(call[0] == "read" for call in c.calls))
            out = Path(tmp) / "out"
            subprocess.run([sys.executable, "-B", str(target / "run_task.py"),
                            "--output", str(out)], check=True, capture_output=True, timeout=5)
            self.assertEqual(json.loads((out / "result.jsonl").read_text())["value"], 18)
            self.assertEqual((target / "run_task.py").read_bytes(), SOURCE)

    def test_input_hash_failure_cleans_only_new_staging(self):
        c = Control(self.request(inputs={"input.json": self.spec()})); c.bad_input = True
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "task"
            with self.assertRaisesRegex(ValueError, "pinned_file_identity_mismatch"):
                self.fetch(c, target)
            self.assertFalse(target.exists())
            target.mkdir(); (target / "old.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                self.fetch(c, target)
            self.assertEqual((target / "old.txt").read_text(), "keep")

    def test_fetch_rejects_wrong_request_blob_or_source_ref(self):
        c = Control()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "request_blob_identity_mismatch"):
                ingress.fetch_task("owner/private", BUNDLE, REF, "0" * 40, tmp, c)
            with self.assertRaisesRegex(ValueError, "run_source_identity_mismatch"):
                ingress.fetch_task("owner/private", BUNDLE, "c" * 40, c.blob, tmp, c)

    def test_legacy_fetch_remains_delegated(self):
        for schema in ("private-task-bundle/2", "private-task-bundle/3", "private-task-bundle/4"):
            c = Control(self.request(schema=schema))
            self.fetch(c, "unused")
            self.assertEqual(c.calls[-1], ("legacy_fetch", ("owner/private", BUNDLE, REF, "unused")))

    def test_admission_saves_request_identity_not_a_new_source(self):
        c = Control()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            env = {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_OUTPUT": str(output)}
            with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
                record = ingress.admit("owner/private", BUNDLE, c)
            self.assertEqual(record["source_ref"], REF)
            self.assertEqual(record["bundle_blob"], c.blob)
            self.assertEqual(record["source_mode"], "reused")
            self.assertEqual(list(c.published), [f"transport/bundle-executions/{BUNDLE}/123-1/admission.json"])
            self.assertIn("source_mode=reused", output.read_text())


if __name__ == "__main__":
    unittest.main()
