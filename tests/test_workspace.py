"""User-visible conveniences: select, reuse, submit once, and move a real task."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import bundle_workspace as workspace


class Control:
    safe_repo = staticmethod(lambda repo: repo)

    def __init__(self):
        self.files, self.issues, self.calls = {}, [], []

    def put_create_only(self, repo, files, message):
        self.calls.append((repo, "save"))
        for path, raw in files.items():
            if path in self.files and self.files[path] != raw:
                raise ValueError("existing different file")
            self.files[path] = raw
        return "c" * 40

    def api(self, path, method="GET", body=None):
        self.calls.append((path, method))
        if path.endswith("/issues"):
            assert body == {"title": "bundle-task:bt-workspace-test"}
            item = {**body, "user": {"login": "owner"}, "html_url": "https://example.invalid/issue/1"}
            self.issues.append(item)
            return item
        if "/issues?" in path:
            return self.issues
        return {"private": True}

    def content(self, repo, path, ref):
        self.calls.append((path, ref))
        return {"type": "file", "sha": "d" * 40}


class WorkspaceTests(unittest.TestCase):
    def test_navigation_and_selection_keep_record_identity_without_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = (b'{"id":"a","species":["Si","O","O"],"source_index":8,"target":99}\n'
                   b'{"id":"b","species":["Fe"],"source_index":2,"target":12}\n')
            (root / "data").write_bytes(raw)
            self.assertEqual(workspace.index_structures(root / "data", root / "index"), 2)
            index = [json.loads(x) for x in (root / "index").read_text().splitlines()]
            self.assertEqual(index[0]["natoms"], 3)
            self.assertEqual(index[1]["source_line"], 2)
            self.assertNotIn("target", index[0])
            self.assertEqual(workspace.select_jsonl(root / "data", root / "selected", ids=["b"]), 1)
            self.assertEqual((root / "selected").read_bytes(), raw.splitlines(keepends=True)[1])
            (root / "result").write_text('{"id":"b","source_line":52,"value":null}\n')
            workspace.select_jsonl(root / "result", root / "saved", source_lines=[52])
            self.assertEqual((root / "saved").read_bytes(), (root / "result").read_bytes())
            with self.assertRaises(ValueError):
                workspace.select_jsonl(root / "data", root / "missing", ids=["c"])
            self.assertFalse((root / "missing").exists())

    def test_existing_requests_and_outputs_need_no_new_protocol_or_manual_hash(self):
        control = Control()
        receipt = {"source_ref": "a" * 40, "bundle_blob": "b" * 40, "status": "failed_or_partial"}
        pinned = workspace.output_binding(control, "owner/private", "transport/run/receipt.json",
                                           receipt, "observations.jsonl", "e" * 40)
        self.assertEqual(pinned["path"], "transport/run/observations.jsonl")
        request = workspace.make_run(receipt, "bt-workspace-test", {"scale": 3}, {"saved.jsonl": pinned})
        self.assertEqual(request["inputs_from"], receipt["bundle_blob"])
        self.assertNotIn("status", request)
        self.assertNotIn("files", request)
        with contextlib.redirect_stdout(io.StringIO()), patch.dict(os.environ, {"PRIVATE_REPO_TOKEN": "private-test", "GH_TOKEN": "public-test"}):
            first = workspace.submit(control, "owner/private", "owner/public", request)
            second = workspace.submit(control, "owner/private", "owner/public", request)
            self.assertEqual(os.environ["PRIVATE_REPO_TOKEN"], "private-test")
        self.assertEqual((first["activation"], second["activation"]), ("created", "already_exists"))
        self.assertEqual(len(control.issues), 1)
        self.assertEqual(json.loads(control.files["transport/bundle_inbox/bt-workspace-test.json"]), request)
        self.assertLess(control.calls.index(("owner/private", "save")),
                        control.calls.index(("/repos/owner/public/issues", "POST")))

    def test_exported_task_runs_after_move_and_preserves_failure_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); task = root / "staged"; task.mkdir()
            source = '''import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output');out=Path(p.parse_args().output)
assert not list(out.iterdir())
assert 'PRIVATE_REPO_TOKEN' not in os.environ
x=json.loads(Path('input.json').read_text())['x']
(out/'result.jsonl').write_text(json.dumps({'value':x*x})+'\\n')
print('private diagnostic')
raise SystemExit(7 if x<0 else 0)
'''
            (task / "run_task.py").write_text(source)
            (task / "input.json").write_text('{"x":3}')
            workspace.export_task(task, root / "packet.zip", {"source_ref": "a" * 40}, b"numpy==2.0\n")
            moved = root / "different-location"; moved.mkdir()
            with zipfile.ZipFile(root / "packet.zip") as packet:
                packet.extractall(moved)
                self.assertIn("environment-reference.txt", packet.namelist())
            for x, expected_code in ((3, 0), (-2, 7)):
                (moved / "task/input.json").write_text(json.dumps({"x": x}))
                output = root / ("output-" + str(x))
                process = subprocess.run([sys.executable, str(moved / "run_local.py"), "--output", str(output)],
                    env={**os.environ, "PRIVATE_REPO_TOKEN": "synthetic-test-token"}, capture_output=True, timeout=10)
                self.assertEqual(process.returncode, expected_code, process.stderr)
                self.assertEqual(json.loads((output / "result.jsonl").read_text())["value"], x*x)
                status = json.loads((output / "execution.json").read_text())
                self.assertEqual(status["exit_code"], expected_code)
                self.assertFalse(status["local_edits_audited"])
                self.assertIn("private diagnostic", (output / "execution.stdout.log").read_text())

    def test_local_return_keeps_actual_sources_and_never_claims_actions_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); task = root / "task"; task.mkdir()
            (task / "run_task.py").write_text("# local source edit\n")
            control = Control()
            returned = workspace.publish_local(control, "owner/private", task, "unused", "local-fixture",
                collector=lambda output: ({"result.jsonl": b'{"id":"a"}\n'}, [{"entry":"bad","reason":"read"}]))
            receipt = json.loads(control.files[returned["receipt_path"]])
            self.assertEqual(receipt["scientific_acceptance"], "not_assessed")
            self.assertNotIn("source_ref", receipt)
            self.assertTrue(receipt["collection_errors"])
            self.assertFalse(any(x.endswith("completed.json") for x in control.files))
            with zipfile.ZipFile(io.BytesIO(control.files[receipt["source_snapshot"]])) as packet:
                self.assertEqual(packet.read("task/run_task.py"), b"# local source edit\n")


if __name__ == "__main__":
    unittest.main()
