"""Exercise optional input-list reuse through the existing run entry."""
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
import urllib.error

import bundle_ingress as ingress

REF = "a" * 40
REPO = "owner/private"
DATA = b'{"x": 3}\n'
SOURCE = b'''import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--output", required=True)
out=Path(p.parse_args().output); out.mkdir()
root=Path(__file__).parent
x=json.loads((root/"input.json").read_text())["x"]
scale=json.loads((root/"run_config.json").read_text())["scale"]
(out/"result.jsonl").write_text(json.dumps({"value": scale*x*x})+"\\n")
'''


def blob(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()


def spec(path="inputs/example.json"):
    return {"path": path, "ref": REF, "blob": blob(DATA)}


def request(**changes):
    return {"schema": ingress.RUN_SCHEMA, "bundle_id": "bt-input-reuse-test",
            "source_ref": REF, "timeout_seconds": 30, **changes}


class Control:
    FILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
    ALLOWED_INPUT_PREFIXES = ("inputs/",)
    safe_repo = staticmethod(lambda repo: repo)
    safe_path = staticmethod(lambda path: path)

    def __init__(self):
        self.objects, self.calls = {}, []

    def save(self, obj):
        raw = json.dumps(obj).encode()
        ref = blob(raw)
        self.objects[ref] = raw
        return ref

    def api(self, path, **kwargs):
        self.calls.append(path)
        ref = path.rsplit("/", 1)[-1]
        raw = self.objects[ref]
        return {"sha": ref, "encoding": "base64", "content": base64.b64encode(raw).decode()}

    @staticmethod
    def decode_contents(item):
        return base64.b64decode(item["content"])

    def read_repository_file(self, repo, path, ref):
        if path.endswith("/completed.json"):
            raise urllib.error.HTTPError("fixture", 404, "missing", {}, None)
        raise AssertionError("fetch must use saved request objects, not mutable inboxes")

    def _load_base_source(self, repo, ref):
        assert ref == REF
        return {"sources": {"run_task.py": SOURCE}}

    def read_pinned_file(self, repo, path, ref, expected_blob):
        assert {"path": path, "ref": ref, "blob": expected_blob} == spec()
        return DATA


class InputReuseTests(unittest.TestCase):
    def test_inherited_inputs_run_with_new_config_without_source_publication(self):
        c = Control()
        parent = c.save(request(inputs={"input.json": spec()}, config={"scale": 2}))
        inherited = request(inputs_from=parent, config={"scale": 3})
        explicit = request(inputs={"input.json": spec()}, config={"scale": 3})
        self.assertEqual(ingress.validate_run_request(c, inherited, inherited["bundle_id"], REPO),
                         ingress.validate_run_request(c, explicit, explicit["bundle_id"], REPO))
        current = c.save(inherited)
        admission = ingress.run_outputs(c, REPO, inherited["bundle_id"], inherited, current)
        self.assertEqual(admission["source_mode"], "reused")
        self.assertEqual(admission["bundle_blob"], current)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            target, out = Path(tmp)/"task", Path(tmp)/"output"
            ingress.fetch_task(REPO, inherited["bundle_id"], REF, current, target, c)
            subprocess.run([sys.executable, "-B", str(target/"run_task.py"), "--output", str(out)],
                           check=True, capture_output=True, timeout=5)
            self.assertEqual(json.loads((out/"result.jsonl").read_text())["value"], 27)
            self.assertEqual((target/"run_task.py").read_bytes(), SOURCE)
        # All remote operations are GET-like fixture reads; there is no writer.
        self.assertTrue(all("/git/blobs/" in call for call in c.calls))
        self.assertNotIn("inputs", inherited)

    def test_chain_overlays_bindings_but_never_inherits_source_config_or_budget(self):
        c = Control()
        for schema in ("private-task-bundle/2", "private-task-bundle/3", "private-task-bundle/4"):
            with self.subTest(schema=schema):
                original = c.save({"schema": schema, "inputs": {"a.json": spec(), "b.json": spec()},
                                   "source_ref": "b"*40, "config": {"scale": 99}, "timeout_seconds": 1700})
                next_blob = c.save(request(inputs_from=original, inputs={"a.json": spec("inputs/next.json")}))
                new = request(inputs_from=next_blob, inputs={"c.json": spec()})
                ref, inputs, budget, config = ingress.validate_run_request(c, new, new["bundle_id"], REPO)
                self.assertEqual(set(inputs), {"a.json", "b.json", "c.json"})
                self.assertEqual(inputs["a.json"]["path"], "inputs/next.json")
                self.assertEqual((ref, budget, config), (REF, 30, None))

    def test_absent_option_keeps_empty_and_explicit_inputs_without_extra_reads(self):
        c = Control()
        for changes, expected in (({}, {}), ({"inputs": {}}, {}),
                                  ({"inputs": {"a.json": spec()}}, {"a.json": spec()})):
            obj = request(**changes)
            self.assertEqual(ingress.validate_run_request(c, obj, obj["bundle_id"])[1], expected)
        self.assertEqual(c.calls, [])

    def test_new_option_uses_existing_input_and_collision_rules(self):
        c = Control()
        invalid = c.save(request(inputs={"../outside": spec()}))
        obj = request(inputs_from=invalid)
        with self.assertRaisesRegex(ValueError, "invalid_input_entry"):
            ingress.validate_run_request(c, obj, obj["bundle_id"], REPO)
        collision = c.save(request(inputs={"run_task.py": spec()}))
        obj = request(inputs_from=collision)
        with self.assertRaisesRegex(ValueError, "input_source_name_collision"):
            ingress.run_outputs(c, REPO, obj["bundle_id"], obj, c.save(obj))
        not_request = c.save({"schema": "unrelated-data/1", "inputs": {}})
        obj = request(inputs_from=not_request)
        with self.assertRaisesRegex(ValueError, "inputs_from_not_a_request"):
            ingress.validate_run_request(c, obj, obj["bundle_id"], REPO)


if __name__ == "__main__":
    unittest.main()
