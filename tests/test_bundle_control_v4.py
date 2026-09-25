import base64
import hashlib
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import bundle_control as bc


def git_blob(raw):
    header = ("blob %d" % len(raw)).encode() + bytes([0])
    return hashlib.sha1(header + raw).hexdigest()


def item(raw):
    return {
        "type": "file",
        "encoding": "base64",
        "content": base64.b64encode(raw).decode(),
        "sha": git_blob(raw),
        "size": len(raw),
    }


class BundleV4ValidationTests(unittest.TestCase):
    def test_v4_minimal_validation(self):
        ref = "a" * 40
        raw = json.dumps({
            "schema": bc.SUCCESSOR_SCHEMA,
            "bundle_id": "bt-v4-unit",
            "base_source_ref": ref,
            "changed_files": {"helper.py": "VALUE = 2\n"},
            "delete_files": ["old.py"],
            "inputs": {},
            "timeout_seconds": 60,
        })
        bundle, changed, inputs, timeout = bc.validate_bundle(raw, "bt-v4-unit")
        self.assertEqual(bundle["base_source_ref"], ref)
        self.assertEqual(changed, {"helper.py": b"VALUE = 2\n"})
        self.assertEqual(inputs, {})
        self.assertEqual(timeout, 60)

    def test_v4_rejects_ambiguous_or_invalid_edits(self):
        ref = "b" * 40
        base = {
            "schema": bc.SUCCESSOR_SCHEMA,
            "bundle_id": "bt-v4-invalid",
            "base_source_ref": ref,
            "inputs": {},
            "timeout_seconds": 30,
        }
        cases = [
            ({**base, "changed_files": {}, "delete_files": []}, "successor_no_requested_changes"),
            ({**base, "changed_files": {"same.py": "x"}, "delete_files": ["same.py"]}, "successor_change_delete_overlap"),
            ({**base, "changed_files": {}, "delete_files": ["run_task.py"]}, "cannot_delete_run_task"),
            ({**base, "changed_files": {"../bad.py": "x"}, "delete_files": []}, "invalid_changed_source_entry"),
            ({**base, "files": {"run_task.py": "x"}, "changed_files": {"x.py": "x"}, "delete_files": []}, "successor_ambiguous_source_fields"),
        ]
        for spec, code in cases:
            with self.subTest(code=code):
                with self.assertRaisesRegex(bc.BundleError, "^" + code):
                    bc.validate_bundle(json.dumps(spec), "bt-v4-invalid")


class SuccessorCompositionTests(unittest.TestCase):
    def test_overlay_delete_inherit_and_noop_detection(self):
        base = {
            "run_task.py": b"old-run\n",
            "helper.py": b"old-helper\n",
            "same.py": b"same\n",
        }
        final, meta = bc._apply_successor_sources(
            base,
            {"run_task.py": b"new-run\n", "same.py": b"same\n", "new.py": b"new\n"},
            ["helper.py"],
        )
        self.assertEqual(final, {
            "run_task.py": b"new-run\n",
            "same.py": b"same\n",
            "new.py": b"new\n",
        })
        self.assertEqual(meta["actual_changed_files"], ["new.py", "run_task.py"])
        self.assertEqual(meta["unchanged_overlay_files"], ["same.py"])
        self.assertEqual(meta["deleted_files"], ["helper.py"])
        self.assertEqual(meta["inherited_files"], ["same.py"])

        with self.assertRaisesRegex(bc.BundleError, "^successor_no_effect"):
            bc._apply_successor_sources(base, {"same.py": b"same\n"}, [])
        with self.assertRaisesRegex(bc.BundleError, "^successor_delete_missing_file"):
            bc._apply_successor_sources(base, {}, ["missing.py"])

    def test_final_source_must_retain_run_task(self):
        with self.assertRaisesRegex(bc.BundleError, "^invalid_final_source_manifest"):
            bc._apply_successor_sources(
                {"run_task.py": b"x\n", "helper.py": b"y\n"}, {}, ["run_task.py"]
            )


class BaseSourceResolutionTests(unittest.TestCase):
    def setUp(self):
        self.ref = "c" * 40
        self.bundle_id = "bt-base-unit"
        self.prefix = f"transport/tasks/bundle-materialized/{self.bundle_id}"
        self.sources = {
            "run_task.py": b"print('ok')\n",
            "helper.py": b"VALUE = 1\n",
        }
        self.manifest = {
            "schema": bc.SOURCE_SCHEMA,
            "bundle_id": self.bundle_id,
            "files": {k: hashlib.sha256(v).hexdigest() for k, v in self.sources.items()},
        }
        self.manifest_raw = (json.dumps(self.manifest) + "\n").encode()
        self.manifest_path = self.prefix + "/BUNDLE_SOURCE.json"
        self.items = {self.manifest_path: item(self.manifest_raw)}
        for name, raw in self.sources.items():
            self.items[f"{self.prefix}/{name}"] = item(raw)

    def commit(self, extra=None):
        files = [{"filename": self.manifest_path}]
        files += [{"filename": f"{self.prefix}/{name}"} for name in self.sources]
        files += list(extra or [])
        return {"sha": self.ref, "files": files}

    def test_resolves_materialized_source_from_ref_only(self):
        with mock.patch.object(bc, "api", return_value=self.commit()), \
             mock.patch.object(bc, "content", side_effect=lambda repo, path, ref: self.items[path]):
            got = bc._load_base_source("owner/private", self.ref)
        self.assertEqual(got["bundle_id"], self.bundle_id)
        self.assertEqual(got["sources"], self.sources)
        self.assertEqual(
            got["blob_shas"],
            {name: git_blob(raw) for name, raw in self.sources.items()},
        )

    def test_rejects_commit_with_unrelated_changes(self):
        with mock.patch.object(
            bc, "api", return_value=self.commit([{"filename": "unrelated.txt"}])
        ), mock.patch.object(
            bc, "content", side_effect=lambda repo, path, ref: self.items[path]
        ):
            with self.assertRaisesRegex(bc.BundleError, "^base_source_commit_scope_mismatch"):
                bc._load_base_source("owner/private", self.ref)


class BlobReuseTests(unittest.TestCase):
    def test_put_create_only_reuses_verified_inherited_blob(self):
        repo = "owner/private"
        parent = "d" * 40
        tree = "e" * 40
        new_tree = "f" * 40
        commit = "1" * 40
        raw_a, raw_b = b"same-base\n", b"new-change\n"
        files = {"dest/a.py": raw_a, "dest/b.py": raw_b}
        known = {"dest/a.py": git_blob(raw_a)}
        calls = []

        def missing(*args, **kwargs):
            raise urllib.error.HTTPError("", 404, "missing", None, None)

        def fake_api(path, method="GET", body=None):
            calls.append((path, method, body))
            if method == "GET" and path.endswith("/git/ref/heads/main"):
                return {"object": {"sha": parent}}
            if method == "POST" and path.endswith("/git/blobs"):
                raw = base64.b64decode(body["content"])
                return {"sha": git_blob(raw)}
            if method == "GET" and path.endswith("/git/commits/" + parent):
                return {"tree": {"sha": tree}}
            if method == "POST" and path.endswith("/git/trees"):
                return {"sha": new_tree}
            if method == "POST" and path.endswith("/git/commits"):
                return {"sha": commit}
            if method == "PATCH" and path.endswith("/git/refs/heads/main"):
                return {}
            raise AssertionError((path, method, body))

        with mock.patch.object(bc, "content", side_effect=missing), \
             mock.patch.object(bc, "api", side_effect=fake_api):
            got = bc.put_create_only(repo, files, "unit", known_blob_shas=known)

        self.assertEqual(got, commit)
        blob_posts = [c for c in calls if c[1] == "POST" and c[0].endswith("/git/blobs")]
        self.assertEqual(len(blob_posts), 1)
        tree_post = next(c for c in calls if c[1] == "POST" and c[0].endswith("/git/trees"))
        entries = {e["path"]: e["sha"] for e in tree_post[2]["tree"]}
        self.assertEqual(entries["dest/a.py"], git_blob(raw_a))
        self.assertEqual(entries["dest/b.py"], git_blob(raw_b))

    def test_rejects_wrong_known_blob_identity(self):
        with self.assertRaisesRegex(bc.BundleError, "^known_blob_identity_mismatch"):
            bc.put_create_only(
                "owner/private",
                {"a.py": b"x"},
                "unit",
                known_blob_shas={"a.py": "0" * 40},
            )


if __name__ == "__main__":
    unittest.main()
