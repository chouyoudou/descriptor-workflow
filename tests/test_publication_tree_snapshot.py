import base64
import hashlib
import unittest
import urllib.error
from unittest import mock

import bundle_control as bc
import private_io as pio
from git_publication_snapshot import existing_blob_snapshot, TreeSnapshotError


PARENT = "a" * 40
BASE_TREE = "b" * 40
NEW_TREE = "c" * 40
COMMIT = "d" * 40


def git_blob(raw):
    return hashlib.sha1(("blob %d\0" % len(raw)).encode() + raw).hexdigest()


class SnapshotHelperTests(unittest.TestCase):
    def test_complete_tree_returns_requested_blob_map(self):
        calls = []
        def api(path, method="GET", body=None):
            calls.append((path, method))
            if path.endswith("/git/commits/" + PARENT):
                return {"tree": {"sha": BASE_TREE}}
            if path.endswith("/git/trees/" + BASE_TREE + "?recursive=1"):
                return {"truncated": False, "tree": [
                    {"path": "x/a.txt", "type": "blob", "sha": "1" * 40},
                    {"path": "other.txt", "type": "blob", "sha": "2" * 40},
                ]}
            raise AssertionError(path)
        tree, got = existing_blob_snapshot(
            api, "owner/private", PARENT, ["x/a.txt", "x/missing.txt"]
        )
        self.assertEqual(tree, BASE_TREE)
        self.assertEqual(got, {"x/a.txt": "1" * 40, "x/missing.txt": None})
        self.assertEqual(len(calls), 2)

    def test_truncated_tree_requests_fallback(self):
        def api(path, method="GET", body=None):
            if path.endswith("/git/commits/" + PARENT):
                return {"tree": {"sha": BASE_TREE}}
            return {"truncated": True, "tree": []}
        tree, got = existing_blob_snapshot(api, "owner/private", PARENT, ["x"])
        self.assertEqual(tree, BASE_TREE)
        self.assertIsNone(got)

    def test_nonblob_existing_path_rejected(self):
        def api(path, method="GET", body=None):
            if path.endswith("/git/commits/" + PARENT):
                return {"tree": {"sha": BASE_TREE}}
            return {"truncated": False, "tree": [
                {"path": "x", "type": "tree", "sha": "3" * 40}
            ]}
        with self.assertRaisesRegex(TreeSnapshotError, "existing_path_not_blob"):
            existing_blob_snapshot(api, "owner/private", PARENT, ["x"])


class BundlePublisherBatchTests(unittest.TestCase):
    def fake_api(self, tree_entries):
        calls = []
        def api(path, method="GET", body=None):
            calls.append((path, method, body))
            if method == "GET" and path.endswith("/git/ref/heads/main"):
                return {"object": {"sha": PARENT}}
            if method == "GET" and path.endswith("/git/commits/" + PARENT):
                return {"tree": {"sha": BASE_TREE}}
            if method == "GET" and path.endswith("/git/trees/" + BASE_TREE + "?recursive=1"):
                return {"truncated": False, "tree": tree_entries}
            if method == "POST" and path.endswith("/git/blobs"):
                raw = base64.b64decode(body["content"])
                return {"sha": git_blob(raw)}
            if method == "POST" and path.endswith("/git/trees"):
                return {"sha": NEW_TREE}
            if method == "POST" and path.endswith("/git/commits"):
                return {"sha": COMMIT}
            if method == "PATCH" and path.endswith("/git/refs/heads/main"):
                return {}
            raise AssertionError((path, method, body))
        return api, calls

    def test_many_new_files_use_tree_snapshot_not_contents_queries(self):
        files = {f"transport/results/f{i}.bin": f"value-{i}".encode() for i in range(25)}
        api, calls = self.fake_api([])
        with mock.patch.object(bc, "api", side_effect=api), \
             mock.patch.object(bc, "content", side_effect=AssertionError("per-file query used")):
            got = bc.put_create_only("owner/private", files, "fixture")
        self.assertEqual(got, COMMIT)
        get_calls = [x for x in calls if x[1] == "GET"]
        self.assertEqual(len(get_calls), 3)  # main ref + commit tree + recursive tree
        self.assertEqual(len([x for x in calls if x[1] == "POST" and x[0].endswith("/git/blobs")]), 25)

    def test_same_existing_files_return_parent_without_new_commit(self):
        files = {"transport/results/a.bin": b"same", "transport/results/b.bin": b"same2"}
        entries = [
            {"path": path, "type": "blob", "sha": git_blob(raw)}
            for path, raw in files.items()
        ]
        api, calls = self.fake_api(entries)
        with mock.patch.object(bc, "api", side_effect=api), \
             mock.patch.object(bc, "content", side_effect=AssertionError("per-file query used")):
            got = bc.put_create_only("owner/private", files, "fixture")
        self.assertEqual(got, PARENT)
        self.assertFalse(any(x[1] == "POST" for x in calls))
        self.assertFalse(any(x[1] == "PATCH" for x in calls))

    def test_different_existing_file_still_refuses_overwrite(self):
        path = "transport/results/a.bin"
        api, calls = self.fake_api([
            {"path": path, "type": "blob", "sha": git_blob(b"old")}
        ])
        with mock.patch.object(bc, "api", side_effect=api), \
             mock.patch.object(bc, "content", side_effect=AssertionError("per-file query used")):
            with self.assertRaisesRegex(bc.BundleError, "refuse_different_existing_file"):
                bc.put_create_only("owner/private", {path: b"new"}, "fixture")
        self.assertFalse(any(x[1] == "POST" for x in calls))

    def test_truncated_tree_falls_back_to_existing_contents_checks(self):
        files = {"transport/results/a.bin": b"same", "transport/results/new.bin": b"new"}
        calls = []
        def api(path, method="GET", body=None):
            calls.append((path, method, body))
            if method == "GET" and path.endswith("/git/ref/heads/main"):
                return {"object": {"sha": PARENT}}
            if method == "GET" and path.endswith("/git/commits/" + PARENT):
                return {"tree": {"sha": BASE_TREE}}
            if method == "GET" and path.endswith("/git/trees/" + BASE_TREE + "?recursive=1"):
                return {"truncated": True, "tree": []}
            if method == "POST" and path.endswith("/git/blobs"):
                raw = base64.b64decode(body["content"]); return {"sha": git_blob(raw)}
            if method == "POST" and path.endswith("/git/trees"):
                return {"sha": NEW_TREE}
            if method == "POST" and path.endswith("/git/commits"):
                return {"sha": COMMIT}
            if method == "PATCH" and path.endswith("/git/refs/heads/main"):
                return {}
            raise AssertionError((path, method))
        missing = urllib.error.HTTPError("u", 404, "missing", {}, None)
        def content(repo, path, ref):
            if path.endswith("a.bin"):
                return {"sha": git_blob(b"same")}
            raise missing
        with mock.patch.object(bc, "api", side_effect=api), \
             mock.patch.object(bc, "content", side_effect=content) as contents:
            self.assertEqual(bc.put_create_only("owner/private", files, "fixture"), COMMIT)
        self.assertEqual(contents.call_count, 2)


class LegacyPublisherBatchTests(unittest.TestCase):
    def test_private_io_put_files_uses_same_fast_path(self):
        files = {f"transport/legacy/f{i}.bin": f"value-{i}".encode() for i in range(12)}
        calls = []
        def api(path, method="GET", body=None):
            calls.append((path, method, body))
            if method == "GET" and path.endswith("/git/ref/heads/main"):
                return {"object": {"sha": PARENT}}
            if method == "GET" and path.endswith("/git/commits/" + PARENT):
                return {"tree": {"sha": BASE_TREE}}
            if method == "GET" and path.endswith("/git/trees/" + BASE_TREE + "?recursive=1"):
                return {"truncated": False, "tree": []}
            if method == "POST" and path.endswith("/git/blobs"):
                raw = base64.b64decode(body["content"]); return {"sha": git_blob(raw)}
            if method == "POST" and path.endswith("/git/trees"):
                return {"sha": NEW_TREE}
            if method == "POST" and path.endswith("/git/commits"):
                return {"sha": COMMIT}
            if method == "PATCH" and path.endswith("/git/refs/heads/main"):
                return {}
            raise AssertionError((path, method, body))
        with mock.patch.object(pio, "api", side_effect=api), \
             mock.patch.object(pio, "content", side_effect=AssertionError("per-file query used")):
            self.assertEqual(pio.put_files("owner/private", files), COMMIT)
        self.assertEqual(len([x for x in calls if x[1] == "GET"]), 3)


if __name__ == "__main__":
    unittest.main()
