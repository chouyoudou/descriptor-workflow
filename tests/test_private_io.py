"""Focused tests for private result publication invariants and recovery refs."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from types import SimpleNamespace
from unittest import mock

import private_io as pio


def _state(root: Path) -> Path:
    state = {
        "repository": "owner/private",
        "task_blob": "a" * 40,
        "run_id": "12345",
        "run_attempt": "1",
        "destination": "transport/executions/test-request/12345-1",
        "task": {
            "source_ref": "b" * 40,
            "output_prefix": "transport/executions/test-request",
        },
    }
    path = root / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


def _write_success_metadata(out: Path, rows: int = 1) -> None:
    (out / "summary.json").write_text(
        json.dumps({"stage": "materialized", "rows": rows}) + "\n",
        encoding="utf-8",
    )
    (out / "execution.json").write_text(
        json.dumps({"exit_code": 0, "timed_out": False,
                    "output_limit_exceeded": False}) + "\n",
        encoding="utf-8",
    )


class ResultValidationTests(unittest.TestCase):
    def test_missing_result_cannot_create_completed_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            out.mkdir()
            state = _state(root)
            _write_success_metadata(out, rows=1)
            captured = {}

            def fake_put(repo, files, recovery_branch=None):
                captured.update(files)
                captured["__recovery_branch__"] = recovery_branch

            with mock.patch.object(pio, "put_files", side_effect=fake_put):
                with self.assertRaisesRegex(
                    pio.PrivateIOError, "materialized_result_invalid"
                ):
                    pio.publish(state, out)

            self.assertNotIn(
                "transport/executions/test-request/completed.json", captured
            )
            receipt = json.loads(
                captured[
                    "transport/executions/test-request/12345-1/receipt.json"
                ].decode("utf-8")
            )
            self.assertEqual(receipt["result_validation"]["status"], "missing")
            self.assertRegex(
                captured["__recovery_branch__"],
                r"^recovery-actions-12345-1-[0-9a-f]{12}$",
            )

    def test_malformed_result_is_persisted_for_diagnosis_but_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            out.mkdir()
            state = _state(root)
            _write_success_metadata(out, rows=1)
            (out / "result.jsonl").write_text("{not-json}\n", encoding="utf-8")
            captured = {}

            with mock.patch.object(
                pio, "put_files",
                side_effect=lambda repo, files, recovery_branch=None: captured.update(files),
            ):
                with self.assertRaisesRegex(
                    pio.PrivateIOError, "materialized_result_invalid"
                ):
                    pio.publish(state, out)

            self.assertIn(
                "transport/executions/test-request/12345-1/result.jsonl", captured
            )
            self.assertNotIn(
                "transport/executions/test-request/completed.json", captured
            )
            receipt = json.loads(
                captured[
                    "transport/executions/test-request/12345-1/receipt.json"
                ].decode("utf-8")
            )
            self.assertEqual(receipt["result_validation"]["status"], "invalid")
            self.assertEqual(receipt["result_validation"]["reason"], "invalid_json")

    def test_valid_nonempty_jsonl_is_required_for_completed_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            out.mkdir()
            state = _state(root)
            _write_success_metadata(out, rows=2)
            (out / "result.jsonl").write_text(
                '{"id":"a","value":1}\n{"id":"b","value":2}\n',
                encoding="utf-8",
            )
            captured = {}

            with mock.patch.object(
                pio, "put_files",
                side_effect=lambda repo, files, recovery_branch=None: captured.update(files),
            ):
                pio.publish(state, out)

            self.assertIn(
                "transport/executions/test-request/completed.json", captured
            )
            receipt = json.loads(
                captured[
                    "transport/executions/test-request/12345-1/receipt.json"
                ].decode("utf-8")
            )
            self.assertEqual(
                receipt["result_validation"], {"status": "valid", "records": 2}
            )

    def test_summary_row_count_mismatch_prevents_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            out.mkdir()
            state = _state(root)
            _write_success_metadata(out, rows=2)
            (out / "result.jsonl").write_text('{"id":"only"}\n', encoding="utf-8")
            captured = {}

            with mock.patch.object(
                pio, "put_files",
                side_effect=lambda repo, files, recovery_branch=None: captured.update(files),
            ):
                with self.assertRaisesRegex(
                    pio.PrivateIOError, "materialized_result_invalid"
                ):
                    pio.publish(state, out)

            receipt = json.loads(
                captured[
                    "transport/executions/test-request/12345-1/receipt.json"
                ].decode("utf-8")
            )
            self.assertEqual(
                receipt["result_validation"]["reason"], "row_count_mismatch"
            )
            self.assertNotIn(
                "transport/executions/test-request/completed.json", captured
            )


class RecoveryRefTests(unittest.TestCase):
    def test_recovery_ref_is_created_before_main_move_and_cleared_after(self):
        calls = []
        missing = urllib.error.HTTPError("u", 404, "missing", {}, None)

        def fake_content(repo, path, ref):
            raise missing

        def fake_api(path, method="GET", body=None):
            calls.append((path, method, body))
            if path.endswith("/git/ref/heads/main") and method == "GET":
                return {"object": {"sha": "parent"}}
            if path.endswith("/git/blobs") and method == "POST":
                return {"sha": "blob"}
            if path.endswith("/git/commits/parent") and method == "GET":
                return {"tree": {"sha": "tree0"}}
            if path.endswith("/git/trees") and method == "POST":
                return {"sha": "tree1"}
            if path.endswith("/git/commits") and method == "POST":
                return {"sha": "commit1"}
            if "/git/ref/heads/recovery-actions-" in path and method == "GET":
                raise missing
            if path.endswith("/git/refs") and method == "POST":
                return {"ref": body["ref"]}
            if path.endswith("/git/refs/heads/main") and method == "PATCH":
                return {"object": {"sha": body["sha"]}}
            if "/git/refs/heads/recovery-actions-" in path and method == "DELETE":
                return None
            raise AssertionError((path, method, body))

        branch = "recovery-actions-12345-1-0123456789ab"
        with mock.patch.object(pio, "content", side_effect=fake_content), \
             mock.patch.object(pio, "api", side_effect=fake_api):
            pio.put_files(
                "owner/private",
                {"transport/executions/x/result.jsonl": b'{"id":"a"}\n'},
                recovery_branch=branch,
            )

        create_ref_index = next(
            i for i, item in enumerate(calls)
            if item[0].endswith("/git/refs") and item[1] == "POST"
        )
        main_move_index = next(
            i for i, item in enumerate(calls)
            if item[0].endswith("/git/refs/heads/main") and item[1] == "PATCH"
        )
        clear_ref_index = next(
            i for i, item in enumerate(calls)
            if item[0].endswith(branch) and item[1] == "DELETE"
        )
        self.assertLess(create_ref_index, main_move_index)
        self.assertLess(main_move_index, clear_ref_index)


class CandidateCpuBudgetTests(unittest.TestCase):
    def test_four_cpu_budget_and_worker_hint_without_nested_blas_threads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "task"
            task.mkdir()
            result = SimpleNamespace(stdout=b"", stderr=b"", child_exit_code=0,
                                     timed_out=False, output_limit_exceeded=False)
            with mock.patch.dict(pio.os.environ, {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}), \
                 mock.patch.object(pio.subprocess, "run", return_value=SimpleNamespace(returncode=0, stderr=b"")) as docker, \
                 mock.patch("restricted_exec.run_bounded", return_value=result):
                pio.execute(task, root / "out", "public-runtime:test", 30)
            command = docker.call_args_list[0].args[0]
            self.assertEqual(command[command.index("--cpus") + 1], "4")
            self.assertEqual(command[command.index("--memory") + 1], "4g")
            self.assertEqual(command[command.index("--network") + 1], "none")
            self.assertIn("DESCRIPTOR_WORKERS=4", command)
            self.assertIn("OPENBLAS_NUM_THREADS=1", command)
            self.assertIn("OMP_NUM_THREADS=1", command)
            self.assertFalse(any("PRIVATE_REPO_TOKEN" in argument for argument in command))


if __name__ == "__main__":
    unittest.main()
