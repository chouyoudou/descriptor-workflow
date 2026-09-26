"""Focused tests for private result publication invariants and recovery refs."""
from __future__ import annotations

import json
import base64
import hashlib
from pathlib import Path
import threading
import tempfile
import unittest
import urllib.error
from types import SimpleNamespace
from unittest import mock

import private_io as pio
import bundle_control as bc


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
            if path.endswith("/git/trees/tree0?recursive=1") and method == "GET":
                return {"truncated": False, "tree": []}
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


class ProgressDiagnosticsTests(unittest.TestCase):
    def test_valid_progress_summary_reports_median_slowest_and_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "progress.jsonl"
            rows = [
                {"schema":"descriptor-progress/1","processed":1,"total":4,
                 "elapsed_seconds":2.0,"last_id":"a","item_seconds":2.0},
                {"schema":"descriptor-progress/1","processed":2,"total":4,
                 "elapsed_seconds":5.0,"last_id":"b","item_seconds":3.0},
                {"schema":"descriptor-progress/1","processed":3,"total":4,
                 "elapsed_seconds":9.0,"last_id":"c","item_seconds":4.0},
            ]
            p.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
            got = pio.inspect_progress_jsonl(
                p, {"timed_out": True, "wall_seconds": 12.5}
            )
            self.assertEqual(got["status"], "valid")
            self.assertEqual(got["records"], 3)
            self.assertEqual(got["processed"], 3)
            self.assertEqual(got["total"], 4)
            self.assertEqual(got["last_id"], "c")
            self.assertEqual(got["median_item_seconds"], 3.0)
            self.assertEqual(got["slowest_item_seconds"], 4.0)
            self.assertEqual(got["slowest_id"], "c")
            self.assertAlmostEqual(got["seconds_since_last_progress"], 3.5)
            self.assertEqual(got["timeout_pattern"], "timeout_recent_progress")

    def test_truncated_final_progress_line_preserves_prior_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "progress.jsonl"
            p.write_text(
                json.dumps({"schema":"descriptor-progress/1","processed":1,"total":3,
                            "elapsed_seconds":1.5,"last_id":"a","item_seconds":1.5})+"\n"+
                '{"schema":"descriptor-progress/1","processed":2',
                encoding="utf-8",
            )
            got = pio.inspect_progress_jsonl(
                p, {"timed_out": True, "wall_seconds": 130.0}
            )
            self.assertEqual(got["status"], "partial")
            self.assertTrue(got["truncated_final_line"])
            self.assertEqual(got["records"], 1)
            self.assertEqual(got["processed"], 1)
            self.assertEqual(got["last_id"], "a")
            self.assertEqual(got["timeout_pattern"], "timeout_long_gap_since_progress")

    def test_timeout_without_progress_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = pio.inspect_progress_jsonl(
                Path(tmp) / "missing.jsonl",
                {"timed_out": True, "wall_seconds": 30.0},
            )
            self.assertEqual(got["status"], "missing")
            self.assertEqual(got["timeout_pattern"], "timeout_no_progress_records")


class CandidateCpuBudgetTests(unittest.TestCase):
    def test_cleanup_timeout_preserves_result_and_failure_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=root/'task';task.mkdir();out=root/'out'
            result=SimpleNamespace(stdout=b'',stderr=b'',child_exit_code=0,timed_out=False,output_limit_exceeded=False)
            def run(*a,**k):
                (out/'result.jsonl').write_text('{"id":"preserved"}\n')
                return result
            calls=[SimpleNamespace(returncode=0,stderr=b''),pio.subprocess.TimeoutExpired(['docker','rm'],15)]
            with mock.patch.dict(pio.os.environ,{'GITHUB_RUN_ID':'123','GITHUB_RUN_ATTEMPT':'1'}),mock.patch.object(pio.subprocess,'run',side_effect=calls) as docker,mock.patch('restricted_exec.run_bounded',side_effect=run):
                with self.assertRaises(pio.PrivateIOError):pio.execute(task,out,'public-runtime:test',30)
            self.assertEqual((out/'result.jsonl').read_text(),'{"id":"preserved"}\n')
            status=json.loads((out/'execution.json').read_text())
            self.assertTrue(status['cleanup_timed_out']);self.assertEqual(status['candidate_exit_code'],0);self.assertNotEqual(status['exit_code'],0)
            self.assertEqual([c.kwargs['timeout'] for c in docker.call_args_list],[30,15])

    def test_create_timeout_still_produces_typed_execution_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=root/'task';task.mkdir();out=root/'out'
            calls=[pio.subprocess.TimeoutExpired(['docker','create'],30),SimpleNamespace(returncode=1,stderr=b'')]
            with mock.patch.dict(pio.os.environ,{'GITHUB_RUN_ID':'123','GITHUB_RUN_ATTEMPT':'1'}),mock.patch.object(pio.subprocess,'run',side_effect=calls),mock.patch('restricted_exec.run_bounded') as run:
                with self.assertRaises(pio.PrivateIOError):pio.execute(task,out,'public-runtime:test',30)
                run.assert_not_called()
            status=json.loads((out/'execution.json').read_text());self.assertTrue(status['timed_out']);self.assertEqual(status['host_failure_class'],'TimeoutExpired');self.assertNotEqual(status['exit_code'],0)

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
            self.assertIn("DESCRIPTOR_PROGRESS_PATH=/output/progress.jsonl", command)
            self.assertIn("DESCRIPTOR_TIMEOUT_SECONDS=30", command)
            status = json.loads((root / "out" / "execution.json").read_text())
            self.assertTrue(status["candidate_started"])
            self.assertEqual(status["timeout_budget_seconds"], 30)
            self.assertGreaterEqual(status["wall_seconds"], 0)
            self.assertFalse(any("PRIVATE_REPO_TOKEN" in argument for argument in command))


class BundleRecoveryTests(unittest.TestCase):
    def publish_fixture(self, out, *, exit_code=0, result=True, metadata=True):
        if metadata:
            _write_success_metadata(out, rows=2)
            (out / 'execution.json').write_text(json.dumps({
                'exit_code':exit_code,'timed_out':False,'output_limit_exceeded':False}))
            (out / 'focused-and-batch.log').write_text('fixture only\n')
        if result:
            (out / 'result.jsonl').write_text('{"id":"a"}\n{"id":"b"}\n')
        captured={}
        def put(repo,files,message,recovery_ref=None):
            captured.update(files);captured['__recovery_ref__']=recovery_ref
            return 'c'*40
        with mock.patch.object(bc,'put_create_only',side_effect=put), \
             mock.patch.dict(bc.os.environ,{'GITHUB_RUN_ID':'123','GITHUB_RUN_ATTEMPT':'1'}):
            if exit_code or not result or not metadata:
                with self.assertRaisesRegex(bc.BundleError,'failed_execution_preserved'):
                    bc.publish('owner/private','bt-fixture','b'*40,'a'*40,out)
            else:
                bc.publish('owner/private','bt-fixture','b'*40,'a'*40,out)
        return captured

    def test_failure_keeps_prefix_and_receipt_without_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);files=self.publish_fixture(out,exit_code=7)
            prefix='transport/bundle-executions/bt-fixture/123-1/'
            self.assertIn(prefix+'result.jsonl',files)
            receipt=json.loads(files[prefix+'receipt.json'])
            self.assertEqual(receipt['status'],'failed_or_partial')
            self.assertEqual(receipt['result_rows'],2)
            self.assertNotIn('transport/bundle-executions/bt-fixture/completed.json',files)
            self.assertRegex(files['__recovery_ref__'],r'^recovery-actions-123-1-[0-9a-f]{12}$')

    def test_timeout_progress_is_preserved_and_summarized_in_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_success_metadata(out, rows=2)
            (out / "execution.json").write_text(json.dumps({
                "exit_code":124,"timed_out":True,"output_limit_exceeded":False,
                "candidate_started":True,"wall_seconds":1800.0,
                "timeout_budget_seconds":1800})+"\n")
            (out / "focused-and-batch.log").write_text("fixture only\n")
            (out / "result.jsonl").write_text('{"id":"a"}\n{"id":"b"}\n')
            progress = [
                {"schema":"descriptor-progress/1","processed":1,"total":10,
                 "elapsed_seconds":2.0,"last_id":"a","item_seconds":2.0},
                {"schema":"descriptor-progress/1","processed":2,"total":10,
                 "elapsed_seconds":5.0,"last_id":"b","item_seconds":3.0},
            ]
            (out / "progress.jsonl").write_text(
                "".join(json.dumps(x)+"\n" for x in progress)
            )
            captured={}
            def put(repo,files,message,recovery_ref=None):
                captured.update(files); return "c"*40
            with mock.patch.object(bc,"put_create_only",side_effect=put), \
                 mock.patch.dict(bc.os.environ,{"GITHUB_RUN_ID":"123","GITHUB_RUN_ATTEMPT":"1"}):
                with self.assertRaisesRegex(bc.BundleError,"failed_execution_preserved"):
                    bc.publish("owner/private","bt-fixture","b"*40,"a"*40,out)
            prefix="transport/bundle-executions/bt-fixture/123-1/"
            self.assertIn(prefix+"progress.jsonl",captured)
            receipt=json.loads(captured[prefix+"receipt.json"])
            self.assertEqual(receipt["progress"]["processed"],2)
            self.assertEqual(receipt["progress"]["last_id"],"b")
            self.assertEqual(receipt["progress"]["median_item_seconds"],2.5)
            self.assertEqual(
                receipt["progress"]["timeout_pattern"],
                "timeout_long_gap_since_progress",
            )

    def test_success_still_requires_result_and_persists_complete_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            files=self.publish_fixture(Path(tmp))
            self.assertIn('transport/bundle-executions/bt-fixture/completed.json',files)
            queued=json.loads(files['transport/reviews/pending/bt-fixture/123-1.json'])
            self.assertEqual(queued['review_status'],'pending')
            self.assertEqual(queued['execution_status'],'materialized')
            self.assertEqual(queued['next_action'],'independent_review')
            self.assertEqual(queued['source_ref'],'b'*40)
        with tempfile.TemporaryDirectory() as tmp:
            files=self.publish_fixture(Path(tmp),result=False)
            self.assertNotIn('transport/bundle-executions/bt-fixture/completed.json',files)
            queued=json.loads(files['transport/reviews/pending/bt-fixture/123-1.json'])
            self.assertEqual(queued['review_status'],'pending')
            self.assertEqual(queued['next_action'],'execution_recovery')
            self.assertIsNone(queued['result_path'])

    def test_no_output_directory_still_gets_failure_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            files=self.publish_fixture(Path(tmp)/'absent',result=False,metadata=False)
            self.assertIn('transport/bundle-executions/bt-fixture/123-1/receipt.json',files)

    def test_large_pinned_blob_read_and_identity_rejection(self):
        raw=b'x'*(2*1024*1024+3);blob=pio.blob_sha(raw)
        metadata={'type':'file','encoding':'none','size':len(raw),'sha':blob}
        response={'encoding':'base64','sha':blob,'size':len(raw),
                  'content':base64.b64encode(raw).decode()}
        with mock.patch.object(bc,'content',return_value=metadata), \
             mock.patch.object(bc,'api',return_value=response) as get:
            self.assertEqual(bc.read_pinned_file('owner/private','transport/executions/old/result.jsonl',
                                               'f'*40,blob),raw)
            self.assertEqual(get.call_args.args[0],'/repos/owner/private/git/blobs/'+blob)
        with mock.patch.object(bc,'content',return_value={**metadata,'sha':'0'*40}), \
             mock.patch.object(bc,'api') as get:
            with self.assertRaisesRegex(bc.BundleError,'pinned_file_identity_mismatch'):
                bc.read_pinned_file('owner/private','inputs/x','f'*40,blob)
            get.assert_not_called()

    def test_bundle_publisher_accepts_task_defined_regular_outputs_and_no_focused_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            _write_success_metadata(out, rows=1)
            (out/'result.jsonl').write_text('{"id":"a"}\n')
            (out/'timings.json').write_text('{"stage_seconds":0.25}\n')
            (out/'diagnostics.txt').write_text('useful task-local evidence\n')
            captured={}
            def put(repo,files,message,recovery_ref=None):
                captured.update(files); return 'c'*40
            with mock.patch.object(bc,'put_create_only',side_effect=put), \
                 mock.patch.dict(bc.os.environ,{'GITHUB_RUN_ID':'123','GITHUB_RUN_ATTEMPT':'1'}):
                bc.publish('owner/private','bt-fixture','b'*40,'a'*40,out)
            prefix='transport/bundle-executions/bt-fixture/123-1/'
            self.assertIn(prefix+'timings.json',captured)
            self.assertIn(prefix+'diagnostics.txt',captured)
            receipt=json.loads(captured[prefix+'receipt.json'])
            self.assertEqual(receipt['status'],'materialized')
            self.assertNotIn('missing_focused_log',receipt['metadata_errors'])

    def test_legacy_publisher_accepts_task_defined_regular_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'out'; out.mkdir()
            state=_state(root)
            _write_success_metadata(out, rows=1)
            (out/'result.jsonl').write_text('{"id":"a"}\n')
            (out/'timings.json').write_text('{"seconds":0.1}\n')
            captured={}
            def put(repo,files,recovery_branch=None):
                captured.update(files)
            with mock.patch.object(pio,'put_files',side_effect=put):
                pio.publish(state,out)
            self.assertIn('transport/executions/test-request/12345-1/timings.json',captured)

    def test_already_completed_bundle_skips_without_new_git_write(self):
        code='print(1)\n';bundle={'schema':bc.SCHEMA,'bundle_id':'bt-fixture',
            'files':{'run_task.py':code},'sha256':{'run_task.py':hashlib.sha256(code.encode()).hexdigest()}}
        raw=json.dumps(bundle).encode();blob=pio.blob_sha(raw)
        done=json.dumps({'bundle_blob':blob,'status':'materialized','source_ref':'c'*40}).encode()
        def item(data):return {'type':'file','encoding':'base64','sha':pio.blob_sha(data),
                               'content':base64.b64encode(data).decode()}
        outputs={}
        with mock.patch.object(bc,'content',side_effect=[item(raw),item(done)]), \
             mock.patch.object(bc,'put_create_only') as put, \
             mock.patch.object(bc,'append_output',side_effect=lambda k,v:outputs.update({k:v})):
            bc.materialize('owner/private','bt-fixture')
        self.assertEqual(outputs['skip'],'true');put.assert_not_called()

    def test_retry_after_is_never_capped_to_an_early_retry(self):
        error=urllib.error.HTTPError('u',429,'limited',{'Retry-After':'600'},None)
        self.assertEqual(bc._retry_delay(error,0),600)
        self.assertGreaterEqual(bc._retry_delay(urllib.error.HTTPError('u',429,'limited',{},None),0),60)

    def test_create_only_parallelizes_new_blobs_but_keeps_commit_steps_serial(self):
        files = {f"transport/x/{i}.txt": f"value-{i}".encode() for i in range(4)}
        barrier = threading.Barrier(4)
        lock = threading.Lock()
        blob_threads = set()
        calls = []

        def fake_api(path, method="GET", body=None):
            with lock:
                calls.append((path, method))
            if path.endswith("/git/ref/heads/main") and method == "GET":
                return {"object": {"sha": "parent"}}
            if path.endswith("/git/commits/parent") and method == "GET":
                return {"tree": {"sha": "base"}}
            if path.endswith("/git/trees/base?recursive=1") and method == "GET":
                return {"truncated": False, "tree": []}
            if path.endswith("/git/blobs") and method == "POST":
                with lock:
                    blob_threads.add(threading.get_ident())
                barrier.wait(timeout=2)
                raw = base64.b64decode(body["content"])
                return {"sha": pio.blob_sha(raw)}
            if path.endswith("/git/trees") and method == "POST":
                self.assertEqual(len(blob_threads), 4)
                return {"sha": "tree"}
            if path.endswith("/git/commits") and method == "POST":
                return {"sha": "commit"}
            if path.endswith("/git/refs/heads/main") and method == "PATCH":
                return {"object": {"sha": "commit"}}
            raise AssertionError((path, method, body))

        with mock.patch.object(bc, "api", side_effect=fake_api), \
             mock.patch.object(pio, "_set_recovery_ref"), \
             mock.patch.object(pio, "_clear_recovery_ref"):
            commit = bc.put_create_only("owner/private", files, "parallel fixture")

        self.assertEqual(commit, "commit")
        self.assertEqual(len(blob_threads), 4)
        first_tree = next(i for i, item in enumerate(calls)
                          if item[0].endswith("/git/trees") and item[1] == "POST")
        blob_indices = [i for i, item in enumerate(calls)
                        if item[0].endswith("/git/blobs") and item[1] == "POST"]
        self.assertTrue(blob_indices)
        self.assertLess(max(blob_indices), first_tree)

    def test_parallel_blob_retry_reuses_successful_uploads(self):
        files = {"transport/x/a.txt": b"a", "transport/x/b.txt": b"b"}
        failure = urllib.error.HTTPError("u", 503, "retry", {}, None)
        lock = threading.Lock()
        attempts = {"a": 0, "b": 0}

        def fake_api(path, method="GET", body=None):
            if path.endswith("/git/ref/heads/main") and method == "GET":
                return {"object": {"sha": "parent"}}
            if path.endswith("/git/commits/parent") and method == "GET":
                return {"tree": {"sha": "base"}}
            if path.endswith("/git/trees/base?recursive=1") and method == "GET":
                return {"truncated": False, "tree": []}
            if path.endswith("/git/blobs") and method == "POST":
                raw = base64.b64decode(body["content"])
                key = raw.decode()
                with lock:
                    attempts[key] += 1
                    n = attempts[key]
                if key == "b" and n == 1:
                    raise failure
                return {"sha": pio.blob_sha(raw)}
            if path.endswith("/git/trees") and method == "POST":
                return {"sha": "tree"}
            if path.endswith("/git/commits") and method == "POST":
                return {"sha": "commit"}
            if path.endswith("/git/refs/heads/main") and method == "PATCH":
                return {"object": {"sha": "commit"}}
            raise AssertionError((path, method, body))

        with mock.patch.object(bc, "api", side_effect=fake_api), \
             mock.patch.object(bc.time, "sleep"), \
             mock.patch.object(pio, "_set_recovery_ref"), \
             mock.patch.object(pio, "_clear_recovery_ref"):
            self.assertEqual(
                bc.put_create_only("owner/private", files, "retry fixture"),
                "commit",
            )

        self.assertEqual(attempts["a"], 1)
        self.assertEqual(attempts["b"], 2)
    def test_recovery_ref_precedes_main_and_generic_422_does_not_retry(self):
        missing=urllib.error.HTTPError('u',404,'missing',{},None);calls=[]
        def api(path,method='GET',body=None):
            calls.append((path,method))
            if path.endswith('/git/ref/heads/main'):return {'object':{'sha':'parent'}}
            if path.endswith('/git/blobs'):return {'sha':pio.blob_sha(b'fixture')}
            if path.endswith('/git/commits/parent'):return {'tree':{'sha':'base'}}
            if path.endswith('/git/trees/base?recursive=1'):return {'truncated':False,'tree':[]}
            if path.endswith('/git/trees'):return {'sha':'tree'}
            if path.endswith('/git/commits'):return {'sha':'commit'}
            if path.endswith('/git/refs/heads/main'):return {'object':{'sha':'commit'}}
            raise AssertionError(path)
        def recover(repo,branch,commit):calls.append(('recovery','POST'))
        with mock.patch.object(bc,'content',side_effect=missing),mock.patch.object(bc,'api',side_effect=api), \
             mock.patch.object(pio,'_set_recovery_ref',side_effect=recover),mock.patch.object(pio,'_clear_recovery_ref'):
            bc.put_create_only('owner/private',{'transport/x':b'fixture'},'fixture',
                               recovery_ref='recovery-actions-123-1-0123456789ab')
        self.assertLess(calls.index(('recovery','POST')),calls.index(('/repos/owner/private/git/refs/heads/main','PATCH')))
        error=urllib.error.HTTPError('u',422,'invalid',{},None)
        with mock.patch.object(bc,'api',side_effect=error) as api,mock.patch.object(bc.time,'sleep') as sleep:
            with self.assertRaises(urllib.error.HTTPError):bc.put_create_only('owner/private',{'transport/x':b'x'},'x')
            self.assertEqual(api.call_count,1);sleep.assert_not_called()


class TextBundleIdentityTests(unittest.TestCase):
    def bundle(self):
        return {"schema": bc.TEXT_SCHEMA, "bundle_id": "bt-text-fixture",
                "files": {"run_task.py": "\n# Unicode: 晶体\nprint('ok')\n"},
                "inputs": {}, "timeout_seconds": 30}

    def item(self, data):
        return {"type": "file", "encoding": "base64", "sha": pio.blob_sha(data),
                "content": base64.b64encode(data).decode()}

    def test_v2_still_requires_exact_manifest_v3_preserves_text(self):
        bundle = self.bundle()
        raw = json.dumps(bundle).encode()
        _, sources, _, _ = bc.validate_bundle(raw, bundle["bundle_id"])
        self.assertEqual(sources["run_task.py"], bundle["files"]["run_task.py"].encode())
        bundle["schema"] = bc.SCHEMA
        with self.assertRaisesRegex(bc.BundleError, "invalid_source_manifest"):
            bc.validate_bundle(json.dumps(bundle), bundle["bundle_id"])
        code = bundle["files"]["run_task.py"]
        bundle["sha256"] = {"run_task.py": hashlib.sha256(code.lstrip().encode()).hexdigest()}
        with self.assertRaisesRegex(bc.BundleError, "source_hash_mismatch"):
            bc.validate_bundle(json.dumps(bundle), bundle["bundle_id"])
        bundle["sha256"]["run_task.py"] = hashlib.sha256(code.encode()).hexdigest()
        self.assertEqual(bc.validate_bundle(json.dumps(bundle), bundle["bundle_id"])[1], sources)

    def test_v3_hash_and_path_safety_remain_but_source_caps_are_removed(self):
        for hashes in (None, {}, {"run_task.py": "0" * 64}):
            bundle = self.bundle(); bundle["sha256"] = hashes
            with self.subTest(hashes=hashes), self.assertRaises(bc.BundleError):
                bc.validate_bundle(json.dumps(bundle), bundle["bundle_id"])

        # Former project caps (12 files, 128 KiB/file, 512 KiB total, ~1 MiB
        # bundle envelope) are deliberately gone. Platform limits now govern.
        bundle = self.bundle()
        bundle["files"] = {"run_task.py": "x" * (1024 * 1024 + 17)}
        bundle["files"].update({f"module_{i}.py": "y" * 4096 for i in range(20)})
        raw = json.dumps(bundle).encode()
        self.assertGreater(len(raw), 1024 * 1024)
        parsed = bc.validate_bundle(raw, bundle["bundle_id"])[1]
        self.assertEqual(len(parsed), 21)
        self.assertEqual(len(parsed["run_task.py"]), 1024 * 1024 + 17)

        for files in ({"run_task.py": "", "../escape.py": ""},
                      {"run_task.py": "", "BUNDLE_SOURCE.json": ""}):
            bundle = self.bundle(); bundle["files"] = files
            with self.subTest(files=list(files)), self.assertRaises(bc.BundleError):
                bc.validate_bundle(json.dumps(bundle), bundle["bundle_id"])
        bundle = self.bundle()
        bundle["inputs"] = {"x": {"path": "outside/asset", "ref": "a"*40, "blob": "b"*40}}
        with self.assertRaisesRegex(bc.BundleError, "input_path_not_allowed"):
            bc.validate_bundle(json.dumps(bundle), bundle["bundle_id"])

    def test_v3_host_derives_manifest_and_fetch_rechecks_exact_source(self):
        bundle = self.bundle(); raw = json.dumps(bundle).encode()
        path = "transport/bundle_inbox/" + bundle["bundle_id"] + ".json"
        stored = {path: raw}; outputs = {}; ref = "a" * 40
        def content(repo, name, revision):
            if name not in stored:
                raise urllib.error.HTTPError("u", 404, "missing", {}, None)
            return self.item(stored[name])
        def put(repo, files, message, recovery_ref=None):
            stored.update(files); return ref
        with mock.patch.object(bc, "content", side_effect=content), \
             mock.patch.object(bc, "put_create_only", side_effect=put), \
             mock.patch.object(bc, "append_output", side_effect=lambda k,v: outputs.update({k:v})):
            bc.materialize("owner/private", bundle["bundle_id"])
            prefix = "transport/tasks/bundle-materialized/" + bundle["bundle_id"] + "/"
            manifest = json.loads(stored[prefix + "BUNDLE_SOURCE.json"])
            code = bundle["files"]["run_task.py"].encode()
            self.assertEqual(manifest["files"]["run_task.py"], hashlib.sha256(code).hexdigest())
            self.assertEqual(manifest["bundle_blob"], pio.blob_sha(raw))
            self.assertEqual(manifest["submission_schema"], bc.TEXT_SCHEMA)
            self.assertEqual(manifest["source_hash_authority"], "actions_received_utf8")
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "task"
                bc.fetch_task("owner/private", bundle["bundle_id"], ref, target)
                self.assertEqual((target / "run_task.py").read_bytes(), code)
                stored[prefix + "run_task.py"] = code + b"# altered\n"
                with self.assertRaisesRegex(bc.BundleError, "source_fetch_hash_mismatch"):
                    bc.fetch_task("owner/private", bundle["bundle_id"], ref, Path(tmp) / "bad")
        self.assertEqual(outputs["skip"], "false")

    def test_completed_v3_skips_but_changed_same_id_is_rejected(self):
        bundle = self.bundle(); raw = json.dumps(bundle).encode()
        done = json.dumps({"bundle_blob": pio.blob_sha(raw), "status": "materialized",
                           "source_ref": "c"*40}).encode()
        outputs = {}
        with mock.patch.object(bc, "content", side_effect=[self.item(raw), self.item(done)]), \
             mock.patch.object(bc, "put_create_only") as put, \
             mock.patch.object(bc, "append_output", side_effect=lambda k,v: outputs.update({k:v})):
            bc.materialize("owner/private", bundle["bundle_id"])
            put.assert_not_called()
        self.assertEqual(outputs["skip"], "true")
        bundle["files"]["run_task.py"] += "# different\n"
        with mock.patch.object(bc, "content", side_effect=[self.item(json.dumps(bundle).encode()), self.item(done)]), \
             mock.patch.object(bc, "put_create_only") as put:
            with self.assertRaisesRegex(bc.BundleError, "completed_bundle_identity_changed"):
                bc.materialize("owner/private", bundle["bundle_id"])
            put.assert_not_called()


if __name__ == "__main__":
    unittest.main()
