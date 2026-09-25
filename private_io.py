"""Scoped private-file I/O around public Actions execution. No scientific definitions."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import random
import re
import stat
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

class PrivateIOError(RuntimeError):
    pass

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def api(path, method="GET", body=None):
    token = os.environ.get("PRIVATE_REPO_TOKEN", "").strip()
    if not token:
        raise PrivateIOError("missing_private_token")
    request = urllib.request.Request(
        "https://api.github.com" + path,
        data=None if body is None else json.dumps(body, separators=(",", ":")).encode(),
        method=method,
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "User-Agent": "scoped-private-workflow"})
    with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
        raw = response.read(64 * 1024 * 1024 + 1)
        if len(raw) > 64 * 1024 * 1024:
            raise PrivateIOError("response_too_large")
        return None if not raw else json.loads(raw)

def safe_repo(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise PrivateIOError("invalid_repository")
    return value

def safe_path(value):
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value or ".." in value.split("/"):
        raise PrivateIOError("invalid_path")
    return value

def blob_sha(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()

def content(repo, path, ref):
    path = urllib.parse.quote(safe_path(path), safe="/")
    return api(f"/repos/{repo}/contents/{path}?ref={urllib.parse.quote(ref, safe='')}")

def decode_file(item, max_bytes=2 * 1024 * 1024):
    if item.get("type") != "file" or item.get("encoding") != "base64":
        raise PrivateIOError("expected_small_file")
    raw = base64.b64decode(item["content"])
    if len(raw) > max_bytes or blob_sha(raw) != item["sha"]:
        raise PrivateIOError("file_identity_or_size")
    return raw

def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    path.chmod(0o600)

def append_github_output(key, value):
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(f"{key}={value}\n")

def _recovery_branch_name(state):
    digest = hashlib.sha256(state["destination"].encode("utf-8")).hexdigest()[:12]
    run, attempt = str(state["run_id"]), str(state["run_attempt"])
    if not run.isdigit() or not attempt.isdigit():
        raise PrivateIOError("invalid_run_identity")
    return f"recovery-actions-{run}-{attempt}-{digest}"

def _validate_recovery_branch(branch):
    if not re.fullmatch(r"recovery-actions-[0-9]+-[0-9]+-[0-9a-f]{12}", branch or ""):
        raise PrivateIOError("invalid_recovery_branch")
    return branch

def _set_recovery_ref(repo, branch, commit_sha):
    branch = _validate_recovery_branch(branch)
    get_path = f"/repos/{repo}/git/ref/heads/{branch}"
    patch_path = f"/repos/{repo}/git/refs/heads/{branch}"
    try:
        existing = api(get_path)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        api(f"/repos/{repo}/git/refs", "POST",
            {"ref": f"refs/heads/{branch}", "sha": commit_sha})
    else:
        if existing["object"]["sha"] != commit_sha:
            api(patch_path, "PATCH", {"sha": commit_sha, "force": True})
    print(f"RECOVERY_STAGE=private-ref-ready REF={branch}")

def _clear_recovery_ref(repo, branch):
    branch = _validate_recovery_branch(branch)
    path = f"/repos/{repo}/git/refs/heads/{branch}"
    try:
        api(path, "DELETE")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return
        print(f"RECOVERY_STAGE=private-ref-retained REF={branch}", file=sys.stderr)
    except (urllib.error.URLError, TimeoutError):
        print(f"RECOVERY_STAGE=private-ref-retained REF={branch}", file=sys.stderr)
    else:
        print(f"RECOVERY_STAGE=private-ref-cleared REF={branch}")

def put_files(repo, files, recovery_branch=None):
    # Read/check/build against one immutable parent; never force or overwrite main.
    # Blob SHAs are cached across retries to reduce API traffic under concurrent writers.
    if recovery_branch is not None:
        recovery_branch = _validate_recovery_branch(recovery_branch)
    blob_shas = {}
    for path, raw in files.items():
        safe_path(path)
        if not isinstance(raw, (bytes, bytearray)):
            raise PrivateIOError("invalid_file_bytes")
    for attempt in range(7):
        try:
            parent = api(f"/repos/{repo}/git/ref/heads/main")["object"]["sha"]
            entries = []
            for path, raw in files.items():
                try:
                    old = content(repo, path, parent)
                except urllib.error.HTTPError as exc:
                    if exc.code != 404:
                        raise
                else:
                    if old["sha"] != blob_sha(raw):
                        raise PrivateIOError("refuse_different_existing_file")
                    continue
                if path not in blob_shas:
                    blob_shas[path] = api(
                        f"/repos/{repo}/git/blobs", "POST",
                        {"content": base64.b64encode(raw).decode(), "encoding": "base64"}
                    )["sha"]
                entries.append({"path": path, "mode": "100644", "type": "blob",
                                "sha": blob_shas[path]})
            if not entries:
                if recovery_branch is not None:
                    _clear_recovery_ref(repo, recovery_branch)
                return parent
            tree = api(f"/repos/{repo}/git/commits/{parent}")["tree"]["sha"]
            new_tree = api(f"/repos/{repo}/git/trees", "POST",
                           {"base_tree": tree, "tree": entries})["sha"]
            commit = api(f"/repos/{repo}/git/commits", "POST",
                         {"message": "Record scoped computation output",
                          "tree": new_tree, "parents": [parent]})["sha"]
            if recovery_branch is not None:
                _set_recovery_ref(repo, recovery_branch, commit)
            api(f"/repos/{repo}/git/refs/heads/main", "PATCH",
                {"sha": commit, "force": False})
            if recovery_branch is not None:
                _clear_recovery_ref(repo, recovery_branch)
            return commit
        except urllib.error.HTTPError as exc:
            if exc.code not in {409, 422, 429, 500, 502, 503, 504} or attempt == 6:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 6:
                raise
        time.sleep(min(2 ** attempt, 16) + random.uniform(0.0, 0.5))
    raise PrivateIOError("publication_failed")

def _reject_json_constant(value):
    raise ValueError("nonfinite_json_constant")

def inspect_result_jsonl(path, expected_rows=None):
    path = Path(path)
    if not path.exists():
        return {"status": "missing", "records": 0}
    try:
        st = path.lstat()
    except OSError:
        return {"status": "invalid", "reason": "stat_failed", "records": 0}
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        return {"status": "invalid", "reason": "not_regular_file", "records": 0}
    if st.st_size <= 0:
        return {"status": "invalid", "reason": "empty", "records": 0}
    records = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    return {"status": "invalid", "reason": "blank_line",
                            "records": records, "line": line_number}
                try:
                    row = json.loads(line, parse_constant=_reject_json_constant)
                except (json.JSONDecodeError, ValueError, TypeError):
                    return {"status": "invalid", "reason": "invalid_json",
                            "records": records, "line": line_number}
                if not isinstance(row, dict):
                    return {"status": "invalid", "reason": "non_object_row",
                            "records": records, "line": line_number}
                records += 1
    except (OSError, UnicodeError):
        return {"status": "invalid", "reason": "read_failed", "records": records}
    if records == 0:
        return {"status": "invalid", "reason": "empty", "records": 0}
    if type(expected_rows) is int and expected_rows >= 0 and records != expected_rows:
        return {"status": "invalid", "reason": "row_count_mismatch",
                "records": records, "expected_rows": expected_rows}
    return {"status": "valid", "records": records}

def _finite_nonnegative_number(value):
    return (type(value) in (int, float)
            and value >= 0
            and value != float("inf")
            and value != float("-inf")
            and value == value)

def inspect_progress_jsonl(path, execution=None):
    """Summarize optional candidate progress while tolerating a truncated final line."""
    path = Path(path)
    execution = execution if isinstance(execution, dict) else {}
    if not path.exists():
        result = {"status": "missing", "records": 0}
        if execution.get("timed_out"):
            result["timeout_pattern"] = "timeout_no_progress_records"
        return result
    try:
        st = path.lstat()
    except OSError:
        return {"status": "invalid", "reason": "stat_failed", "records": 0}
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        return {"status": "invalid", "reason": "not_regular_file", "records": 0}
    if st.st_size <= 0:
        return {"status": "invalid", "reason": "empty", "records": 0}

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {"status": "invalid", "reason": "read_failed", "records": 0}
    lines = text.splitlines(keepends=True)
    valid = []
    item_timings = []
    truncated_final = False
    previous_processed = -1
    previous_elapsed = -1.0
    for index, line in enumerate(lines):
        line_number = index + 1
        if not line.strip():
            return {"status": "invalid", "reason": "blank_line",
                    "records": len(valid), "line": line_number}
        try:
            row = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError, TypeError):
            if index == len(lines) - 1 and not line.endswith("\n"):
                truncated_final = True
                break
            return {"status": "invalid", "reason": "invalid_json",
                    "records": len(valid), "line": line_number}
        if not isinstance(row, dict) or row.get("schema") != "descriptor-progress/1":
            return {"status": "invalid", "reason": "invalid_progress_record",
                    "records": len(valid), "line": line_number}
        processed = row.get("processed")
        total = row.get("total")
        elapsed = row.get("elapsed_seconds")
        last_id = row.get("last_id")
        item_seconds = row.get("item_seconds")
        phase = row.get("phase")
        if type(processed) is not int or processed < 0:
            return {"status": "invalid", "reason": "invalid_processed",
                    "records": len(valid), "line": line_number}
        if total is not None and (type(total) is not int or total < processed or total < 0):
            return {"status": "invalid", "reason": "invalid_total",
                    "records": len(valid), "line": line_number}
        if not _finite_nonnegative_number(elapsed):
            return {"status": "invalid", "reason": "invalid_elapsed",
                    "records": len(valid), "line": line_number}
        if last_id is not None and not isinstance(last_id, str):
            return {"status": "invalid", "reason": "invalid_last_id",
                    "records": len(valid), "line": line_number}
        if item_seconds is not None and not _finite_nonnegative_number(item_seconds):
            return {"status": "invalid", "reason": "invalid_item_seconds",
                    "records": len(valid), "line": line_number}
        if phase is not None and not isinstance(phase, str):
            return {"status": "invalid", "reason": "invalid_phase",
                    "records": len(valid), "line": line_number}
        if processed < previous_processed or elapsed < previous_elapsed:
            return {"status": "invalid", "reason": "nonmonotonic_progress",
                    "records": len(valid), "line": line_number}
        previous_processed = processed
        previous_elapsed = float(elapsed)
        normalized = {
            "processed": processed,
            "total": total,
            "elapsed_seconds": float(elapsed),
            "last_id": last_id,
            "item_seconds": None if item_seconds is None else float(item_seconds),
            "phase": phase,
        }
        valid.append(normalized)
        if item_seconds is not None:
            item_timings.append((float(item_seconds), last_id))

    if not valid:
        return {"status": "partial" if truncated_final else "invalid",
                "reason": "truncated_final_line" if truncated_final else "no_valid_records",
                "records": 0}
    last = valid[-1]
    result = {
        "status": "partial" if truncated_final else "valid",
        "records": len(valid),
        "processed": last["processed"],
        "total": last["total"],
        "reported_elapsed_seconds": last["elapsed_seconds"],
        "last_id": last["last_id"],
        "last_item_seconds": last["item_seconds"],
        "phase": last["phase"],
        "truncated_final_line": truncated_final,
    }
    if item_timings:
        seconds = [x[0] for x in item_timings]
        slowest_seconds, slowest_id = max(item_timings, key=lambda x: x[0])
        result["timed_items"] = len(seconds)
        result["median_item_seconds"] = float(statistics.median(seconds))
        result["slowest_item_seconds"] = slowest_seconds
        result["slowest_id"] = slowest_id
    wall = execution.get("wall_seconds")
    if _finite_nonnegative_number(wall):
        gap = max(0.0, float(wall) - last["elapsed_seconds"])
        result["seconds_since_last_progress"] = gap
        if execution.get("timed_out"):
            median = result.get("median_item_seconds")
            budget = execution.get("timeout_budget_seconds")
            if median is not None:
                floor = 60.0
                if _finite_nonnegative_number(budget) and budget > 0:
                    floor = min(floor, max(1.0, 0.10 * float(budget)))
                threshold = max(floor, 5.0 * median)
            else:
                threshold = 120.0
                if _finite_nonnegative_number(budget) and budget > 0:
                    threshold = min(threshold, max(2.0, 0.25 * float(budget)))
            result["long_gap_threshold_seconds"] = threshold
            result["timeout_pattern"] = (
                "timeout_long_gap_since_progress"
                if gap >= threshold
                else "timeout_recent_progress"
            )
    elif execution.get("timed_out"):
        result["timeout_pattern"] = "timeout_progress_present_wall_unknown"
    return result

def prepare(repo, task_path, request_path, state_path):
    repo = safe_repo(repo)
    if api(f"/repos/{repo}").get("private") is not True:
        raise PrivateIOError("destination_is_not_private")
    task_file = content(repo, task_path, "main")
    task = json.loads(decode_file(task_file, 65536))
    public_request = json.loads(Path(request_path).read_text())
    if task.get("trigger_id") != public_request.get("id"):
        raise PrivateIOError("task_trigger_mismatch")
    if task.get("version") != 1 or not re.fullmatch(r"[A-Za-z0-9_.-]+", task.get("request_id", "")):
        raise PrivateIOError("invalid_task")
    if not re.fullmatch(r"[0-9a-f]{40}", task.get("source_ref", "")):
        raise PrivateIOError("source_must_be_immutable")
    files = task.get("files")
    if not isinstance(files, dict) or not 1 <= len(files) <= 16 or "run_task.py" not in files:
        raise PrivateIOError("invalid_task_files")
    for name, source in files.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise PrivateIOError("invalid_local_filename")
        safe_path(source)
    prefix = safe_path(task["output_prefix"])
    if not prefix.startswith("transport/"):
        raise PrivateIOError("output_outside_permitted_area")
    run, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    if not run.isdigit() or not attempt.isdigit():
        raise PrivateIOError("invalid_run_identity")
    state = {"repository": repo, "task": task, "task_blob": task_file["sha"],
             "public_commit": os.environ["GITHUB_SHA"],
             "run_id": run, "run_attempt": attempt,
             "destination": f"{prefix}/{run}-{attempt}"}
    write_json(state_path, state)
    try:
        complete = json.loads(decode_file(content(repo, prefix + "/completed.json", "main"), 65536))
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    else:
        if complete.get("task_blob") != task_file["sha"]:
            raise PrivateIOError("completed_request_identity_changed")
        append_github_output("skip", "true")
        print("PRIVATE_IO_STAGE=already-completed")
        return
    marker = {"stage": "authorized", "source_ref": task["source_ref"],
              "public_commit": state["public_commit"], "run_id": run, "run_attempt": attempt}
    put_files(repo, {state["destination"] + "/start.json": json.dumps(marker, sort_keys=True).encode() + b"\n"})
    append_github_output("skip", "false")
    print("PRIVATE_IO_STAGE=read-write-authorized")

def fetch(state_path, task_dir):
    state = json.loads(Path(state_path).read_text())
    target = Path(task_dir)
    target.mkdir(mode=0o700, exist_ok=False)
    identities = {}
    for name, path in state["task"]["files"].items():
        item = content(state["repository"], path, state["task"]["source_ref"])
        raw = decode_file(item)
        file = target / name
        file.write_bytes(raw)
        file.chmod(0o400)
        identities[name] = {"path": path, "blob": item["sha"], "bytes": len(raw)}
    state["source_files"] = identities
    write_json(state_path, state)
    print("PRIVATE_IO_STAGE=input-ready")

def execute(task_dir, output_dir, image, timeout):
    from restricted_exec import run_bounded
    out = Path(output_dir)
    out.mkdir(mode=0o700, exist_ok=False)
    name = "private-candidate-" + os.environ["GITHUB_RUN_ID"] + "-" + os.environ["GITHUB_RUN_ATTEMPT"]
    cmd = ["docker", "create", "--name", name, "--network", "none", "--read-only",
           "--user", f"{os.getuid()}:{os.getgid()}",
           "--security-opt", "no-new-privileges:true", "--cap-drop", "ALL",
           "--pids-limit", "128", "--memory", "4g", "--cpus", "4",
           "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=256m,mode=1777",
           "-e", "MPLCONFIGDIR=/tmp/matplotlib", "-e", "XDG_CACHE_HOME=/tmp/cache",
           "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
           "-e", "DESCRIPTOR_WORKERS=4",
           "-e", "DESCRIPTOR_PROGRESS_PATH=/output/progress.jsonl",
           "-e", f"DESCRIPTOR_TIMEOUT_SECONDS={timeout}",
           "--mount", f"type=bind,src={Path(task_dir).resolve()},dst=/task,readonly",
           "--mount", f"type=bind,src={out.resolve()},dst=/output",
           "--workdir", "/task", image, "python3", "-B", "/task/run_task.py", "--output", "/output"]
    result = None
    failure = None
    creation_stderr = b''
    cleanup_failed = False
    candidate_started = False
    candidate_started_at = None
    candidate_wall_seconds = None
    try:
        created = subprocess.run(cmd, capture_output=True, check=False, timeout=30)
        if created.returncode:
            creation_stderr = created.stderr[:1048576]
            raise PrivateIOError("container_create_failed")
        print("EXECUTION_STAGE=candidate-start", flush=True)
        candidate_started = True
        candidate_started_at = time.monotonic()
        result = run_bounded(["docker", "start", "-a", name],
                             timeout_seconds=timeout, max_output_bytes=1048576)
        candidate_wall_seconds = time.monotonic() - candidate_started_at
        print("EXECUTION_STAGE=candidate-returned", flush=True)
    except Exception as exc:
        failure = exc
        if candidate_started_at is not None and candidate_wall_seconds is None:
            candidate_wall_seconds = time.monotonic() - candidate_started_at
    finally:
        # Preserve exit facts before asking the daemon to clean up. A hung
        # control call must not block private publication of durable prefixes.
        status = {"exit_code": result.child_exit_code if result else 125,
                  "timed_out": result.timed_out if result else isinstance(failure, subprocess.TimeoutExpired),
                  "output_limit_exceeded": result.output_limit_exceeded if result else False,
                  "host_failure_class": type(failure).__name__ if failure else None,
                  "candidate_started": candidate_started,
                  "wall_seconds": candidate_wall_seconds,
                  "timeout_budget_seconds": timeout}
        (out / "execution.stdout.log").write_bytes(result.stdout if result else b'')
        (out / "execution.stderr.log").write_bytes(result.stderr if result else creation_stderr)
        write_json(out / "execution.json", status)
        try:
            cleanup = subprocess.run(["docker", "rm", "-f", name],
                                     capture_output=True, check=False, timeout=15)
            cleanup_failed = cleanup.returncode != 0
            status['cleanup_timed_out'] = False
        except subprocess.TimeoutExpired:
            cleanup_failed = True
            status['cleanup_timed_out'] = True
        except Exception as exc:
            cleanup_failed = True
            status['cleanup_failure_class'] = type(exc).__name__
        status['cleanup_failed'] = cleanup_failed
        if cleanup_failed:
            status['candidate_exit_code'] = status['exit_code']
            status['exit_code'] = 125
        write_json(out / "execution.json", status)
        print("EXECUTION_STAGE=cleanup-failed" if cleanup_failed else "EXECUTION_STAGE=cleanup-complete", flush=True)
    if failure is not None or cleanup_failed or result is None or result.child_exit_code != 0 or result.timed_out or result.output_limit_exceeded:
        raise PrivateIOError("candidate_failed")
    print("COMPUTE_STAGE=complete")

def publish(state_path, output_dir):
    state = json.loads(Path(state_path).read_text())
    out = Path(output_dir)
    allowed = {"result.jsonl", "summary.json", "focused-and-batch.log", "progress.jsonl",
               "execution.stdout.log", "execution.stderr.log", "execution.json"}
    files, descriptions = {}, {}
    if out.exists():
        for path in out.iterdir():
            st = path.lstat()
            if path.name not in allowed or not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_size > 32*1024*1024:
                raise PrivateIOError("unexpected_output_file")
            raw = path.read_bytes()
            files[state["destination"] + "/" + path.name] = raw
            descriptions[path.name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    summary = None
    execution = None
    if "summary.json" in descriptions:
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if "execution.json" in descriptions:
        execution = json.loads((out / "execution.json").read_text(encoding="utf-8"))
    result_validation = inspect_result_jsonl(
        out / "result.jsonl",
        None if not isinstance(summary, dict) else summary.get("rows"),
    )
    progress_validation = inspect_progress_jsonl(out / "progress.jsonl", execution)
    receipt = {"schema": "private-computation-receipt/1", "state": state,
               "files": descriptions,
               "materialized_file_present": "result.jsonl" in descriptions,
               "result_validation": result_validation,
               "progress": progress_validation,
               "note": "Inspect execution/result/progress; 30 minutes is an interactive diagnostic budget, not evidence that heavier compute is impossible."}
    files[state["destination"] + "/receipt.json"] = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"

    success_claim = (
        isinstance(summary, dict) and summary.get("stage") == "materialized"
        and isinstance(execution, dict) and execution.get("exit_code") == 0
        and not execution.get("timed_out")
        and not execution.get("output_limit_exceeded")
    )
    if success_claim and result_validation.get("status") == "valid":
        complete = {"task_blob": state["task_blob"], "source_ref": state["task"]["source_ref"],
                    "result_path": state["destination"] + "/result.jsonl",
                    "receipt_path": state["destination"] + "/receipt.json"}
        files[state["task"]["output_prefix"] + "/completed.json"] = json.dumps(complete, sort_keys=True).encode() + b"\n"

    recovery_branch = _recovery_branch_name(state)
    append_github_output("recovery_branch", recovery_branch)
    put_files(state["repository"], files, recovery_branch=recovery_branch)
    if success_claim and result_validation.get("status") != "valid":
        print("PRIVATE_IO_STAGE=invalid-materialized-result", file=sys.stderr)
        raise PrivateIOError("materialized_result_invalid")
    print("PRIVATE_IO_STAGE=results-saved")

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--repo", required=True); q.add_argument("--task-path", required=True); q.add_argument("--request", required=True); q.add_argument("--state", required=True)
    q = sub.add_parser("fetch"); q.add_argument("--state", required=True); q.add_argument("--task-dir", required=True)
    q = sub.add_parser("execute"); q.add_argument("--task-dir", required=True); q.add_argument("--output", required=True); q.add_argument("--image", required=True); q.add_argument("--timeout", type=int, default=1800)
    q = sub.add_parser("publish"); q.add_argument("--state", required=True); q.add_argument("--output", required=True)
    a = p.parse_args()
    if a.mode == "prepare": prepare(a.repo, a.task_path, a.request, a.state)
    elif a.mode == "fetch": fetch(a.state, a.task_dir)
    elif a.mode == "execute": execute(a.task_dir, a.output, a.image, a.timeout)
    else: publish(a.state, a.output)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("PRIVATE_IO_STAGE=failed CLASS=" + type(exc).__name__ +
              (" HTTP=" + str(exc.code) if isinstance(exc, urllib.error.HTTPError) else ""), file=sys.stderr)
        raise SystemExit(2)
