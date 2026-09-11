"""Scoped private-file I/O around public Actions execution. No scientific definitions."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
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

def put_files(repo, files):
    # Read/check/build against one immutable parent; never force or overwrite.
    for attempt in range(5):
        try:
            parent = api(f"/repos/{repo}/git/ref/heads/main")["object"]["sha"]
            entries = []
            for path, raw in files.items():
                safe_path(path)
                try:
                    old = content(repo, path, parent)
                except urllib.error.HTTPError as exc:
                    if exc.code != 404:
                        raise
                else:
                    if old["sha"] != blob_sha(raw):
                        raise PrivateIOError("refuse_different_existing_file")
                    continue
                entries.append({"path": path, "mode": "100644", "type": "blob",
                                "sha": api(f"/repos/{repo}/git/blobs", "POST",
                                           {"content": base64.b64encode(raw).decode(), "encoding": "base64"})["sha"]})
            if not entries:
                return
            tree = api(f"/repos/{repo}/git/commits/{parent}")["tree"]["sha"]
            new_tree = api(f"/repos/{repo}/git/trees", "POST", {"base_tree": tree, "tree": entries})["sha"]
            commit = api(f"/repos/{repo}/git/commits", "POST",
                         {"message": "Record scoped computation output",
                          "tree": new_tree, "parents": [parent]})["sha"]
            api(f"/repos/{repo}/git/refs/heads/main", "PATCH", {"sha": commit, "force": False})
            return
        except urllib.error.HTTPError as exc:
            if exc.code not in {409, 422, 429, 500, 502, 503, 504} or attempt == 4:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise
        time.sleep(min(2 ** attempt, 16))
    raise PrivateIOError("publication_failed")

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
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("skip=true\n")
        print("PRIVATE_IO_STAGE=already-completed")
        return
    marker = {"stage": "authorized", "source_ref": task["source_ref"],
              "public_commit": state["public_commit"], "run_id": run, "run_attempt": attempt}
    put_files(repo, {state["destination"] + "/start.json": json.dumps(marker, sort_keys=True).encode() + b"\n"})
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write("skip=false\n")
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
           "--pids-limit", "128", "--memory", "4g", "--cpus", "2",
           "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=256m,mode=1777",
           "-e", "MPLCONFIGDIR=/tmp/matplotlib", "-e", "XDG_CACHE_HOME=/tmp/cache",
           "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
           "--mount", f"type=bind,src={Path(task_dir).resolve()},dst=/task,readonly",
           "--mount", f"type=bind,src={out.resolve()},dst=/output",
           "--workdir", "/task", image, "python3", "-B", "/task/run_task.py", "--output", "/output"]
    result = None
    try:
        created = subprocess.run(cmd, capture_output=True, check=False)
        if created.returncode:
            (out / "execution.stderr.log").write_bytes(created.stderr)
            raise PrivateIOError("container_create_failed")
        result = run_bounded(["docker", "start", "-a", name],
                             timeout_seconds=timeout, max_output_bytes=1048576)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
    (out / "execution.stdout.log").write_bytes(result.stdout)
    (out / "execution.stderr.log").write_bytes(result.stderr)
    status = {"exit_code": result.child_exit_code, "timed_out": result.timed_out,
              "output_limit_exceeded": result.output_limit_exceeded}
    write_json(out / "execution.json", status)
    if result.child_exit_code != 0 or result.timed_out or result.output_limit_exceeded:
        raise PrivateIOError("candidate_failed")
    print("COMPUTE_STAGE=complete")

def publish(state_path, output_dir):
    state = json.loads(Path(state_path).read_text())
    out = Path(output_dir)
    allowed = {"result.jsonl", "summary.json", "focused-and-batch.log",
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
    receipt = {"schema": "private-computation-receipt/1", "state": state,
               "files": descriptions, "materialized_file_present": "result.jsonl" in descriptions,
               "note": "Inspect summary and execution status; receipt alone does not establish scientific success."}
    files[state["destination"] + "/receipt.json"] = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    if "summary.json" in descriptions and "execution.json" in descriptions:
        summary = json.loads((out / "summary.json").read_text())
        execution = json.loads((out / "execution.json").read_text())
        if summary.get("stage") == "materialized" and execution.get("exit_code") == 0 and not execution.get("timed_out") and not execution.get("output_limit_exceeded"):
            complete = {"task_blob": state["task_blob"], "source_ref": state["task"]["source_ref"],
                        "result_path": state["destination"] + "/result.jsonl",
                        "receipt_path": state["destination"] + "/receipt.json"}
            files[state["task"]["output_prefix"] + "/completed.json"] = json.dumps(complete, sort_keys=True).encode() + b"\n"
    put_files(state["repository"], files)
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
