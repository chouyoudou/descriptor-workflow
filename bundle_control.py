"""Deterministic control bridge: one private bundle -> immutable source -> isolated execution.

No scientific definitions live here. The public trigger carries only an opaque id.
PRIVATE_REPO_TOKEN is used only by trusted Actions steps; candidate execution is delegated
to the existing private_io.py container boundary and receives no token/network.
"""
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
import time
import urllib.error
import urllib.parse
import urllib.request

BUNDLE_RE = re.compile(r"^bt-[A-Za-z0-9_.-]{1,64}$")
FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
REF_RE = re.compile(r"^[0-9a-f]{40}$")
BLOB_RE = re.compile(r"^[0-9a-f]{40}$")
SCHEMA = "private-task-bundle/2"
TRIGGER_SCHEMA = "private-bundle-trigger/2"
SOURCE_SCHEMA = "bundle-materialized-source/2"
ALLOWED_INPUT_PREFIXES = (
    "inputs/",
    "elements/",
    "transport/parallel-",
    "transport/reference/",
)
MAX_SOURCE_FILES = 12
MAX_INPUT_FILES = 16
MAX_SOURCE_FILE = 128 * 1024
MAX_SOURCE_TOTAL = 512 * 1024
MAX_INPUT_FILE = 2 * 1024 * 1024
MAX_INPUT_TOTAL = 8 * 1024 * 1024
MAX_OUTPUT_FILE = 32 * 1024 * 1024

class BundleError(RuntimeError):
    pass

def safe_repo(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value or ""):
        raise BundleError("invalid_repository")
    return value

def safe_path(value):
    if (not isinstance(value, str) or not value or value.startswith("/")
            or "\\" in value or ".." in value.split("/")):
        raise BundleError("invalid_path")
    return value

def _retry_delay(exc, attempt):
    h = getattr(exc, "headers", None)
    if h is not None:
        ra = h.get("Retry-After")
        if ra and str(ra).isdigit():
            return min(max(int(ra), 1), 120)
        reset = h.get("X-RateLimit-Reset")
        remaining = h.get("X-RateLimit-Remaining")
        if reset and remaining == "0":
            try:
                return min(max(int(reset) - int(time.time()) + 1, 1), 120)
            except ValueError:
                pass
    return min(2 ** attempt, 16) + random.uniform(0.0, 0.4)

def api(path, method="GET", body=None):
    token = os.environ.get("PRIVATE_REPO_TOKEN", "").strip()
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "descriptor-bundle-control",
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(
        "https://api.github.com" + path,
        data=None if body is None else json.dumps(body, separators=(",", ":")).encode(),
        method=method,
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read(64 * 1024 * 1024 + 1)
        if len(raw) > 64 * 1024 * 1024:
            raise BundleError("response_too_large")
        return None if not raw else json.loads(raw)

def content(repo, path, ref):
    q = urllib.parse.quote(safe_path(path), safe="/")
    rr = urllib.parse.quote(ref, safe="")
    return api(f"/repos/{safe_repo(repo)}/contents/{q}?ref={rr}")

def decode_contents(item, limit):
    if item.get("type") != "file" or item.get("encoding") != "base64":
        raise BundleError("expected_small_base64_file")
    raw = base64.b64decode(item["content"])
    if len(raw) > limit:
        raise BundleError("file_too_large")
    if hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != item.get("sha"):
        raise BundleError("blob_identity_mismatch")
    return raw

def append_output(key, value):
    p = os.environ.get("GITHUB_OUTPUT")
    if p:
        with open(p, "a", encoding="utf-8") as handle:
            handle.write(f"{key}={value}\n")

def resolve(event_path):
    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    actor = os.environ.get("GITHUB_ACTOR", "")
    owner = os.environ.get("GITHUB_REPOSITORY_OWNER", "")
    title = str(event.get("issue", {}).get("title", ""))
    body = str(event.get("issue", {}).get("body") or "")
    m = re.fullmatch(r"bundle-task:(bt-[A-Za-z0-9_.-]{1,64})", title)
    bundle_id = m.group(1) if m else ""
    expected = f"schema={TRIGGER_SCHEMA}\nid={bundle_id}\n" if bundle_id else ""
    ok = bool(m and actor == owner and body == expected)
    append_output("ok", "true" if ok else "false")
    append_output("bundle_id", bundle_id)
    print("RESOLVE_STAGE=" + ("authorized" if ok else "ignored"))

def validate_bundle(raw, bundle_id):
    try:
        b = json.loads(raw)
    except Exception as exc:
        raise BundleError("invalid_bundle_json") from exc
    if b.get("schema") != SCHEMA or b.get("bundle_id") != bundle_id:
        raise BundleError("bundle_identity_mismatch")

    files = b.get("files")
    hashes = b.get("sha256")
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_SOURCE_FILES:
        raise BundleError("invalid_source_files")
    if "run_task.py" not in files or not isinstance(hashes, dict) or set(hashes) != set(files):
        raise BundleError("invalid_source_manifest")
    total = 0
    sources = {}
    for name, text in files.items():
        if not FILE_RE.fullmatch(name) or not isinstance(text, str):
            raise BundleError("invalid_source_entry")
        data = text.encode("utf-8")
        total += len(data)
        if len(data) > MAX_SOURCE_FILE or total > MAX_SOURCE_TOTAL:
            raise BundleError("source_too_large")
        if hashlib.sha256(data).hexdigest() != hashes.get(name):
            raise BundleError("source_hash_mismatch:" + name)
        sources[name] = data

    inputs = b.get("inputs", {})
    if not isinstance(inputs, dict) or len(inputs) > MAX_INPUT_FILES:
        raise BundleError("invalid_inputs")
    normalized_inputs = {}
    for local, spec in inputs.items():
        if not FILE_RE.fullmatch(local) or local in sources or not isinstance(spec, dict):
            raise BundleError("invalid_input_entry")
        path = safe_path(spec.get("path"))
        if not any(path.startswith(prefix) for prefix in ALLOWED_INPUT_PREFIXES):
            raise BundleError("input_path_not_allowed:" + path)
        ref = spec.get("ref", "")
        blob = spec.get("blob", "")
        if not REF_RE.fullmatch(ref) or not BLOB_RE.fullmatch(blob):
            raise BundleError("input_identity_invalid:" + local)
        normalized_inputs[local] = {"path": path, "ref": ref, "blob": blob}

    timeout_seconds = b.get("timeout_seconds", 1800)
    if type(timeout_seconds) is not int or not 30 <= timeout_seconds <= 1800:
        raise BundleError("invalid_timeout")
    return b, sources, normalized_inputs, timeout_seconds

def put_create_only(repo, files, message):
    repo = safe_repo(repo)
    blob_shas = {}
    for path, raw in files.items():
        safe_path(path)
        if not isinstance(raw, (bytes, bytearray)):
            raise BundleError("invalid_file_bytes")
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
                    expected = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
                    if old.get("sha") == expected:
                        continue
                    raise BundleError("refuse_different_existing_file:" + path)
                if path not in blob_shas:
                    blob_shas[path] = api(
                        f"/repos/{repo}/git/blobs", "POST",
                        {"content": base64.b64encode(raw).decode(), "encoding": "base64"},
                    )["sha"]
                entries.append({
                    "path": path, "mode": "100644", "type": "blob",
                    "sha": blob_shas[path],
                })
            if not entries:
                return parent
            tree = api(f"/repos/{repo}/git/commits/{parent}")["tree"]["sha"]
            new_tree = api(
                f"/repos/{repo}/git/trees", "POST",
                {"base_tree": tree, "tree": entries},
            )["sha"]
            commit = api(
                f"/repos/{repo}/git/commits", "POST",
                {"message": message, "tree": new_tree, "parents": [parent]},
            )["sha"]
            api(
                f"/repos/{repo}/git/refs/heads/main", "PATCH",
                {"sha": commit, "force": False},
            )
            return commit
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {409, 429, 500, 502, 503, 504}
            limited_403 = exc.code == 403 and (
                (getattr(exc, "headers", None) or {}).get("Retry-After")
                or (getattr(exc, "headers", None) or {}).get("X-RateLimit-Remaining") == "0"
            )
            if (not retryable and not limited_403) or attempt == 6:
                raise
            time.sleep(_retry_delay(exc, attempt))
        except (urllib.error.URLError, TimeoutError):
            if attempt == 6:
                raise
            time.sleep(min(2 ** attempt, 16) + random.uniform(0.0, 0.4))
    raise BundleError("publication_failed")

def materialize(repo, bundle_id):
    if not BUNDLE_RE.fullmatch(bundle_id):
        raise BundleError("invalid_bundle_id")
    bundle_path = f"transport/bundle_inbox/{bundle_id}.json"
    item = content(repo, bundle_path, "main")
    raw = decode_contents(item, 1024 * 1024)
    bundle, sources, inputs, timeout_seconds = validate_bundle(raw, bundle_id)

    prefix = f"transport/tasks/bundle-materialized/{bundle_id}"
    source_manifest = {
        "schema": SOURCE_SCHEMA,
        "bundle_id": bundle_id,
        "bundle_path": bundle_path,
        "bundle_blob": item["sha"],
        "files": {k: hashlib.sha256(v).hexdigest() for k, v in sorted(sources.items())},
        "inputs": inputs,
        "timeout_seconds": timeout_seconds,
    }
    formal = {f"{prefix}/{name}": data for name, data in sources.items()}
    formal[f"{prefix}/BUNDLE_SOURCE.json"] = (
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    source_ref = put_create_only(repo, formal, "Materialize validated private task bundle")

    for name, data in sources.items():
        got = decode_contents(content(repo, f"{prefix}/{name}", source_ref), MAX_SOURCE_FILE)
        if got != data:
            raise BundleError("immutable_source_readback_mismatch:" + name)

    append_output("source_ref", source_ref)
    append_output("bundle_blob", item["sha"])
    append_output("timeout_seconds", str(timeout_seconds))
    print("MATERIALIZE_STAGE=complete")

def fetch_task(repo, bundle_id, source_ref, dest):
    if not REF_RE.fullmatch(source_ref):
        raise BundleError("invalid_source_ref")
    prefix = f"transport/tasks/bundle-materialized/{bundle_id}"
    manifest_raw = decode_contents(
        content(repo, f"{prefix}/BUNDLE_SOURCE.json", source_ref), 1024 * 1024
    )
    manifest = json.loads(manifest_raw)
    if manifest.get("schema") != SOURCE_SCHEMA or manifest.get("bundle_id") != bundle_id:
        raise BundleError("source_manifest_identity")
    target = Path(dest)
    target.mkdir(mode=0o700, parents=True, exist_ok=False)

    total_inputs = 0
    for name, digest in manifest["files"].items():
        raw = decode_contents(content(repo, f"{prefix}/{name}", source_ref), MAX_SOURCE_FILE)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise BundleError("source_fetch_hash_mismatch:" + name)
        p = target / name
        p.write_bytes(raw)
        p.chmod(0o400)

    for local, spec in manifest.get("inputs", {}).items():
        item = content(repo, spec["path"], spec["ref"])
        if item.get("sha") != spec["blob"]:
            raise BundleError("input_blob_mismatch:" + local)
        raw = decode_contents(item, MAX_INPUT_FILE)
        total_inputs += len(raw)
        if total_inputs > MAX_INPUT_TOTAL:
            raise BundleError("input_total_too_large")
        p = target / local
        p.write_bytes(raw)
        p.chmod(0o400)
    print("FETCH_TASK_STAGE=complete")

def _reject_constant(value):
    raise ValueError("nonfinite_json_constant")

def validate_jsonl(path, expected_rows):
    p = Path(path)
    st = p.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_size <= 0:
        raise BundleError("invalid_result_file")
    count = 0
    with p.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                raise BundleError(f"blank_result_line:{lineno}")
            row = json.loads(line, parse_constant=_reject_constant)
            if not isinstance(row, dict):
                raise BundleError(f"nonobject_result_row:{lineno}")
            count += 1
    if count != expected_rows:
        raise BundleError("result_row_count_mismatch")
    return count

def publish(repo, bundle_id, source_ref, bundle_blob, output):
    out = Path(output)
    allowed = {
        "result.jsonl", "summary.json", "focused-and-batch.log",
        "execution.stdout.log", "execution.stderr.log", "execution.json",
    }
    collected = {}
    for p in out.iterdir():
        st = p.lstat()
        if p.name not in allowed or not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise BundleError("unexpected_output_file:" + p.name)
        if st.st_size > MAX_OUTPUT_FILE:
            raise BundleError("output_too_large:" + p.name)
        collected[p.name] = p.read_bytes()

    for required in ("result.jsonl", "summary.json", "focused-and-batch.log", "execution.json"):
        if required not in collected:
            raise BundleError("missing_output:" + required)
    summary = json.loads(collected["summary.json"], parse_constant=_reject_constant)
    execution = json.loads(collected["execution.json"], parse_constant=_reject_constant)
    success = (
        summary.get("stage") == "materialized"
        and execution.get("exit_code") == 0
        and not execution.get("timed_out")
        and not execution.get("output_limit_exceeded")
    )
    if not success:
        raise BundleError("execution_or_summary_not_successful")
    expected_rows = summary.get("rows")
    if type(expected_rows) is not int or expected_rows < 1:
        raise BundleError("invalid_summary_rows")
    actual_rows = validate_jsonl(out / "result.jsonl", expected_rows)

    run = os.environ.get("GITHUB_RUN_ID", "0")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    prefix = f"transport/bundle-executions/{bundle_id}/{run}-{attempt}"
    receipt = {
        "schema": "bundle-computation-receipt/2",
        "bundle_id": bundle_id,
        "bundle_blob": bundle_blob,
        "source_ref": source_ref,
        "run_id": run,
        "run_attempt": attempt,
        "files": {n: hashlib.sha256(b).hexdigest() for n, b in sorted(collected.items())},
        "result_rows": actual_rows,
        "status": "materialized",
    }
    files = {f"{prefix}/{name}": raw for name, raw in collected.items()}
    files[f"{prefix}/receipt.json"] = (
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    ).encode()
    files[f"transport/bundle-executions/{bundle_id}/completed.json"] = (
        json.dumps(receipt, sort_keys=True) + "\n"
    ).encode()
    commit = put_create_only(repo, files, "Record bundle computation result")
    append_output("result_commit", commit)
    print("PUBLISH_STAGE=complete")

def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("resolve")
    q.add_argument("--event", required=True)
    q = sp.add_parser("materialize")
    q.add_argument("--repo", required=True)
    q.add_argument("--bundle-id", required=True)
    q = sp.add_parser("fetch-task")
    q.add_argument("--repo", required=True)
    q.add_argument("--bundle-id", required=True)
    q.add_argument("--source-ref", required=True)
    q.add_argument("--dest", required=True)
    q = sp.add_parser("publish")
    q.add_argument("--repo", required=True)
    q.add_argument("--bundle-id", required=True)
    q.add_argument("--source-ref", required=True)
    q.add_argument("--bundle-blob", required=True)
    q.add_argument("--output", required=True)
    a = p.parse_args()
    if a.cmd == "resolve":
        resolve(a.event)
    elif a.cmd == "materialize":
        materialize(a.repo, a.bundle_id)
    elif a.cmd == "fetch-task":
        fetch_task(a.repo, a.bundle_id, a.source_ref, a.dest)
    else:
        publish(a.repo, a.bundle_id, a.source_ref, a.bundle_blob, a.output)

if __name__ == "__main__":
    main()
