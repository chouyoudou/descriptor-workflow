"""Deterministic control bridge: one private bundle -> immutable source -> isolated execution.

No scientific definitions live here. The public trigger carries only an opaque id.
PRIVATE_REPO_TOKEN is used only by trusted Actions steps; candidate execution is delegated
to the existing private_io.py container boundary and receives no token/network.
"""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import re
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BUNDLE_RE = re.compile(r"^bt-[A-Za-z0-9_.-]{1,64}$")
FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
REF_RE = re.compile(r"^[0-9a-f]{40}$")
BLOB_RE = re.compile(r"^[0-9a-f]{40}$")
SCHEMA = "private-task-bundle/2"
TEXT_SCHEMA = "private-task-bundle/3"
SUCCESSOR_SCHEMA = "private-task-bundle/4"
TRIGGER_SCHEMA = "private-bundle-trigger/2"
SOURCE_SCHEMA = "bundle-materialized-source/2"
SUCCESSOR_SOURCE_SCHEMA = "bundle-materialized-source/3"
ALLOWED_INPUT_PREFIXES = (
    "inputs/",
    "elements/",
    "transport/parallel-",
    "transport/reference/",
    "transport/executions/",
    "transport/bundle-executions/",
    "transport/recovered/",
    "transport/tasks/",
)

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
            return max(int(ra), 1)
        reset = h.get("X-RateLimit-Reset")
        remaining = h.get("X-RateLimit-Remaining")
        if reset and remaining == "0":
            try:
                return max(int(reset) - int(time.time()) + 1, 1)
            except ValueError:
                pass
    if getattr(exc, "code", None) in (403, 429):
        return 60 * 2 ** attempt
    return min(2 ** attempt, 16) + random.uniform(0.0, 0.4)

def api(path, method="GET", body=None, max_response_bytes=64 * 1024 * 1024):
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
    from private_io import NoRedirect
    with urllib.request.build_opener(NoRedirect).open(req, timeout=30) as response:
        if max_response_bytes is None:
            raw = response.read()
        else:
            raw = response.read(max_response_bytes + 1)
            if len(raw) > max_response_bytes:
                raise BundleError("response_too_large")
        return None if not raw else json.loads(raw)

def content(repo, path, ref):
    q = urllib.parse.quote(safe_path(path), safe="/")
    rr = urllib.parse.quote(ref, safe="")
    return api(f"/repos/{safe_repo(repo)}/contents/{q}?ref={rr}")

def decode_contents(item, limit=None):
    if item.get("type") != "file" or item.get("encoding") != "base64":
        raise BundleError("expected_base64_file")
    raw = base64.b64decode(item["content"])
    if limit is not None and len(raw) > limit:
        raise BundleError("file_too_large")
    if hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != item.get("sha"):
        raise BundleError("blob_identity_mismatch")
    return raw

def read_repository_file(repo, path, ref, limit=None, expected_blob=None):
    """Read a GitHub file by immutable identity, falling back to the Git blob body."""
    item = content(repo, path, ref)
    if item.get("type") != "file":
        raise BundleError("expected_file")
    blob = item.get("sha", "")
    if not BLOB_RE.fullmatch(blob):
        raise BundleError("invalid_blob_identity")
    if expected_blob is not None and blob != expected_blob:
        raise BundleError("pinned_file_identity_mismatch")
    size = item.get("size")
    if limit is not None and type(size) is int and size > limit:
        raise BundleError("file_too_large")
    if item.get("encoding") == "base64":
        return decode_contents(item, limit), blob
    blob_item = api(
        f"/repos/{safe_repo(repo)}/git/blobs/{blob}",
        max_response_bytes=None if limit is None else max(64 * 1024 * 1024, limit * 2),
    )
    if blob_item.get("sha") != blob:
        raise BundleError("blob_identity_mismatch")
    return decode_contents({**blob_item, "type": "file"}, limit), blob

def read_pinned_file(repo, path, ref, blob, limit=None):
    """Read one immutable pinned private input; local byte quotas are not imposed."""
    raw, _ = read_repository_file(repo, path, ref, limit=limit, expected_blob=blob)
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
    schema = b.get("schema") if isinstance(b, dict) else None
    if (not isinstance(b, dict)
            or schema not in (SCHEMA, TEXT_SCHEMA, SUCCESSOR_SCHEMA)
            or b.get("bundle_id") != bundle_id):
        raise BundleError("bundle_identity_mismatch")

    sources = {}
    if schema == SUCCESSOR_SCHEMA:
        if "files" in b or "sha256" in b:
            raise BundleError("successor_ambiguous_source_fields")
        base_source_ref = b.get("base_source_ref", "")
        if not REF_RE.fullmatch(base_source_ref):
            raise BundleError("invalid_base_source_ref")
        changed = b.get("changed_files", {})
        delete_files = b.get("delete_files", [])
        if not isinstance(changed, dict):
            raise BundleError("invalid_changed_files")
        if (not isinstance(delete_files, list)
                or len(set(delete_files)) != len(delete_files)):
            raise BundleError("invalid_delete_files")
        if not changed and not delete_files:
            raise BundleError("successor_no_requested_changes")
        for name, text in changed.items():
            if (not isinstance(name, str) or not FILE_RE.fullmatch(name)
                    or name == "BUNDLE_SOURCE.json" or not isinstance(text, str)):
                raise BundleError("invalid_changed_source_entry")
            sources[name] = text.encode("utf-8")
        for name in delete_files:
            if (not isinstance(name, str) or not FILE_RE.fullmatch(name)
                    or name == "BUNDLE_SOURCE.json"):
                raise BundleError("invalid_delete_source_entry")
        if set(changed) & set(delete_files):
            raise BundleError("successor_change_delete_overlap")
        if "run_task.py" in delete_files:
            raise BundleError("cannot_delete_run_task")
    else:
        files = b.get("files")
        hashes = b.get("sha256")
        if not isinstance(files, dict) or not files:
            raise BundleError("invalid_source_files")
        if "run_task.py" not in files:
            raise BundleError("invalid_source_manifest")
        # V2 retains its exact declared-hash contract. V3 can omit the redundant
        # client-computed manifest: the immutable Git bundle binds the received
        # text, and materialize/fetch_task derive and verify its exact UTF-8 bytes.
        # If a V3 client DOES declare hashes, contradictions still fail closed.
        if schema == SCHEMA or "sha256" in b:
            if not isinstance(hashes, dict) or set(hashes) != set(files):
                raise BundleError("invalid_source_manifest")
        for name, text in files.items():
            if (not FILE_RE.fullmatch(name) or name == "BUNDLE_SOURCE.json"
                    or not isinstance(text, str)):
                raise BundleError("invalid_source_entry")
            data = text.encode("utf-8")
            if hashes is not None and hashlib.sha256(data).hexdigest() != hashes.get(name):
                raise BundleError("source_hash_mismatch:" + name)
            sources[name] = data

    inputs = b.get("inputs", {})
    if not isinstance(inputs, dict):
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


def _commit_files(repo, commit_ref):
    """Read all files changed by one commit without imposing a local file-count cap."""
    files = []
    page = 1
    while True:
        commit = api(
            f"/repos/{safe_repo(repo)}/commits/{commit_ref}?per_page=100&page={page}"
        )
        if not isinstance(commit, dict) or commit.get("sha") != commit_ref:
            raise BundleError("invalid_base_source_commit")
        batch = commit.get("files")
        if not isinstance(batch, list):
            raise BundleError("base_source_commit_files_missing")
        files.extend(batch)
        if len(batch) < 100:
            return files
        page += 1

def _load_base_source(repo, base_source_ref):
    """Resolve one prior materialized source commit from its immutable ref alone."""
    if not REF_RE.fullmatch(base_source_ref):
        raise BundleError("invalid_base_source_ref")
    changed = _commit_files(repo, base_source_ref)
    manifest_re = re.compile(
        r"^transport/tasks/bundle-materialized/(bt-[A-Za-z0-9_.-]{1,64})/BUNDLE_SOURCE[.]json$"
    )
    candidates = []
    for item in changed:
        filename = item.get("filename", "") if isinstance(item, dict) else ""
        match = manifest_re.fullmatch(filename)
        if match:
            candidates.append((match.group(1), filename))
    if len(candidates) != 1:
        raise BundleError("base_source_manifest_not_unique")
    base_bundle_id, manifest_path = candidates[0]
    prefix = f"transport/tasks/bundle-materialized/{base_bundle_id}"
    try:
        manifest_raw, _ = read_repository_file(repo, manifest_path, base_source_ref)
        manifest = json.loads(manifest_raw)
    except Exception as exc:
        raise BundleError("invalid_base_source_manifest") from exc
    if (not isinstance(manifest, dict)
            or manifest.get("schema") not in (SOURCE_SCHEMA, SUCCESSOR_SOURCE_SCHEMA)
            or manifest.get("bundle_id") != base_bundle_id):
        raise BundleError("invalid_base_source_manifest")
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise BundleError("invalid_base_source_files")
    if "run_task.py" not in declared:
        raise BundleError("invalid_base_source_files")
    expected_commit_paths = {manifest_path}
    sources, blob_shas = {}, {}
    for name, digest in sorted(declared.items()):
        if (not isinstance(name, str) or not FILE_RE.fullmatch(name)
                or name == "BUNDLE_SOURCE.json"
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise BundleError("invalid_base_source_entry")
        path = f"{prefix}/{name}"
        expected_commit_paths.add(path)
        raw, blob = read_repository_file(repo, path, base_source_ref)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise BundleError("base_source_hash_mismatch:" + name)
        sources[name] = raw
        blob_shas[name] = blob
    actual_commit_paths = {
        item.get("filename") for item in changed if isinstance(item, dict)
    }
    if actual_commit_paths != expected_commit_paths:
        raise BundleError("base_source_commit_scope_mismatch")
    return {
        "bundle_id": base_bundle_id,
        "sources": sources,
        "blob_shas": blob_shas,
        "manifest": manifest,
    }


def _apply_successor_sources(base_sources, changed_sources, delete_files):
    final = dict(base_sources)
    deleted = []
    for name in delete_files:
        if name not in final:
            raise BundleError("successor_delete_missing_file:" + name)
        del final[name]
        deleted.append(name)

    actual_changed, unchanged_overlay = [], []
    for name, raw in changed_sources.items():
        if name in base_sources and base_sources[name] == raw:
            unchanged_overlay.append(name)
            continue
        final[name] = raw
        actual_changed.append(name)

    if "run_task.py" not in final:
        raise BundleError("invalid_final_source_manifest")
    for name, raw in final.items():
        if (not FILE_RE.fullmatch(name) or name == "BUNDLE_SOURCE.json"
                or not isinstance(raw, (bytes, bytearray))):
            raise BundleError("invalid_final_source_entry")
    if not actual_changed and not deleted:
        raise BundleError("successor_no_effect")
    inherited = sorted(name for name in final if name not in actual_changed)
    return final, {
        "requested_changed_files": sorted(changed_sources),
        "actual_changed_files": sorted(actual_changed),
        "unchanged_overlay_files": sorted(unchanged_overlay),
        "deleted_files": sorted(deleted),
        "inherited_files": inherited,
    }


def recovery_branch(bundle_id, phase):
    run = os.environ.get("GITHUB_RUN_ID", "0")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    if not run.isdigit() or not attempt.isdigit():
        raise BundleError("invalid_run_identity")
    suffix = hashlib.sha256((bundle_id + ":" + phase).encode()).hexdigest()[:12]
    return f"recovery-actions-{run}-{attempt}-{suffix}"


def _create_missing_blobs(repo, missing, blob_shas, max_workers=4):
    """Create independent Git blobs concurrently while keeping ref updates serial.

    Successful blob SHAs are retained in blob_shas even if a sibling upload
    fails, so an outer retry does not recreate already accepted objects.
    """
    if not missing:
        return

    def create_one(path, raw):
        item = api(
            f"/repos/{repo}/git/blobs", "POST",
            {"content": base64.b64encode(raw).decode(), "encoding": "base64"},
        )
        sha = item.get("sha") if isinstance(item, dict) else None
        if not BLOB_RE.fullmatch(sha or ""):
            raise BundleError("invalid_created_blob:" + path)
        return path, sha

    if len(missing) == 1 or max_workers <= 1:
        path, sha = create_one(*missing[0])
        blob_shas[path] = sha
        return

    first_error = None
    workers = min(max_workers, len(missing))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="git-blob") as pool:
        futures = [pool.submit(create_one, path, raw) for path, raw in missing]
        for future in as_completed(futures):
            try:
                path, sha = future.result()
                blob_shas[path] = sha
            except Exception as exc:
                if first_error is None:
                    first_error = exc
    if first_error is not None:
        raise first_error

def put_create_only(repo, files, message, recovery_ref=None, known_blob_shas=None):
    from private_io import _set_recovery_ref, _clear_recovery_ref
    repo = safe_repo(repo)
    blob_shas = {}
    known_blob_shas = dict(known_blob_shas or {})
    if set(known_blob_shas) - set(files):
        raise BundleError("known_blob_path_not_in_files")
    for path, raw in files.items():
        known = known_blob_shas.get(path)
        if known is not None:
            expected = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            if not BLOB_RE.fullmatch(known) or known != expected:
                raise BundleError("known_blob_identity_mismatch:" + path)
            blob_shas[path] = known
    for path, raw in files.items():
        safe_path(path)
        if not isinstance(raw, (bytes, bytearray)):
            raise BundleError("invalid_file_bytes")
    from git_publication_snapshot import existing_blob_snapshot, TreeSnapshotError
    for attempt in range(7):
        updating_ref = False
        try:
            parent = api(f"/repos/{repo}/git/ref/heads/main")["object"]["sha"]
            try:
                tree, existing = existing_blob_snapshot(api, repo, parent, files)
            except TreeSnapshotError as exc:
                raise BundleError(str(exc)) from exc
            missing = []
            for path, raw in files.items():
                expected = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
                if existing is None:
                    try:
                        old = content(repo, path, parent)
                    except urllib.error.HTTPError as exc:
                        if exc.code != 404:
                            raise
                        old_sha = None
                    else:
                        old_sha = old.get("sha")
                else:
                    old_sha = existing[path]
                if old_sha is not None:
                    if old_sha == expected:
                        continue
                    raise BundleError("refuse_different_existing_file:" + path)
                missing.append((path, raw))

            to_create = [(path, raw) for path, raw in missing if path not in blob_shas]
            _create_missing_blobs(repo, to_create, blob_shas)
            entries = [{
                "path": path, "mode": "100644", "type": "blob",
                "sha": blob_shas[path],
            } for path, _ in missing]
            if not entries:
                if recovery_ref:
                    _clear_recovery_ref(repo, recovery_ref)
                return parent
            new_tree = api(f"/repos/{repo}/git/trees", "POST",
                           {"base_tree": tree, "tree": entries})["sha"]
            commit = api(
                f"/repos/{repo}/git/commits", "POST",
                {"message": message, "tree": new_tree, "parents": [parent]},
            )["sha"]
            if recovery_ref:
                _set_recovery_ref(repo, recovery_ref, commit)
            updating_ref = True
            api(
                f"/repos/{repo}/git/refs/heads/main", "PATCH",
                {"sha": commit, "force": False},
            )
            if recovery_ref:
                _clear_recovery_ref(repo, recovery_ref)
            return commit
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {409, 429, 500, 502, 503, 504}
            limited_403 = exc.code == 403 and (
                (getattr(exc, "headers", None) or {}).get("Retry-After")
                or (getattr(exc, "headers", None) or {}).get("X-RateLimit-Remaining") == "0"
            )
            # A ref race can be 422. Other 422 validation errors are not retries.
            if exc.code == 422 and updating_ref:
                retryable = api(f"/repos/{repo}/git/ref/heads/main")["object"]["sha"] != parent
            if (not retryable and not limited_403) or attempt == 6:
                raise
            delay = _retry_delay(exc, attempt)
            if delay > 120:
                raise BundleError("retry_after_requires_later_resume") from exc
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 6:
                raise
            time.sleep(min(2 ** attempt, 16) + random.uniform(0.0, 0.4))
    raise BundleError("publication_failed")

def materialize(repo, bundle_id):
    if not BUNDLE_RE.fullmatch(bundle_id):
        raise BundleError("invalid_bundle_id")
    bundle_path = f"transport/bundle_inbox/{bundle_id}.json"
    raw, bundle_blob = read_repository_file(repo, bundle_path, "main")
    bundle, submitted_sources, inputs, timeout_seconds = validate_bundle(raw, bundle_id)

    complete_path = f"transport/bundle-executions/{bundle_id}/completed.json"
    try:
        complete = json.loads(decode_contents(content(repo, complete_path, "main"), 65536))
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    else:
        if (complete.get("bundle_blob") != bundle_blob
                or complete.get("status") != "materialized"
                or not REF_RE.fullmatch(complete.get("source_ref", ""))):
            raise BundleError("completed_bundle_identity_changed")
        append_output("skip", "true")
        append_output("source_ref", complete["source_ref"])
        append_output("bundle_blob", bundle_blob)
        append_output("timeout_seconds", str(timeout_seconds))
        print("MATERIALIZE_STAGE=already-completed")
        return

    successor = None
    known_blob_shas = {}
    if bundle["schema"] == SUCCESSOR_SCHEMA:
        base = _load_base_source(repo, bundle["base_source_ref"])
        sources, successor = _apply_successor_sources(
            base["sources"], submitted_sources, bundle.get("delete_files", [])
        )
        if set(inputs) & set(sources):
            raise BundleError("input_source_name_collision")
    else:
        sources = submitted_sources

    prefix = f"transport/tasks/bundle-materialized/{bundle_id}"
    source_manifest = {
        "schema": SUCCESSOR_SOURCE_SCHEMA if successor is not None else SOURCE_SCHEMA,
        "bundle_id": bundle_id,
        "bundle_path": bundle_path,
        "bundle_blob": bundle_blob,
        "files": {k: hashlib.sha256(v).hexdigest() for k, v in sorted(sources.items())},
        "inputs": inputs,
        "timeout_seconds": timeout_seconds,
    }
    if bundle["schema"] == TEXT_SCHEMA:
        source_manifest["submission_schema"] = TEXT_SCHEMA
        source_manifest["source_hash_authority"] = "actions_received_utf8"
    elif bundle["schema"] == SUCCESSOR_SCHEMA:
        source_manifest["submission_schema"] = SUCCESSOR_SCHEMA
        source_manifest["source_hash_authority"] = "actions_received_utf8_overlay"
        source_manifest["parent_source_ref"] = bundle["base_source_ref"]
        source_manifest["parent_bundle_id"] = base["bundle_id"]
        source_manifest.update(successor)
        for name in successor["inherited_files"]:
            if name in base["blob_shas"] and sources[name] == base["sources"][name]:
                known_blob_shas[f"{prefix}/{name}"] = base["blob_shas"][name]

    formal = {f"{prefix}/{name}": data for name, data in sources.items()}
    formal[f"{prefix}/BUNDLE_SOURCE.json"] = (
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    if successor is not None:
        source_ref = put_create_only(
            repo, formal, "Materialize validated private task bundle",
            recovery_ref=recovery_branch(bundle_id, "source"),
            known_blob_shas=known_blob_shas,
        )
    else:
        # Preserve the V2/V3 publisher call shape exactly for compatibility.
        source_ref = put_create_only(
            repo, formal, "Materialize validated private task bundle",
            recovery_ref=recovery_branch(bundle_id, "source"),
        )

    for name, data in sources.items():
        got, _ = read_repository_file(repo, f"{prefix}/{name}", source_ref)
        if got != data:
            raise BundleError("immutable_source_readback_mismatch:" + name)

    append_output("source_ref", source_ref)
    append_output("bundle_blob", bundle_blob)
    append_output("timeout_seconds", str(timeout_seconds))
    append_output("skip", "false")
    print("MATERIALIZE_STAGE=complete")


def fetch_task(repo, bundle_id, source_ref, dest):
    if not REF_RE.fullmatch(source_ref):
        raise BundleError("invalid_source_ref")
    prefix = f"transport/tasks/bundle-materialized/{bundle_id}"
    manifest_raw, _ = read_repository_file(
        repo, f"{prefix}/BUNDLE_SOURCE.json", source_ref
    )
    manifest = json.loads(manifest_raw)
    if (manifest.get("schema") not in (SOURCE_SCHEMA, SUCCESSOR_SOURCE_SCHEMA)
            or manifest.get("bundle_id") != bundle_id):
        raise BundleError("source_manifest_identity")
    target = Path(dest)
    target.mkdir(mode=0o700, parents=True, exist_ok=False)

    for name, digest in manifest["files"].items():
        raw, _ = read_repository_file(repo, f"{prefix}/{name}", source_ref)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise BundleError("source_fetch_hash_mismatch:" + name)
        p = target / name
        p.write_bytes(raw)
        p.chmod(0o400)

    for local, spec in manifest.get("inputs", {}).items():
        raw = read_pinned_file(repo, spec["path"], spec["ref"], spec["blob"])
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
    from bundle_output_collection import collect_output, inspect_collected
    collected, collection_errors = collect_output(output)

    errors = []
    def parse_metadata(name):
        if name not in collected:
            errors.append("missing_" + name)
            return {}
        try:
            value = json.loads(collected[name], parse_constant=_reject_constant)
            if not isinstance(value, dict):
                raise ValueError("metadata_not_object")
            return value
        except (ValueError, UnicodeError):
            errors.append("invalid_" + name)
            return {}
    summary = parse_metadata("summary.json")
    execution = parse_metadata("execution.json")
    expected_rows = summary.get("rows")
    result_validation, progress_validation = inspect_collected(
        collected, expected_rows, execution
    )
    success = (
        summary.get("stage") == "materialized"
        and execution.get("exit_code") == 0
        and not execution.get("timed_out")
        and not execution.get("output_limit_exceeded")
        and type(expected_rows) is int and expected_rows > 0
        and result_validation.get("status") == "valid"
        and not errors
        and not collection_errors
    )
    actual_rows = result_validation.get("records", 0)

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
        "status": "materialized" if success else "failed_or_partial",
        "result_validation": result_validation,
        "progress": progress_validation,
        "metadata_errors": errors,
        "collection_errors": collection_errors,
    }
    files = {f"{prefix}/{name}": raw for name, raw in collected.items()}
    files[f"{prefix}/receipt.json"] = (
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    ).encode()
    # Execution completion is not reviewer acceptance. Queue the immutable
    # delivery in the same private transaction, including partial failures.
    review_submission = {
        "schema": "private-review-submission/1",
        "review_status": "pending",
        "execution_status": receipt["status"],
        "next_action": "independent_review" if success else "execution_recovery",
        "bundle_id": bundle_id, "bundle_blob": bundle_blob,
        "source_ref": source_ref, "run_id": run, "run_attempt": attempt,
        "receipt_path": f"{prefix}/receipt.json",
        "result_path": f"{prefix}/result.jsonl" if "result.jsonl" in collected else None,
        "summary_path": f"{prefix}/summary.json" if "summary.json" in collected else None,
        "progress_path": f"{prefix}/progress.jsonl" if "progress.jsonl" in collected else None,
    }
    files[f"transport/reviews/pending/{bundle_id}/{run}-{attempt}.json"] = (
        json.dumps(review_submission, indent=2, sort_keys=True) + "\n"
    ).encode()
    if success:
        files[f"transport/bundle-executions/{bundle_id}/completed.json"] = (
            json.dumps(receipt, sort_keys=True) + "\n"
        ).encode()
    commit = put_create_only(repo, files, "Record bundle computation result",
                             recovery_ref=recovery_branch(bundle_id, "result"))
    append_output("result_commit", commit)
    if not success:
        print("PUBLISH_STAGE=failed-prefix-saved")
        raise BundleError("failed_execution_preserved")
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
    try:
        main()
    except Exception as exc:
        code = str(exc).split(":", 1)[0] if isinstance(exc, BundleError) else type(exc).__name__
        if not re.fullmatch(r"[A-Za-z0-9_]+", code):
            code = "unexpected_error"
        print("BUNDLE_STAGE=failed CODE=" + code, file=sys.stderr)
        raise SystemExit(2)
