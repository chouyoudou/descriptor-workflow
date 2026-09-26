"""Resume a durable private result transaction, never re-execute its candidate.

Uses the existing publisher's recovery-actions-* result refs. No new storage,
credentials, source protocol or scientific success definition is introduced.
A missing durable snapshot cannot be recovered by this code.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.error

HEX = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
ID = re.compile(r"bt-[A-Za-z0-9_.-]{1,64}\Z")
ATTEMPT = re.compile(r"([1-9][0-9]*)-([1-9][0-9]*)\Z")


def load(raw):
    def bad(_):
        raise ValueError("nonfinite_metadata")
    value = json.loads(raw, parse_constant=bad)
    if not isinstance(value, dict):
        raise ValueError("metadata_not_object")
    return value


def optional(control, repo, path, ref):
    try:
        return control.read_repository_file(repo, path, ref)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        return None


def tree_at(control, repo, path, ref):
    """Read a non-recursive tree at a fixed commit, not a capped Contents list."""
    tree = control.api(f"/repos/{repo}/git/commits/{ref}")["tree"]["sha"]
    for part in [*path.split("/"), None]:
        listing = control.api(f"/repos/{repo}/git/trees/{tree}")
        if listing.get("truncated") or not isinstance(listing.get("tree"), list):
            raise ValueError("incomplete_git_tree")
        entries = {e["path"]: e for e in listing["tree"]}
        if part is None:
            return entries
        if part not in entries:
            return {}
        if entries[part].get("type") != "tree":
            raise ValueError("expected_result_directory")
        tree = entries[part]["sha"]


def receipt_identity(receipt, bundle_id, bundle_blob, producer=None):
    if (receipt.get("schema") != "bundle-computation-receipt/2"
            or receipt.get("bundle_id") != bundle_id
            or receipt.get("bundle_blob") != bundle_blob
            or not HEX.fullmatch(receipt.get("source_ref", ""))
            or receipt.get("status") not in ("materialized", "failed_or_partial")):
        raise ValueError("result_identity_mismatch")
    identity = str(receipt.get("run_id", "")) + "-" + str(receipt.get("run_attempt", ""))
    if not ATTEMPT.fullmatch(identity) or (producer is not None and identity != producer):
        raise ValueError("producer_identity_mismatch")
    if not isinstance(receipt.get("files"), dict):
        raise ValueError("invalid_result_manifest")
    for name, digest in receipt["files"].items():
        if (not isinstance(name, str) or name in ("", ".", "..", "receipt.json", "admission.json")
                or "/" in name or "\\" in name or "\0" in name
                or not isinstance(digest, str) or not DIGEST.fullmatch(digest)):
            raise ValueError("invalid_result_manifest_entry")
    return identity


def bind_source(control, repo, request, receipt, ref):
    """Bind the result to the immutable request; do not run or rewrite source."""
    if request.get("schema") == "private-run-request/1":
        if request.get("source_ref") != receipt["source_ref"]:
            raise ValueError("request_result_source_mismatch")
    else:
        path = f"transport/tasks/bundle-materialized/{receipt['bundle_id']}/BUNDLE_SOURCE.json"
        raw, _ = control.read_repository_file(repo, path, receipt["source_ref"])
        source = load(raw)
        if (source.get("bundle_id") != receipt["bundle_id"]
                or source.get("bundle_blob") != receipt["bundle_blob"]):
            raise ValueError("source_request_identity_mismatch")


def verify_snapshot(control, repo, bundle_id, bundle_blob, request, snapshot, producer,
                    inspector=None):
    """Recheck exact original bytes/producer scope; never transplant a whole tree."""
    if inspector is None:
        from private_io import inspect_result_jsonl as inspector
    prefix = f"transport/bundle-executions/{bundle_id}/{producer}"
    entries = tree_at(control, repo, prefix, snapshot)
    receipt_path = prefix + "/receipt.json"
    receipt_raw, _ = control.read_repository_file(repo, receipt_path, snapshot)
    receipt = load(receipt_raw)
    receipt_identity(receipt, bundle_id, bundle_blob, producer)
    bind_source(control, repo, request, receipt, snapshot)
    files, blobs = {}, {}
    cached_trees = {prefix: entries}
    def collect(path):
        directory, name = path.rsplit("/", 1)
        if directory not in cached_trees:
            cached_trees[directory] = tree_at(control, repo, directory, snapshot)
        entry = cached_trees[directory].get(name, {})
        if entry.get("type") != "blob" or entry.get("mode") != "100644":
            raise ValueError("nonregular_snapshot_file")
        raw, sha = control.read_repository_file(repo, path, snapshot)
        if sha != entry["sha"]:
            raise ValueError("snapshot_tree_blob_mismatch")
        files[path] = raw
        blobs[path] = sha
        return raw
    for name, digest in receipt["files"].items():
        entry = entries.get(name, {})
        if entry.get("type") != "blob" or entry.get("mode") != "100644":
            raise ValueError("nonregular_snapshot_file")
        raw = collect(prefix + "/" + name)
        if blobs[prefix + "/" + name] != entry["sha"] or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("snapshot_hash_mismatch")
    if entries.get("receipt.json", {}).get("mode") != "100644":
        raise ValueError("nonregular_snapshot_receipt")
    collect(receipt_path)
    summary_raw = files.get(prefix + "/summary.json")
    execution_raw = files.get(prefix + "/execution.json")
    try:
        summary = load(summary_raw) if summary_raw is not None else {}
        execution = load(execution_raw) if execution_raw is not None else {}
    except (ValueError, TypeError, UnicodeError):
        if receipt["status"] == "materialized":
            raise ValueError("invalid_success_metadata")
        summary, execution = {}, {}
    with tempfile.TemporaryDirectory(prefix="result-check-") as tmp:
        path = Path(tmp) / "result.jsonl"
        if prefix + "/result.jsonl" in files:
            path.write_bytes(files[prefix + "/result.jsonl"])
        validation = inspector(path, summary.get("rows"))
    if (validation != receipt.get("result_validation")
            or validation.get("records", 0) != receipt.get("result_rows")):
        raise ValueError("saved_result_validation_mismatch")
    if receipt["status"] == "materialized":
        if not (summary.get("stage") == "materialized"
                and execution.get("exit_code") == 0
                and not execution.get("timed_out")
                and not execution.get("output_limit_exceeded")
                and type(summary.get("rows")) is int and summary["rows"] > 0
                and validation.get("status") == "valid"
                and not receipt.get("metadata_errors")):
            raise ValueError("false_completion_rejected")
    review_path = f"transport/reviews/pending/{bundle_id}/{producer}.json"
    review = load(collect(review_path))
    for key in ("bundle_id", "bundle_blob", "source_ref", "run_id", "run_attempt"):
        if review.get(key) != receipt.get(key):
            raise ValueError("review_identity_mismatch")
    if (review.get("execution_status") != receipt["status"]
            or review.get("receipt_path") != receipt_path):
        raise ValueError("review_status_mismatch")
    completed_path = f"transport/bundle-executions/{bundle_id}/completed.json"
    if receipt["status"] == "materialized":
        if load(collect(completed_path)) != receipt:
            raise ValueError("completed_receipt_mismatch")
    changes = control._commit_files(repo, snapshot)
    if any(e.get("filename") not in files or e.get("status") == "removed" for e in changes):
        raise ValueError("snapshot_changes_outside_delivery")
    return receipt, files, blobs


def saved_attempts(control, repo, bundle_id, ref):
    root = f"transport/bundle-executions/{bundle_id}"
    entries = tree_at(control, repo, root, ref)
    attempts = sorted((n for n, e in entries.items()
                       if ATTEMPT.fullmatch(n) and e.get("type") == "tree"),
                      key=lambda s: tuple(map(int, s.split("-"))), reverse=True)
    receipts, uncertain = [], []
    for producer in attempts:
        item = optional(control, repo, f"{root}/{producer}/receipt.json", ref)
        if item:
            receipts.append((producer, load(item[0])))
            continue
        admission = optional(control, repo, f"{root}/{producer}/admission.json", ref)
        current = os.environ.get("GITHUB_RUN_ID", "") + "-" + os.environ.get("GITHUB_RUN_ATTEMPT", "")
        if admission and producer != current and load(admission[0]).get("state") == "source_ready":
            uncertain.append(producer)
    return receipts, uncertain


def resume(control, repo, bundle_id, request, bundle_blob, inspector=None):
    """Return published evidence, None for first run, or an explicit unknown state."""
    parent = control.api(f"/repos/{repo}/git/ref/heads/main")["object"]["sha"]
    complete = optional(control, repo, f"transport/bundle-executions/{bundle_id}/completed.json", parent)
    if complete:
        receipt = load(complete[0])
        receipt_identity(receipt, bundle_id, bundle_blob)
        bind_source(control, repo, request, receipt, parent)
        return {"state": "already_published", "receipt": receipt, "result_commit": parent}
    suffix = hashlib.sha256((bundle_id + ":result").encode()).hexdigest()[:12]
    pattern = re.compile(r"refs/heads/(recovery-actions-([1-9][0-9]*)-([1-9][0-9]*)-" + suffix + r")\Z")
    refs = control.api(f"/repos/{repo}/git/matching-refs/heads/recovery-actions-")
    if not isinstance(refs, list):
        raise ValueError("invalid_recovery_ref_list")
    matches = [(pattern.fullmatch(r.get("ref", "")), r) for r in refs if isinstance(r, dict)]
    recovered = []
    for match, entry in sorted(((m, r) for m, r in matches if m),
                               key=lambda pair: (int(pair[0][2]), int(pair[0][3]))):
        branch, run, attempt = match.groups()
        snapshot = entry.get("object", {}).get("sha", "")
        if entry.get("object", {}).get("type") != "commit" or not HEX.fullmatch(snapshot):
            raise ValueError("invalid_recovery_commit")
        receipt, files, blobs = verify_snapshot(control, repo, bundle_id, bundle_blob,
                                               request, snapshot, run + "-" + attempt, inspector)
        commit = control.put_create_only(repo, files, "Recover original result publication",
                                        recovery_ref=branch, known_blob_shas=blobs)
        for path, raw in files.items():
            actual, _ = control.read_repository_file(repo, path, commit)
            if actual != raw:
                raise ValueError("recovered_readback_mismatch")
        recovered.append({"producer": run + "-" + attempt, "snapshot": snapshot,
                          "published_commit": commit, "status": receipt["status"]})
    if recovered:
        return {"state": "publication_recovered", "receipt": receipt,
                "result_commit": commit, "recovered": recovered}
    receipts, uncertain = saved_attempts(control, repo, bundle_id, parent)
    if receipts:
        for producer, receipt in receipts:
            receipt_identity(receipt, bundle_id, bundle_blob, producer)
            bind_source(control, repo, request, receipt, parent)
        return {"state": "already_published_partial", "receipt": receipts[0][1],
                "result_commit": parent, "saved_producers": [p for p, _ in receipts]}
    if uncertain:
        return {"state": "prior_execution_unknown", "producer_attempts": uncertain}
    return None


def admit(repo, bundle_id, control=None, admission=None):
    if control is None:
        import bundle_control as control
    if admission is None:
        from bundle_ingress import admit as admission
    repo = control.safe_repo(repo)
    if not isinstance(bundle_id, str) or not ID.fullmatch(bundle_id):
        raise ValueError("invalid_bundle_id")
    raw, blob = control.read_repository_file(repo, f"transport/bundle_inbox/{bundle_id}.json", "main")
    request = load(raw)
    if request.get("bundle_id") != bundle_id:
        raise ValueError("request_identity_mismatch")
    # No candidate code is loaded or invoked by the recovery path.
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            outcome = resume(control, repo, bundle_id, request, blob)
    except Exception as exc:
        outcome = {"state": "recovery_needs_inspection", "error_class": type(exc).__name__,
                   "detail": str(exc)}
    if outcome is None:
        return admission(repo, bundle_id, control)
    run, attempt = os.environ.get("GITHUB_RUN_ID", ""), os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if not ATTEMPT.fullmatch(run + "-" + attempt):
        raise ValueError("invalid_recovery_run")
    record = {"schema": "private-result-recovery/1", "bundle_id": bundle_id,
              "bundle_blob": blob, "recovery_run_id": run, "recovery_run_attempt": attempt,
              "candidate_executed": False, **outcome}
    path = f"transport/bundle-executions/{bundle_id}/{run}-{attempt}/publication-recovery.json"
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        control.put_create_only(repo, {path: (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()},
                                "Record publication-only recovery")
    if outcome["state"] in ("prior_execution_unknown", "recovery_needs_inspection"):
        raise RuntimeError("recovery_needs_inspection_no_automatic_recompute")
    receipt = outcome["receipt"]
    for name, value in {"skip": "true", "source_ref": receipt["source_ref"],
                        "bundle_blob": blob, "timeout_seconds": str(request.get("timeout_seconds", 1800)),
                        "recovery_state": outcome["state"]}.items():
        control.append_output(name, value)
    print("RESULT_RECOVERY_STAGE=" + outcome["state"])
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--bundle-id", required=True)
    a = p.parse_args()
    try:
        admit(a.repo, a.bundle_id)
        return 0
    except Exception:
        # All detailed result bytes/paths stay in the private repository.
        print("RESULT_RECOVERY_STAGE=needs_inspection")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
