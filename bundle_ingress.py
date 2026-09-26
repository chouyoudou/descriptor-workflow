"""Small trusted admission layer for immutable source and run requests.

Issues carry an opaque identifier only. An existing source can be run with new
inputs/config without source publication. No candidate runs during admission.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import urllib.error
import urllib.parse

ID_RE = re.compile(r"bt-[A-Za-z0-9_.-]{1,64}\Z")
REF_RE = re.compile(r"[0-9a-f]{40}\Z")
STATES = {"source_ready", "already_completed", "rejected"}
RUN_SCHEMA = "private-run-request/1"
CONFIG_NAME = "run_config.json"


def trigger_decision(event, actor, owner, triggering_actor=None):
    """Accept an owner-issued title, or the equivalent legacy two-line body."""
    rejected = {"ok": False, "bundle_id": "", "reason": "invalid_trigger"}
    if not isinstance(event, dict) or event.get("action") not in ("opened", "reopened"):
        return {**rejected, "reason": "unsupported_event"}
    issue = event.get("issue")
    if not isinstance(issue, dict) or "pull_request" in issue:
        return rejected
    if (not owner or actor != owner or (triggering_actor or actor) != owner
            or issue.get("user", {}).get("login") != owner):
        return {**rejected, "reason": "owner_required"}
    title, body = issue.get("title"), issue.get("body")
    if not isinstance(title, str) or (body is not None and not isinstance(body, str)):
        return rejected
    match = re.fullmatch(r"bundle-task:(bt-[A-Za-z0-9_.-]{1,64})", title.strip())
    if not match:
        return rejected
    bundle_id = match.group(1)
    lines = [line.strip() for line in (body or "").strip().splitlines() if line.strip()]
    if lines:
        expected = {"schema=private-bundle-trigger/2", "id=" + bundle_id}
        if len(lines) != 2 or set(lines) != expected:
            return {**rejected, "reason": "unexpected_public_body"}
    return {"ok": True, "bundle_id": bundle_id, "reason": "authorized"}


def emit(key, value):
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as stream:
            stream.write(f"{key}={value}\n")


def resolve(event_path):
    try:
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        event = None
    result = trigger_decision(
        event, os.environ.get("GITHUB_ACTOR", ""),
        os.environ.get("GITHUB_REPOSITORY_OWNER", ""),
        os.environ.get("GITHUB_TRIGGERING_ACTOR"),
    )
    emit("ok", "true" if result["ok"] else "false")
    emit("bundle_id", result["bundle_id"])
    print("RESOLVE_STAGE=" + ("authorized" if result["ok"] else "ignored"))
    print("RESOLVE_REASON=" + result["reason"])
    return result


def validate_run_request(control, request, bundle_id):
    """Validate a run, not a source edit. Config is plain JSON, never evaluated."""
    if (not isinstance(request, dict) or request.get("schema") != RUN_SCHEMA
            or request.get("bundle_id") != bundle_id):
        raise ValueError("run_request_identity_mismatch")
    ref = request.get("source_ref")
    if not isinstance(ref, str) or not REF_RE.fullmatch(ref):
        raise ValueError("invalid_run_source_ref")
    if {"files", "changed_files", "delete_files", "sha256", "base_source_ref"} & request.keys():
        raise ValueError("run_request_cannot_edit_source")
    inputs = request.get("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError("invalid_inputs")
    normalized = {}
    for name, spec in inputs.items():
        if (not isinstance(name, str) or not control.FILE_RE.fullmatch(name)
                or name == "BUNDLE_SOURCE.json" or not isinstance(spec, dict)):
            raise ValueError("invalid_input_entry")
        path = control.safe_path(spec.get("path"))
        if not any(path.startswith(p) for p in control.ALLOWED_INPUT_PREFIXES):
            raise ValueError("input_path_not_allowed")
        if any(not isinstance(spec.get(k), str) or not REF_RE.fullmatch(spec[k])
               for k in ("ref", "blob")):
            raise ValueError("input_identity_invalid")
        normalized[name] = {"path": path, "ref": spec["ref"], "blob": spec["blob"]}
    budget = request.get("timeout_seconds", 1800)
    if type(budget) is not int or not 30 <= budget <= 1800:
        raise ValueError("invalid_timeout")
    config = None
    if "config" in request:
        if not isinstance(request["config"], dict):
            raise ValueError("invalid_run_config")
        config = (json.dumps(request["config"], ensure_ascii=False, sort_keys=True,
                             allow_nan=False) + "\n").encode("utf-8")
        if CONFIG_NAME in normalized:
            raise ValueError("run_config_input_collision")
    return ref, normalized, budget, config


def check_run_names(base, inputs, config):
    if set(inputs) & set(base["sources"]):
        raise ValueError("input_source_name_collision")
    if config is not None and CONFIG_NAME in base["sources"]:
        raise ValueError("run_config_source_collision")


def run_outputs(control, repo, bundle_id, request, blob):
    """An immutable request reuses source_ref; it never writes source files."""
    ref, inputs, budget, config = validate_run_request(control, request, bundle_id)
    outputs = {"source_ref": ref, "bundle_blob": blob, "timeout_seconds": str(budget),
               "source_mode": "reused", "skip": "false"}
    path = f"transport/bundle-executions/{bundle_id}/completed.json"
    try:
        raw, _ = control.read_repository_file(repo, path, "main")
        completed = json.loads(raw)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    else:
        if (not isinstance(completed, dict) or completed.get("bundle_blob") != blob
                or completed.get("source_ref") != ref
                or completed.get("status") != "materialized"):
            raise ValueError("completed_run_identity_changed")
        return {**outputs, "skip": "true"}
    base = control._load_base_source(repo, ref)
    check_run_names(base, inputs, config)
    return outputs


def materialize_outputs(control, repo, bundle_id):
    """Route source submissions and source-reusing runs through one admission."""
    raw, blob = control.read_repository_file(
        repo, f"transport/bundle_inbox/{bundle_id}.json", "main")
    request = json.loads(raw)
    if isinstance(request, dict) and request.get("schema") == RUN_SCHEMA:
        return run_outputs(control, repo, bundle_id, request, blob)
    previous = os.environ.get("GITHUB_OUTPUT")
    captured = io.StringIO()
    try:
        with tempfile.TemporaryDirectory(prefix="bundle-admission-") as tmp:
            output = Path(tmp) / "outputs"
            os.environ["GITHUB_OUTPUT"] = str(output)
            with redirect_stdout(captured), redirect_stderr(captured):
                control.materialize(repo, bundle_id)
            values = {}
            for line in output.read_text(encoding="utf-8").splitlines():
                key, sep, value = line.partition("=")
                if not sep or key in values:
                    raise ValueError("invalid_materializer_outputs")
                values[key] = value
    finally:
        if previous is None:
            os.environ.pop("GITHUB_OUTPUT", None)
        else:
            os.environ["GITHUB_OUTPUT"] = previous
    if (values.get("skip") not in ("true", "false")
            or not REF_RE.fullmatch(values.get("source_ref", ""))
            or not REF_RE.fullmatch(values.get("bundle_blob", ""))):
        raise ValueError("invalid_materializer_identity")
    budget = values.get("timeout_seconds", "")
    if not budget.isdigit() or not 30 <= int(budget) <= 1800:
        raise ValueError("invalid_materializer_budget")
    return {key: values[key] for key in
            ("skip", "source_ref", "bundle_blob", "timeout_seconds")}


def original_source_ref(control, repo, bundle_id, outputs):
    """Resolve the original source commit after an ambiguous publication retry."""
    ref = outputs["source_ref"]
    manifest_path = f"transport/tasks/bundle-materialized/{bundle_id}/BUNDLE_SOURCE.json"
    raw, _ = control.read_repository_file(repo, manifest_path, ref)
    manifest = json.loads(raw)
    if (manifest.get("bundle_id") != bundle_id
            or manifest.get("bundle_blob") != outputs["bundle_blob"]):
        raise ValueError("materialized_manifest_identity_mismatch")
    query = urllib.parse.urlencode({"sha": ref, "path": manifest_path, "per_page": 1})
    history = control.api(f"/repos/{repo}/commits?{query}")
    if (not isinstance(history, list) or not history
            or not isinstance(history[0], dict)
            or not REF_RE.fullmatch(history[0].get("sha", ""))):
        raise ValueError("source_origin_unresolved")
    origin = history[0]["sha"]
    if origin != ref:
        verified = control._load_base_source(repo, origin)
        if verified["bundle_id"] != bundle_id or verified["manifest"] != manifest:
            raise ValueError("source_origin_identity_mismatch")
    return origin


def fetch_task(repo, bundle_id, source_ref, bundle_blob, dest, control=None):
    """Fetch the admitted immutable request, never a later mutable inbox body."""
    if control is None:
        import bundle_control as control
    repo = control.safe_repo(repo)
    if (not ID_RE.fullmatch(bundle_id) or not REF_RE.fullmatch(source_ref)
            or not REF_RE.fullmatch(bundle_blob)):
        raise ValueError("invalid_fetch_identity")
    item = control.api(f"/repos/{repo}/git/blobs/{bundle_blob}", max_response_bytes=None)
    if item.get("sha") != bundle_blob:
        raise ValueError("request_blob_identity_mismatch")
    raw = control.decode_contents({**item, "type": "file"})
    request = json.loads(raw)
    if not isinstance(request, dict) or request.get("bundle_id") != bundle_id:
        raise ValueError("run_request_identity_mismatch")
    if request.get("schema") != RUN_SCHEMA:
        manifest, _ = control.read_repository_file(
            repo, f"transport/tasks/bundle-materialized/{bundle_id}/BUNDLE_SOURCE.json", source_ref)
        if json.loads(manifest).get("bundle_blob") != bundle_blob:
            raise ValueError("source_request_identity_mismatch")
        return control.fetch_task(repo, bundle_id, source_ref, dest)
    ref, inputs, _, config = validate_run_request(control, request, bundle_id)
    if ref != source_ref:
        raise ValueError("run_source_identity_mismatch")
    base = control._load_base_source(repo, ref)
    check_run_names(base, inputs, config)
    target = Path(dest)
    target.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        def write(name, data):
            path = target / name
            with path.open("xb") as stream:
                stream.write(data)
            path.chmod(0o400)
        for name, data in base["sources"].items():
            write(name, data)
        for name, spec in inputs.items():
            data = control.read_pinned_file(
                repo, spec["path"], spec["ref"], spec["blob"])
            write(name, data)
        if config is not None:
            write(CONFIG_NAME, config)
    except Exception:
        # Only remove this invocation's newly created staging directory.
        shutil.rmtree(target)
        raise
    print("FETCH_TASK_STAGE=complete")


def error_category(exc):
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 404:
            return "missing_repository_object"
        if exc.code in (401, 403):
            return "authorization_or_rate_limit"
        if exc.code == 429 or exc.code >= 500:
            return "remote_service_error"
        return "remote_request_error"
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return "transport_error"
    return "validation_or_materialization_error"


def admit(repo, bundle_id, control=None):
    if control is None:
        import bundle_control as control
    repo = control.safe_repo(repo)
    if not isinstance(bundle_id, str) or not ID_RE.fullmatch(bundle_id):
        raise ValueError("invalid_bundle_id")
    run, attempt = os.environ.get("GITHUB_RUN_ID", ""), os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if not run.isdigit() or not attempt.isdigit():
        raise ValueError("invalid_run_identity")
    path = f"transport/bundle-executions/{bundle_id}/{run}-{attempt}/admission.json"
    record = {
        "schema": "private-bundle-admission/1", "bundle_id": bundle_id,
        "run_id": run, "run_attempt": attempt,
        "public_runtime_commit": os.environ.get("GITHUB_SHA"),
        "state": "rejected", "source_ref": None, "bundle_blob": None,
        "candidate_started_by_admission": False,
        "compute_success_claimed": False,
    }
    outputs = None
    failure = None
    try:
        outputs = materialize_outputs(control, repo, bundle_id)
        if outputs["skip"] != "true" and outputs.get("source_mode") != "reused":
            outputs["source_ref"] = original_source_ref(control, repo, bundle_id, outputs)
        record.update(
            state="already_completed" if outputs["skip"] == "true" else "source_ready",
            source_ref=outputs["source_ref"], bundle_blob=outputs["bundle_blob"],
            timeout_seconds=int(outputs["timeout_seconds"]),
            source_mode=outputs.get("source_mode", "materialized"),
        )
    except Exception as exc:
        failure = exc
        record.update(error_category=error_category(exc),
                      error_class=type(exc).__name__, detail=str(exc))
    raw = (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            control.put_create_only(
                repo, {path: raw}, "Record bundle admission outcome",
                recovery_ref=control.recovery_branch(bundle_id, "admission"),
            )
    except Exception:
        print("ADMISSION_STAGE=receipt_unavailable")
        raise RuntimeError("admission_receipt_unavailable") from None
    if failure is not None:
        print("ADMISSION_STAGE=rejected")
        raise RuntimeError("admission_rejected") from None
    for key, value in outputs.items():
        emit(key, value)
    print("ADMISSION_STAGE=" + record["state"])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    resolver = sub.add_parser("resolve")
    resolver.add_argument("--event", required=True)
    admission = sub.add_parser("admit")
    admission.add_argument("--repo", required=True)
    admission.add_argument("--bundle-id", required=True)
    fetcher = sub.add_parser("fetch-task")
    for name in ("repo", "bundle-id", "source-ref", "bundle-blob", "dest"):
        fetcher.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        if args.command == "resolve":
            resolve(args.event)
        elif args.command == "admit":
            admit(args.repo, args.bundle_id)
        else:
            fetch_task(args.repo, args.bundle_id, args.source_ref, args.bundle_blob, args.dest)
    except Exception:
        print("INGRESS_STAGE=failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
