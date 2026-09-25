"""Small trusted admission layer for immutable bundle execution.

Issues carry an opaque identifier only. Admission runs before dependency setup;
source generation, validation and publication remain in bundle_control.py.
No candidate is executed here and no external submit service is required.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.error
import urllib.parse

ID_RE = re.compile(r"bt-[A-Za-z0-9_.-]{1,64}\Z")
REF_RE = re.compile(r"[0-9a-f]{40}\Z")
STATES = {"source_ready", "already_completed", "rejected"}


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


def materialize_outputs(control, repo, bundle_id):
    """Use the existing materializer's output contract without echoing details.

    A temporary GITHUB_OUTPUT belongs to this trusted step only. The final job
    outputs are released only after the private admission receipt is durable.
    """
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
    """Resolve the original source commit after an ambiguous publication retry.

    put_create_only may return today's main if identical files already exist.
    That is a valid read ref but not necessarily the original source commit.
    Follow the immutable manifest's last change and revalidate if it differs.
    """
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
        # Completed receipts already contain their exact production source ref.
        if outputs["skip"] != "true":
            outputs["source_ref"] = original_source_ref(control, repo, bundle_id, outputs)
        record.update(
            state="already_completed" if outputs["skip"] == "true" else "source_ready",
            source_ref=outputs["source_ref"], bundle_blob=outputs["bundle_blob"],
            timeout_seconds=int(outputs["timeout_seconds"]),
        )
    except Exception as exc:
        failure = exc
        # Detailed reason stays private. Public logs contain only fixed enums.
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
    args = parser.parse_args()
    try:
        if args.command == "resolve":
            resolve(args.event)
        else:
            admit(args.repo, args.bundle_id)
    except Exception:
        # No traceback, API response body, source name, token or private path.
        print("INGRESS_STAGE=failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
