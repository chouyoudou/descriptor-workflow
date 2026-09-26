"""Optional conveniences around the existing research interface, not a new runner.

Offline: index/select JSONL and export an already staged task. Authenticated host:
draft ordinary run requests, submit them, fetch a portable task or save local work.
No operation decides scientific cache compatibility or starts a cluster job.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.parse
import zipfile


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def rows(path):
    with Path(path).open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError("JSONL row is not an object")
            yield number, row, raw


def index_structures(path, output):
    """Cheap navigation metadata only; no labels, geometry calculation or sampling."""
    count = 0
    with Path(output).open("xb") as sink:
        for number, row, _ in rows(path):
            composition = Counter(row["species"])
            item = {"id": row["id"], "source_line": row.get("source_line", number),
                    "source_index": row.get("source_index"), "natoms": sum(composition.values()),
                    "composition": dict(sorted(composition.items()))}
            sink.write((json.dumps(item, ensure_ascii=False) + "\n").encode())
            count += 1
    return count


def select_jsonl(path, output, ids=None, source_lines=None):
    """Copy selected records byte-for-byte, retaining original IDs and line fields.

    When both selectors are present they are a union. With neither, copy all.
    A missing requested record is reported, not silently treated as computed.
    """
    wanted_ids, wanted_lines = set(ids or []), set(source_lines or [])
    found_ids, found_lines, count = set(), set(), 0
    target = Path(output)
    with target.open("xb") as sink:
        try:
            for number, row, raw in rows(path):
                key, line = row.get("id"), row.get("source_line", number)
                if not (wanted_ids or wanted_lines) or key in wanted_ids or line in wanted_lines:
                    sink.write(raw)
                    found_ids.add(key); found_lines.add(line); count += 1
            if wanted_ids - found_ids or wanted_lines - found_lines:
                raise ValueError("requested records missing from this saved file")
        except BaseException:
            target.unlink(missing_ok=True)
            raise
    return count


def make_run(receipt, bundle_id, config=None, inputs=None):
    """Use existing receipt fields; never inherit config or a completion status."""
    request = {"schema": "private-run-request/1", "bundle_id": bundle_id,
               "source_ref": receipt["source_ref"], "inputs_from": receipt["bundle_blob"]}
    if config is not None:
        request["config"] = config
    if inputs:
        request["inputs"] = inputs
    return request


def read_receipt(control, repo, path, ref=None):
    repo = control.safe_repo(repo)
    if ref is None:
        ref = control.api(f"/repos/{repo}/git/ref/heads/main")["object"]["sha"]
    raw, _ = control.read_repository_file(repo, path, ref)
    return json.loads(raw), ref


def output_binding(control, repo, receipt_path, receipt, filename, ref):
    """Turn a saved output into an ordinary pinned input without manual hash work."""
    if Path(filename).name != filename or "\\" in filename or filename in ("", ".", ".."):
        raise ValueError("output filename must be a basename")
    prefix = receipt.get("output_directory", str(Path(receipt_path).parent))
    path = prefix + "/" + filename
    item = control.content(repo, path, ref)
    if item.get("type") != "file":
        raise ValueError("saved output is not a file")
    return {"path": path, "ref": ref, "blob": item["sha"]}


@contextmanager
def public_token():
    # Serial host operations only. No token is exported into a task or packet.
    old = os.environ.get("PRIVATE_REPO_TOKEN")
    token = os.environ.get("GH_TOKEN") or old
    if token:
        os.environ["PRIVATE_REPO_TOKEN"] = token
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("PRIVATE_REPO_TOKEN", None)
        else:
            os.environ["PRIVATE_REPO_TOKEN"] = old


def submit(control, private_repo, public_repo, request):
    """One host command, the same private-write/public-title operations as before.

    A repeated invocation finds an existing activation rather than manufacturing
    another issue. Concurrent submitters still have no universal exactly-once claim.
    No alternate route after an explicit platform refusal is provided here.
    """
    private_repo, public_repo = map(control.safe_repo, (private_repo, public_repo))
    bundle_id = request.get("bundle_id", "")
    if not re.fullmatch(r"bt-[A-Za-z0-9_.-]{1,64}", bundle_id):
        raise ValueError("invalid bundle ID")
    if control.api(f"/repos/{private_repo}").get("private") is not True:
        raise ValueError("request destination must be private")
    path = f"transport/bundle_inbox/{bundle_id}.json"
    raw = json_bytes(request)
    commit = control.put_create_only(private_repo, {path: raw}, "Submit research request")
    print("SUBMIT_STAGE=private-request-saved ID=" + bundle_id)
    title, owner = "bundle-task:" + bundle_id, public_repo.split("/", 1)[0]
    with public_token():
        page = 1
        while True:
            query = urllib.parse.urlencode({"state": "all", "creator": owner,
                                             "per_page": 100, "page": page})
            items = control.api(f"/repos/{public_repo}/issues?{query}")
            for issue in items:
                if (issue.get("title") == title and not issue.get("pull_request")
                        and issue.get("user", {}).get("login") == owner):
                    return {"request_commit": commit, "issue": issue["html_url"],
                            "activation": "already_exists"}
            if len(items) < 100:
                break
            page += 1
        issue = control.api(f"/repos/{public_repo}/issues", "POST", {"title": title})
    return {"request_commit": commit, "issue": issue["html_url"], "activation": "created"}


def export_task(task, archive, origin=None, environment=None):
    """Package flat staged files, including checkpoints explicitly supplied as inputs.

    There is no archive extraction, hidden cache discovery or dependency install.
    The existing publisher can store the single ZIP as an ordinary private file.
    """
    task = Path(task)
    runner = Path(__file__).with_name("bundle_portable_runner.py")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as packet:
        for path in sorted(task.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise ValueError("export expects the existing flat regular-file task layout")
            packet.write(path, "task/" + path.name)
        packet.writestr("EXPORT.json", json_bytes({"origin": origin or {},
            "source_note": "Origin identifies the export, not subsequent local edits.",
            "environment_note": "Reference requirements are not an installed/native environment guarantee."}))
        packet.write(runner, "run_local.py")
        if environment is not None:
            packet.writestr("environment-reference.txt", environment)
        packet.writestr("README.txt", "Private research packet. Extract into a new private directory.\n"
            "Run: python3 run_local.py --output /absolute/path/to/new-output\n"
            "Use --python to choose an already prepared Python environment.\n"
            "The native launcher is NOT a sandbox; run only source you trust.\n"
            "No cluster queue, login, allocation or dependency install is selected.\n"
            "Saved inputs are included; algorithm-specific resume is not inferred.\n"
            "For return: bundle_workspace.py publish-local --task task --output-dir <output> "
            "--repo <private-repo> --id local-<unique-id>\n"
            "Local edits travel in a source/input snapshot; they are not silently a hosted source_ref.\n")
    return str(archive)


def export_from_receipt(control, repo, receipt_path, archive, ref=None):
    import bundle_ingress
    receipt, ref = read_receipt(control, repo, receipt_path, ref)
    with tempfile.TemporaryDirectory() as temp:
        task = Path(temp) / "task"
        bundle_ingress.fetch_task(repo, receipt["bundle_id"], receipt["source_ref"],
                                  receipt["bundle_blob"], task, control=control)
        lock = Path(__file__).parent / "public_env/resolved.lock"
        return export_task(task, archive, {"receipt_path": receipt_path, "receipt_ref": ref,
                            "source_ref": receipt["source_ref"], "bundle_blob": receipt["bundle_blob"]},
                           lock.read_bytes() if lock.exists() else None)


def publish_local(control, repo, task, output, run_id, collector=None):
    """Save local output plus actual current source/inputs; do not invent Actions success."""
    if collector is None:
        from bundle_output_collection import collect_output as collector
    repo = control.safe_repo(repo)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("invalid local run ID")
    if control.api(f"/repos/{repo}").get("private") is not True:
        raise ValueError("local delivery destination must be private")
    collected, errors = collector(output)
    prefix = "transport/executions/" + run_id
    snapshot = io.BytesIO()
    export_task(task, snapshot)
    receipt = {"schema": "workspace-local-delivery/1", "status": "local_output_saved",
               "output_directory": prefix + "/outputs", "files": sorted(collected),
               "collection_errors": errors, "source_snapshot": prefix + "/task.zip",
               "scientific_acceptance": "not_assessed", "producer": "local_not_GitHub_Actions"}
    files = {prefix + "/outputs/" + name: raw for name, raw in collected.items()}
    files[prefix + "/task.zip"] = snapshot.getvalue()
    files[prefix + "/local-delivery.json"] = json_bytes(receipt)
    commit = control.put_create_only(repo, files, "Save local research output and source snapshot")
    return {"commit": commit, "receipt_path": prefix + "/local-delivery.json",
            "status": receipt["status"], "collection_errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "select"):
        p = sub.add_parser(name); p.add_argument("input"); p.add_argument("--output", required=True)
        if name == "select":
            p.add_argument("--ids", nargs="*"); p.add_argument("--source-lines", nargs="*", type=int)
    p = sub.add_parser("draft")
    p.add_argument("--repo", required=True); p.add_argument("--receipt", required=True)
    p.add_argument("--id", required=True); p.add_argument("--config", type=json.loads)
    p.add_argument("--reuse", action="append", default=[], help="output basename or local=basename")
    p.add_argument("--output", required=True)
    p = sub.add_parser("submit")
    p.add_argument("request"); p.add_argument("--private-repo", required=True)
    p.add_argument("--public-repo", required=True)
    p = sub.add_parser("export")
    p.add_argument("--task"); p.add_argument("--repo"); p.add_argument("--receipt")
    p.add_argument("--output", required=True)
    p = sub.add_parser("publish-local")
    p.add_argument("--task", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--repo", required=True); p.add_argument("--id", required=True)
    args = parser.parse_args(argv)
    if args.command == "index":
        result = {"rows": index_structures(args.input, args.output)}
    elif args.command == "select":
        result = {"rows": select_jsonl(args.input, args.output, args.ids, args.source_lines)}
    elif args.command == "export" and args.task:
        result = {"archive": export_task(args.task, args.output)}
    else:
        import bundle_control as control
        if args.command == "draft":
            receipt, ref = read_receipt(control, args.repo, args.receipt)
            inputs = {}
            for value in args.reuse:
                local, sep, filename = value.partition("=")
                inputs[local] = output_binding(control, args.repo, args.receipt, receipt,
                                               filename if sep else local, ref)
            request = make_run(receipt, args.id, args.config, inputs)
            with Path(args.output).open("xb") as stream:
                stream.write(json_bytes(request))
            result = {"request": args.output}
        elif args.command == "submit":
            result = submit(control, args.private_repo, args.public_repo,
                            json.loads(Path(args.request).read_bytes()))
        elif args.command == "export":
            if not args.repo or not args.receipt:
                parser.error("export needs --task or both --repo and --receipt")
            result = {"archive": export_from_receipt(control, args.repo, args.receipt, args.output)}
        else:
            result = publish_local(control, args.repo, args.task, args.output_dir, args.id)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Request bodies and credentials must not become failure-log output.
        print("WORKSPACE_STAGE=failed TYPE=" + type(exc).__name__)
        raise SystemExit(1)
