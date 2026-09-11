#!/usr/bin/env python3
"""Validate and discover non-sensitive public task triggers.

Legacy requests keep the original single pointer. Version 2 maps an opaque id to
one immutable private task descriptor. Version 3 additionally assigns the task
to a bounded execution lane/slot and validates that the public slot filename
matches that scheduling identity.

For push events, the head commit must change exactly one trigger file and no
other file. This prevents a trigger commit from silently carrying code or
workflow changes. Slot files may be updated repeatedly; each workflow run is
still pinned to its own public commit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

LEGACY_REQUEST_PATH = "private_job/request.json"
TASK_PREFIX = "transport/task_requests"
SLOT_PREFIX = "private_job/slots"

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
LANE_SLOTS = {
    "descriptor": 20,
    "ml": 8,
    "infrastructure": 2,
}


class RequestContractError(ValueError):
    """The public trigger envelope or event is malformed or over-scoped."""


def _trigger_id(value: Any) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise RequestContractError("invalid_trigger_id")
    return value


def _lane_slot(lane: Any, slot: Any) -> tuple[str, int]:
    if not isinstance(lane, str) or lane not in LANE_SLOTS:
        raise RequestContractError("invalid_execution_lane")
    if isinstance(slot, bool) or not isinstance(slot, int):
        raise RequestContractError("invalid_execution_slot")
    if slot < 0 or slot >= LANE_SLOTS[lane]:
        raise RequestContractError("execution_slot_out_of_range")
    return lane, slot


def slot_request_path(lane: str, slot: int) -> str:
    return f"{SLOT_PREFIX}/{lane}-{slot:02d}.json"


def resolve_request(
    payload: Mapping[str, Any], *, request_path: str | None = None
) -> dict[str, Any]:
    """Return the bounded private-task lookup and scheduling contract."""
    if not isinstance(payload, Mapping):
        raise RequestContractError("request_must_be_object")

    keys = set(payload)
    if keys == {"id"}:
        trigger_id = _trigger_id(payload["id"])
        if request_path is not None and request_path != LEGACY_REQUEST_PATH:
            raise RequestContractError("legacy_request_path_mismatch")
        return {
            "schema": "scoped-private-trigger/1",
            "trigger_id": trigger_id,
            "task_path": "transport/active_task.json",
            "request_path": LEGACY_REQUEST_PATH,
            "lane": "descriptor",
            "slot": 0,
            "parallel_preparation_safe": False,
            "parallel_execution_safe": False,
        }

    if keys == {"version", "id"} and payload.get("version") == 2:
        trigger_id = _trigger_id(payload["id"])
        if request_path is not None and request_path != LEGACY_REQUEST_PATH:
            raise RequestContractError("v2_request_path_mismatch")
        return {
            "schema": "scoped-private-trigger/2",
            "trigger_id": trigger_id,
            "task_path": f"{TASK_PREFIX}/{trigger_id}.json",
            "request_path": LEGACY_REQUEST_PATH,
            "lane": "descriptor",
            "slot": 0,
            "parallel_preparation_safe": True,
            "parallel_execution_safe": False,
        }

    if keys == {"version", "id", "lane", "slot"} and payload.get("version") == 3:
        trigger_id = _trigger_id(payload["id"])
        lane, slot = _lane_slot(payload["lane"], payload["slot"])
        expected_path = slot_request_path(lane, slot)
        if request_path is not None and request_path != expected_path:
            raise RequestContractError("slot_request_path_mismatch")
        return {
            "schema": "scoped-private-trigger/3",
            "trigger_id": trigger_id,
            "task_path": f"{TASK_PREFIX}/{trigger_id}.json",
            "request_path": expected_path,
            "lane": lane,
            "slot": slot,
            "parallel_preparation_safe": True,
            "parallel_execution_safe": True,
        }

    raise RequestContractError("unsupported_request_shape")


def load_request(path: Path, *, repository_path: str | None = None) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RequestContractError("request_not_found") from exc
    except json.JSONDecodeError as exc:
        raise RequestContractError("invalid_request_json") from exc
    except OSError as exc:
        raise RequestContractError("request_read_failed") from exc
    return resolve_request(payload, request_path=repository_path)


def _commit_changes(commit: Mapping[str, Any]) -> tuple[set[str], set[str], set[str]]:
    def paths(name: str) -> set[str]:
        value = commit.get(name, [])
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise RequestContractError("invalid_push_change_list")
        out = set()
        for item in value:
            if not isinstance(item, str):
                raise RequestContractError("invalid_push_change_path")
            out.add(item)
        return out

    return paths("added"), paths("modified"), paths("removed")


def discover_request(
    event: Mapping[str, Any], *, event_name: str, repository_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Discover one trigger file from a workflow event without searching."""
    if event_name == "workflow_dispatch":
        path = LEGACY_REQUEST_PATH
        return repository_root / path, load_request(
            repository_root / path, repository_path=path
        )
    if event_name != "push":
        raise RequestContractError("unsupported_event")

    commits = event.get("commits")
    if not isinstance(commits, list) or len(commits) != 1:
        raise RequestContractError("push_must_contain_one_commit")
    added, modified, removed = _commit_changes(commits[0])
    changed = added | modified | removed
    candidates = {
        path
        for path in changed
        if path == LEGACY_REQUEST_PATH
        or (
            path.startswith(SLOT_PREFIX + "/")
            and path.endswith(".json")
            and "/" not in path[len(SLOT_PREFIX) + 1 :]
        )
    }
    if len(candidates) != 1:
        raise RequestContractError("push_must_change_one_trigger")
    path = next(iter(candidates))
    if changed != {path}:
        raise RequestContractError("trigger_commit_must_be_isolated")
    if path in removed:
        raise RequestContractError("trigger_file_removed")
    if path != LEGACY_REQUEST_PATH and path not in (added | modified):
        raise RequestContractError("slot_trigger_not_written")

    file_path = repository_root / path
    return file_path, load_request(file_path, repository_path=path)


def _write_outputs(path: Path, contract: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for key in (
            "trigger_id",
            "task_path",
            "request_path",
            "lane",
            "slot",
        ):
            handle.write(f"{key}={contract[key]}\n")
        handle.write(
            "parallel_preparation_safe="
            + str(bool(contract["parallel_preparation_safe"])).lower()
            + "\n"
        )
        handle.write(
            "parallel_execution_safe="
            + str(bool(contract["parallel_execution_safe"])).lower()
            + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", type=Path)
    source.add_argument("--event", type=Path)
    parser.add_argument("--event-name")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    try:
        if args.request is not None:
            contract = load_request(args.request)
        else:
            if not args.event_name:
                raise RequestContractError("missing_event_name")
            event = json.loads(args.event.read_text(encoding="utf-8"))
            _, contract = discover_request(
                event,
                event_name=args.event_name,
                repository_root=args.repository_root,
            )
    except (RequestContractError, json.JSONDecodeError, OSError) as exc:
        reason = exc if isinstance(exc, RequestContractError) else type(exc).__name__
        print(f"REQUEST_STAGE=failed REASON={reason}")
        return 2

    if args.github_output is not None:
        _write_outputs(args.github_output, contract)

    print(
        "REQUEST_STAGE=resolved "
        f"SCHEMA={contract['schema']} "
        f"LANE={contract['lane']} SLOT={contract['slot']} "
        "PARALLEL_EXECUTION_SAFE="
        f"{str(contract['parallel_execution_safe']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
