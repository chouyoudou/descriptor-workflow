#!/usr/bin/env python3
"""Validate a non-sensitive public trigger and derive its private task descriptor.

The public request never contains private repository paths, source code, data, or
credentials. Version 1 preserves the original single-pointer workflow. Version 2
maps an opaque trigger id to a create-only private task descriptor, allowing
multiple agents to prepare independent tasks without racing on active_task.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping

LEGACY_TASK_PATH = "transport/active_task.json"
TASK_PREFIX = "transport/task_requests"
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")


class RequestContractError(ValueError):
    """The public trigger envelope is malformed or over-scoped."""


def _trigger_id(value: Any) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise RequestContractError("invalid_trigger_id")
    return value


def resolve_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bounded task lookup contract for one public request."""
    if not isinstance(payload, Mapping):
        raise RequestContractError("request_must_be_object")

    keys = set(payload)
    if keys == {"id"}:
        trigger_id = _trigger_id(payload["id"])
        return {
            "schema": "scoped-private-trigger/1",
            "trigger_id": trigger_id,
            "task_path": LEGACY_TASK_PATH,
            "parallel_preparation_safe": False,
        }

    if keys == {"version", "id"} and payload.get("version") == 2:
        trigger_id = _trigger_id(payload["id"])
        return {
            "schema": "scoped-private-trigger/2",
            "trigger_id": trigger_id,
            "task_path": f"{TASK_PREFIX}/{trigger_id}.json",
            "parallel_preparation_safe": True,
        }

    raise RequestContractError("unsupported_request_shape")


def load_request(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RequestContractError("request_not_found") from exc
    except json.JSONDecodeError as exc:
        raise RequestContractError("invalid_request_json") from exc
    except OSError as exc:
        raise RequestContractError("request_read_failed") from exc
    return resolve_request(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    try:
        contract = load_request(args.request)
    except RequestContractError as exc:
        print(f"REQUEST_STAGE=failed REASON={exc}")
        return 2

    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"trigger_id={contract['trigger_id']}\n")
            handle.write(f"task_path={contract['task_path']}\n")
            handle.write(
                "parallel_preparation_safe="
                + ("true" if contract["parallel_preparation_safe"] else "false")
                + "\n"
            )

    print(
        "REQUEST_STAGE=resolved "
        f"SCHEMA={contract['schema']} "
        f"PARALLEL_PREPARATION_SAFE="
        f"{str(contract['parallel_preparation_safe']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
