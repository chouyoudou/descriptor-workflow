#!/usr/bin/env python3
"""Content-addressed public runtime-image cache helpers.

The cache contains only a Docker image assembled from the repository's pinned
public base image, public lock, and verified public wheelhouse. Private task
source, inputs, outputs, logs, and credentials are never inputs to this cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

SCHEMA = "descriptor-public-runtime-profile/1"
PROFILE_NAME = "local-environment-cpu-v1"


class RuntimeCacheError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_identity(
    *, lock: Path, manifest: Path, dockerfile: Path, base_image: str
) -> dict[str, Any]:
    if "@sha256:" not in base_image:
        raise RuntimeCacheError("base_image_must_be_immutable")
    components = {
        "schema": SCHEMA,
        "profile": PROFILE_NAME,
        "base_image": base_image,
        "lock_sha256": sha256_file(lock),
        "wheel_manifest_sha256": sha256_file(manifest),
        "dockerfile_sha256": sha256_file(dockerfile),
        "python_abi": "cp312",
        "platform": "linux/amd64",
    }
    canonical = json.dumps(
        components, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    components["profile_digest"] = hashlib.sha256(canonical).hexdigest()
    return components


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeCacheError("runtime_verification_command_failed")
    return completed


def verify_image(
    *,
    image: str,
    identity: dict[str, Any],
    lock: Path,
    public_deps: Path,
) -> dict[str, Any]:
    inspect = json.loads(_run(["docker", "image", "inspect", image]).stdout)
    if not isinstance(inspect, list) or len(inspect) != 1:
        raise RuntimeCacheError("unexpected_image_inspect")
    image_data = inspect[0]
    if image_data.get("Os") != "linux" or image_data.get("Architecture") != "amd64":
        raise RuntimeCacheError("runtime_platform_mismatch")
    labels = (image_data.get("Config") or {}).get("Labels") or {}
    expected_labels = {
        "io.descriptor.runtime.profile": PROFILE_NAME,
        "io.descriptor.runtime.profile-digest": identity["profile_digest"],
        "io.descriptor.runtime.lock-sha256": identity["lock_sha256"],
        "io.descriptor.runtime.base-image": identity["base_image"],
    }
    if any(labels.get(key) != value for key, value in expected_labels.items()):
        raise RuntimeCacheError("runtime_label_mismatch")

    common = [
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--security-opt", "no-new-privileges:true", "--cap-drop", "ALL",
        "--pids-limit", "64", "--memory", "1g", "--cpus", "1",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777",
        "--mount", f"type=bind,src={public_deps.resolve()},dst=/verify/public_deps.py,readonly",
        "--mount", f"type=bind,src={lock.resolve()},dst=/verify/resolved.lock,readonly",
        image, "python3", "-B", "/verify/public_deps.py",
    ]
    installed = _run(common + ["verify-installed", "/verify/resolved.lock"])
    smoke = _run(common + ["smoke"])
    return {
        "schema": "descriptor-public-runtime-verification/1",
        "profile": PROFILE_NAME,
        "profile_digest": identity["profile_digest"],
        "image": image,
        "image_id": image_data.get("Id"),
        "platform": "linux/amd64",
        "installed_check": json.loads(installed.stdout),
        "smoke": json.loads(smoke.stdout),
    }


def _write_outputs(path: Path, identity: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for key in (
            "profile", "profile_digest", "lock_sha256",
            "wheel_manifest_sha256", "dockerfile_sha256", "base_image",
        ):
            handle.write(f"{key}={identity[key]}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    identity_parser = sub.add_parser("identity")
    identity_parser.add_argument("--lock", type=Path, required=True)
    identity_parser.add_argument("--manifest", type=Path, required=True)
    identity_parser.add_argument("--dockerfile", type=Path, required=True)
    identity_parser.add_argument("--base-image", required=True)
    identity_parser.add_argument("--github-output", type=Path)
    identity_parser.add_argument("--json-output", type=Path)

    verify_parser = sub.add_parser("verify-image")
    verify_parser.add_argument("--image", required=True)
    verify_parser.add_argument("--lock", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--dockerfile", type=Path, required=True)
    verify_parser.add_argument("--base-image", required=True)
    verify_parser.add_argument("--public-deps", type=Path, required=True)
    verify_parser.add_argument("--json-output", type=Path)

    args = parser.parse_args()
    try:
        identity = profile_identity(
            lock=args.lock,
            manifest=args.manifest,
            dockerfile=args.dockerfile,
            base_image=args.base_image,
        )
        if args.command == "identity":
            result = identity
            if args.github_output is not None:
                _write_outputs(args.github_output, identity)
        else:
            result = verify_image(
                image=args.image,
                identity=identity,
                lock=args.lock,
                public_deps=args.public_deps,
            )
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except (OSError, ValueError, RuntimeCacheError, json.JSONDecodeError) as exc:
        print(f"RUNTIME_CACHE_STAGE=failed REASON={type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
