#!/usr/bin/env python3
"""Audit and smoke-test the pinned public scientific dependency environment."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys

PIN_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s=]+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_lock(lock_path: Path) -> list[tuple[str, str]]:
    pins: list[tuple[str, str]] = []
    normalized_seen: set[str] = set()
    raw = lock_path.read_text(encoding="utf-8")
    if not raw.endswith("\n"):
        raise SystemExit("lock must end with a newline")
    for line in raw.splitlines():
        match = PIN_RE.fullmatch(line)
        if match is None:
            raise SystemExit("lock contains a non-exact requirement")
        name, version = match.groups()
        normalized = name.lower().replace("_", "-")
        if normalized in normalized_seen:
            raise SystemExit("lock contains a duplicate distribution")
        normalized_seen.add(normalized)
        pins.append((name, version))
    if not pins:
        raise SystemExit("lock is empty")
    return pins


def verify_lock(lock_path: Path, expected_sha256: str, expected_count: int) -> dict[str, object]:
    observed = sha256_file(lock_path)
    pins = parse_lock(lock_path)
    if observed != expected_sha256:
        raise SystemExit("pinned lock SHA256 mismatch")
    if len(pins) != expected_count:
        raise SystemExit("pinned lock package count mismatch")
    return {
        "schema": "public-pinned-lock-check/1",
        "sha256": observed,
        "package_count": len(pins),
    }


def wheel_manifest(wheelhouse: Path) -> dict[str, object]:
    entries = sorted(wheelhouse.iterdir(), key=lambda p: p.name)
    if not entries:
        raise SystemExit("wheelhouse is empty")
    for path in entries:
        if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".whl":
            raise SystemExit("wheelhouse contains a non-wheel entry")
    return {
        "wheel_bytes": sum(path.stat().st_size for path in entries),
        "wheel_count": len(entries),
        "wheel_filenames": [path.name for path in entries],
        "wheel_sha256": {path.name: sha256_file(path) for path in entries},
    }


def verify_wheelhouse(wheelhouse: Path, expected_manifest_path: Path) -> dict[str, object]:
    expected = json.loads(expected_manifest_path.read_text(encoding="utf-8"))
    actual = wheel_manifest(wheelhouse)
    if actual != expected:
        raise SystemExit("restored wheelhouse does not match pinned manifest")
    manifest_sha = sha256_file(expected_manifest_path)
    return {
        "schema": "public-pinned-wheelhouse-check/1",
        "manifest_sha256": manifest_sha,
        "wheel_count": actual["wheel_count"],
        "wheel_bytes": actual["wheel_bytes"],
        "exact_file_set_and_sha256": True,
    }


def verify_installed(lock_path: Path) -> dict[str, object]:
    pins = parse_lock(lock_path)
    for name, expected in pins:
        try:
            observed = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise SystemExit("locked distribution is not installed") from exc
        if observed != expected:
            raise SystemExit("installed distribution version does not match lock")
    return {
        "schema": "public-installed-lock-check/1",
        "verified_packages": len(pins),
    }


def smoke() -> dict[str, object]:
    import numpy as np
    import scipy
    import spglib
    from matminer.featurizers.site import CrystalNNFingerprint
    from pymatgen.core import Lattice, Structure

    fp = CrystalNNFingerprint.from_preset("ops", x_diff_weight=None, distance_cutoffs=None)
    labels = list(fp.feature_labels())
    if len(labels) != 61 or len(set(labels)) != 61:
        raise SystemExit("unexpected public fingerprint roster")
    structure = Structure(Lattice.cubic(3.6), ["Cu"], [[0, 0, 0]])
    values = np.asarray(fp.featurize(structure, 0), dtype=float)
    if values.shape != (61,) or not np.isfinite(values).all():
        raise SystemExit("public synthetic fingerprint smoke failed")
    return {
        "schema": "public-scientific-env-smoke/2",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "packages": {
            name: metadata.version(name)
            for name in ("matminer", "pymatgen", "pymatgen-core", "numpy", "scipy", "spglib")
        },
        "fingerprint_coordinates": int(values.size),
        "finite_coordinates": int(np.isfinite(values).sum()),
        "synthetic_only": True,
    }


def emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify-lock")
    p.add_argument("lock", type=Path)
    p.add_argument("expected_sha256")
    p.add_argument("expected_count", type=int)
    p = sub.add_parser("verify-wheelhouse")
    p.add_argument("wheelhouse", type=Path)
    p.add_argument("expected_manifest", type=Path)
    p = sub.add_parser("verify-installed")
    p.add_argument("lock", type=Path)
    p = sub.add_parser("wheel-manifest")
    p.add_argument("wheelhouse", type=Path)
    sub.add_parser("smoke")
    args = parser.parse_args()
    if args.cmd == "verify-lock":
        emit(verify_lock(args.lock, args.expected_sha256, args.expected_count))
    elif args.cmd == "verify-wheelhouse":
        emit(verify_wheelhouse(args.wheelhouse, args.expected_manifest))
    elif args.cmd == "verify-installed":
        emit(verify_installed(args.lock))
    elif args.cmd == "wheel-manifest":
        emit(wheel_manifest(args.wheelhouse))
    else:
        emit(smoke())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
