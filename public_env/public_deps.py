#!/usr/bin/env python3
"""Resolve, audit, and smoke-test the public scientific dependency environment."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

PUBLIC_PACKAGE_HOSTS = {"files.pythonhosted.org", "pypi.org"}


def resolve_lock(report_path: Path, lock_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pins = {}
    for item in report.get("install", []):
        info = item.get("download_info") or {}
        url = str(info.get("url") or "")
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in PUBLIC_PACKAGE_HOSTS:
            raise SystemExit("resolver returned a non-PyPI package source")
        meta = item.get("metadata") or {}
        name, version = str(meta.get("name") or ""), str(meta.get("version") or "")
        if not name or not version or "\n" in name or "\n" in version:
            raise SystemExit("resolver returned invalid package metadata")
        normalized = name.lower().replace("_", "-")
        old = pins.setdefault(normalized, (name, version))
        if old[1] != version:
            raise SystemExit("resolver returned conflicting package versions")
    if not pins:
        raise SystemExit("resolver returned an empty lock")
    text = "".join(f"{name}=={version}\n" for _, (name, version) in sorted(pins.items()))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(text, encoding="utf-8")


def lock_sha(lock_path: Path) -> str:
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()


def wheel_manifest(wheelhouse: Path) -> dict[str, object]:
    files = sorted(p for p in wheelhouse.iterdir() if p.is_file())
    if not files:
        raise SystemExit("wheelhouse is empty")
    return {
        "wheel_count": len(files),
        "wheel_bytes": sum(p.stat().st_size for p in files),
        "wheel_filenames": [p.name for p in files],
        "wheel_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }


def smoke() -> None:
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
    payload = {
        "schema": "public-scientific-env-smoke/1",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "packages": {
            name: metadata.version(name)
            for name in ("matminer", "pymatgen", "pymatgen-core", "numpy", "scipy", "spglib")
        },
        "fingerprint_coordinates": int(values.size),
        "finite_coordinates": int(np.isfinite(values).sum()),
        "synthetic_only": True,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("resolve-lock")
    p.add_argument("report", type=Path)
    p.add_argument("lock", type=Path)
    p = sub.add_parser("lock-sha")
    p.add_argument("lock", type=Path)
    p = sub.add_parser("wheel-manifest")
    p.add_argument("wheelhouse", type=Path)
    sub.add_parser("smoke")
    args = parser.parse_args()
    if args.cmd == "resolve-lock":
        resolve_lock(args.report, args.lock)
    elif args.cmd == "lock-sha":
        print(lock_sha(args.lock))
    elif args.cmd == "wheel-manifest":
        print(json.dumps(wheel_manifest(args.wheelhouse), sort_keys=True, separators=(",", ":")))
    else:
        smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
