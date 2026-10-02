#!/usr/bin/env python3
"""Deterministic 1000-structure qualification for PHGG.

The generator is target-blind: it perturbs analytic families with known
connectivity rank and guest presence, runs the public ``compute`` path, checks
the fixed schema and finiteness, and writes one CSV plus stable JSON evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from descriptors.periodic_host_guest_geometry import (
    DESCRIPTOR_VERSION,
    FEATURE_ID,
    FEATURE_NAMES,
    FEATURE_UNITS,
    compute,
)


def _poscar(
    title: str,
    lattice: list[tuple[float, float, float]],
    species: list[str],
    counts: list[int],
    coords: list[tuple[float, float, float]],
) -> str:
    return "\n".join(
        [
            title,
            "1.0",
            *(" ".join(f"{value:.12g}" for value in vector) for vector in lattice),
            " ".join(species),
            " ".join(str(value) for value in counts),
            "Direct",
            *(" ".join(f"{value % 1.0:.12g}" for value in coord) for coord in coords),
        ]
    ) + "\n"


def _translated(coords: list[tuple[float, float, float]], shift: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    return [
        ((x + shift[0]) % 1.0, (y + shift[1]) % 1.0, (z + shift[2]) % 1.0)
        for x, y, z in coords
    ]


def generate_case(index: int, rng: random.Random) -> tuple[str, dict[str, Any]]:
    family = index % 5
    shift = (rng.random(), rng.random(), rng.random())
    if family == 0:
        a = 1.66 + 0.16 * rng.random()
        b = 8.0 + 4.0 * rng.random()
        c = 8.0 + 4.0 * rng.random()
        shear = 0.08 * (rng.random() - 0.5)
        lattice = [(a, 0.0, 0.0), (shear, b, 0.0), (0.0, 0.0, c)]
        coords = _translated([(0.0, 0.0, 0.0), (0.5, 0.47, 0.53)], shift)
        text = _poscar(f"phgg-{index}-1d-guest", lattice, ["C", "He"], [1, 1], coords)
        expected = {"family": "1d_host_single_guest", "dimension": 1, "guest": True, "n_atoms": 2}
    elif family == 1:
        a = 1.64 + 0.10 * rng.random()
        b = 1.64 + 0.10 * rng.random()
        c = 8.0 + 4.0 * rng.random()
        shear = 0.05 * (rng.random() - 0.5)
        lattice = [(a, 0.0, 0.0), (shear, b, 0.0), (0.0, 0.0, c)]
        dz = (0.65 + 0.12 * rng.random()) / c
        coords = _translated(
            [(0.0, 0.0, 0.05), (0.5, 0.5, 0.55 - dz / 2.0), (0.5, 0.5, 0.55 + dz / 2.0)],
            shift,
        )
        text = _poscar(f"phgg-{index}-2d-dimer", lattice, ["C", "H"], [1, 2], coords)
        expected = {"family": "2d_host_dimer_guest", "dimension": 2, "guest": True, "n_atoms": 3}
    elif family == 2:
        a = 2.90 + 0.18 * rng.random()
        b = 2.90 + 0.18 * rng.random()
        c = 2.90 + 0.18 * rng.random()
        shear = 0.06 * (rng.random() - 0.5)
        lattice = [(a, 0.0, 0.0), (shear, b, 0.0), (0.0, shear, c)]
        coords = _translated([(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)], shift)
        text = _poscar(f"phgg-{index}-3d-guest", lattice, ["Cu", "He"], [1, 1], coords)
        expected = {"family": "3d_host_single_guest", "dimension": 3, "guest": True, "n_atoms": 2}
    elif family == 3:
        a = 2.90 + 0.18 * rng.random()
        b = 2.90 + 0.18 * rng.random()
        c = 2.90 + 0.18 * rng.random()
        lattice = [(a, 0.0, 0.0), (0.03, b, 0.0), (0.02, 0.04, c)]
        coords = _translated([(0.0, 0.0, 0.0)], shift)
        text = _poscar(f"phgg-{index}-3d-empty", lattice, ["Cu"], [1], coords)
        expected = {"family": "3d_host_no_guest", "dimension": 3, "guest": False, "n_atoms": 1}
    else:
        a = 10.0 + 3.0 * rng.random()
        b = 10.0 + 3.0 * rng.random()
        c = 10.0 + 3.0 * rng.random()
        lattice = [(a, 0.0, 0.0), (0.1, b, 0.0), (0.0, 0.1, c)]
        separation = 1.35 + 0.08 * rng.random()
        dx = separation / a
        coords = _translated([(0.45 - dx / 2, 0.5, 0.5), (0.45 + dx / 2, 0.5, 0.5), (0.82, 0.80, 0.78)], shift)
        text = _poscar(f"phgg-{index}-0d-fallback", lattice, ["C", "He"], [2, 1], coords)
        expected = {"family": "finite_host_fallback_guest", "dimension": 0, "guest": True, "n_atoms": 3}
    return text, expected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_evidence(root: Path, output: Path) -> list[str]:
    paths = [
        "descriptors/__init__.py",
        "descriptors/periodic_host_guest_geometry.py",
        "descriptors/phgg_graph.py",
        "descriptors/phgg_types.py",
        "descriptors/phgg_radii.py",
        "descriptors/periodic_host_guest_geometry.LICENSE",
        "descriptors/README.md",
        "tests/features/__init__.py",
        "tests/features/test_periodic_host_guest_geometry.py",
        "scripts/qualify_periodic_host_guest_geometry.py",
        ".github/workflows/feature-periodic-host-guest-geometry.yml",
        "docs/periodic_host_guest_geometry_provenance.md",
        "examples/POSCAR_phgg_host_guest",
    ]
    copied: list[str] = []
    for relative in paths:
        source = root / relative
        destination = output / "evidence" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(str(destination.relative_to(output)))
    return copied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=721234)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1000:
        raise SystemExit("qualification requires at least 1000 structures")
    root = REPOSITORY_ROOT
    output = args.out.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    rng = random.Random(args.seed)
    params = {
        "max_grid_points": 64,
        "grid_spacing_A": 2.0,
    }
    rows: list[dict[str, Any]] = []
    runtimes: list[float] = []
    failures: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    start_total = time.perf_counter()
    for index in range(args.samples):
        poscar, expected = generate_case(index, rng)
        family_counts[expected["family"]] = family_counts.get(expected["family"], 0) + 1
        started = time.perf_counter()
        error = ""
        failure_code = ""
        valid = False
        features = {name: 0.0 for name in FEATURE_NAMES}
        try:
            result = compute(poscar, params)
            runtime_ms = (time.perf_counter() - started) * 1000.0
            features = result.features
            valid = (
                tuple(features) == FEATURE_NAMES
                and all(math.isfinite(value) for value in features.values())
                and int(features["host_network_dimensionality"]) == expected["dimension"]
                and bool(features["host_guest_present_ratio"]) is expected["guest"]
                and result.metadata["n_atoms"] == expected["n_atoms"]
            )
            if not valid:
                failure_code = "semantic_expectation_failed"
                error = failure_code
        except Exception as exc:  # evidence must record, then fail the run below
            runtime_ms = (time.perf_counter() - started) * 1000.0
            failure_code = "compute_exception"
            error = f"{type(exc).__name__}: {exc}"
        runtimes.append(runtime_ms)
        row: dict[str, Any] = {
            "sample_id": f"phgg-{index:04d}",
            "family": expected["family"],
            "n_atoms": expected["n_atoms"],
            "value": features["host_largest_grid_included_diameter_A"],
            "is_finite": all(math.isfinite(value) for value in features.values()),
            "valid": valid,
            "runtime_ms": runtime_ms,
            "error": error,
            "failure_code": failure_code,
            **features,
        }
        rows.append(row)
        if not valid:
            failures.append({"sample_id": row["sample_id"], "family": expected["family"], "failure_code": failure_code, "error": error})
    total_seconds = time.perf_counter() - start_total

    csv_path = output / "samples.csv"
    columns = [
        "sample_id", "family", "n_atoms", "value", "is_finite", "valid", "runtime_ms", "error", "failure_code", *FEATURE_NAMES
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema": "periodic-host-guest-geometry-qualification-summary/1",
        "feature_id": FEATURE_ID,
        "descriptor_version": DESCRIPTOR_VERSION,
        "sample_target": args.samples,
        "sample_success": args.samples - len(failures),
        "sample_fail": len(failures),
        "seed": args.seed,
        "family_counts": family_counts,
        "total_runtime_seconds": total_seconds,
        "average_runtime_ms": statistics.fmean(runtimes),
        "median_runtime_ms": statistics.median(runtimes),
        "max_runtime_ms": max(runtimes),
        "structures_per_second": args.samples / total_seconds if total_seconds else 0.0,
        "failures": failures[:20],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    evidence_paths = _copy_evidence(root, output)
    readme = output / "README.txt"
    readme.write_text(
        "PHGG deterministic synthetic qualification artifact.\n"
        "samples.csv is the main table; summary.json and manifest.json are machine-readable evidence.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "descriptor-qualification-manifest/1",
        "feature_id": FEATURE_ID,
        "research_id": "paper-721fa2a1b8b1b60dbf4da234",
        "descriptor_version": DESCRIPTOR_VERSION,
        "commit_sha": os.environ.get("GITHUB_SHA", "local-development-only"),
        "seed": args.seed,
        "config": params,
        "sample_target": args.samples,
        "sample_success": args.samples - len(failures),
        "sample_fail": len(failures),
        "fixed_columns": list(FEATURE_NAMES),
        "units": FEATURE_UNITS,
        "source_method": {
            "connectivity": "periodic quotient-graph cycle-translation rank",
            "pore": "host-only deterministic grid-clearance treatment",
            "contact": "radius-normalized nearest host/guest surface geometry",
        },
        "license": "MIT for PHGG implementation; method/data attribution in provenance",
        "artifact_paths": {
            "main_csv": "samples.csv",
            "summary_json": "summary.json",
            "manifest_json": "manifest.json",
            "readme": "README.txt",
            "evidence": evidence_paths,
        },
        "checksums": {
            "samples.csv": _sha256(csv_path),
            "summary.json": _sha256(summary_path),
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"qualification failed for {len(failures)} structures")
    if len(rows) < 1000:
        raise SystemExit("qualification produced fewer than 1000 rows")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
