#!/usr/bin/env python3
"""Deterministic 1000-case synthetic validation for layer_bridge_geometry.

This is a geometry regression set, not a materials-property benchmark.  It uses
one-atom periodic carbon fixtures whose expected connected-component rank is
known by construction (250 each of 0D, 1D, 2D, and 3D).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_bridge_geometry import DESCRIPTOR_NAMES, Structure, analyze_structure


SCHEMA = "layer-bridge-frozen-validation/1"
EXPECTED_CASES = 1000
# Filled from the checked-in implementation.  Rounding to 12 decimal places
# makes this a deterministic algorithmic regression rather than a platform
# timing test.
EXPECTED_DESCRIPTOR_DIGEST = "a43948921b31cdaa17fee170d98597ba536d9018091fd30c75d9c31440fb4d46"


def _matmul_row(row, matrix):
    return tuple(sum(row[k] * matrix[k][j] for k in range(3)) for j in range(3))


def _rotation(index: int):
    angle_z = math.radians((17 * index) % 360)
    angle_x = math.radians((29 * index) % 360)
    cz, sz = math.cos(angle_z), math.sin(angle_z)
    cx, sx = math.cos(angle_x), math.sin(angle_x)
    rz = ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0))
    rx = ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx))
    return tuple(_matmul_row(row, rx) for row in rz)


def synthetic_case(index: int) -> tuple[int, Structure]:
    if not 0 <= index < EXPECTED_CASES:
        raise ValueError("synthetic-case index must be in [0, 1000)")
    expected_dimension = index // 250
    local = index % 250

    small = [1.30 + 0.007 * ((local * (axis + 3) + 5 * axis) % 23) for axis in range(3)]
    large = [6.20 + 0.031 * ((local * (axis + 5) + 11 * axis) % 41) for axis in range(3)]
    lengths = [small[axis] if axis < expected_dimension else large[axis] for axis in range(3)]

    permutations = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2))
    order = permutations[local % len(permutations)]
    diagonal = [
        tuple(lengths[order[row]] if row == column else 0.0 for column in range(3))
        for row in range(3)
    ]
    rotation = _rotation(local % 19)
    lattice = tuple(_matmul_row(row, rotation) for row in diagonal)
    frac = (
        ((37 * local + 3) % 251) / 251.0,
        ((53 * local + 7) % 257) / 257.0,
        ((71 * local + 11) % 263) / 263.0,
    )
    return expected_dimension, Structure(lattice, ("C",), (frac,), f"frozen-{index:04d}")


def run_validation() -> dict[str, object]:
    started = time.perf_counter()
    expected_counts = {str(dimension): 0 for dimension in range(4)}
    observed_counts = {str(dimension): 0 for dimension in range(4)}
    exact_matches = 0
    finite_values = 0
    layered_positive_clearance = 0
    rows: list[dict[str, object]] = []

    for index in range(EXPECTED_CASES):
        expected, structure = synthetic_case(index)
        result = analyze_structure(structure)
        observed = int(round(float(result["descriptors"]["framework_dimensionality_max"])))
        expected_counts[str(expected)] += 1
        observed_counts[str(observed)] += 1
        exact_matches += int(expected == observed)
        if expected == 2 and float(result["descriptors"]["interlayer_clearance_angstrom"]) > 0:
            layered_positive_clearance += 1
        rounded = []
        for name in DESCRIPTOR_NAMES:
            value = float(result["descriptors"][name])
            if not math.isfinite(value):
                raise AssertionError(f"non-finite descriptor {name} in case {index}")
            finite_values += 1
            rounded.append(round(value, 12))
        rows.append({"index": index, "expected": expected, "observed": observed, "values": rounded})

    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    if exact_matches != EXPECTED_CASES:
        raise AssertionError(f"only {exact_matches}/{EXPECTED_CASES} dimensionality matches")
    if layered_positive_clearance != 250:
        raise AssertionError("all 250 synthetic 2D fixtures must have positive clearance")
    if EXPECTED_DESCRIPTOR_DIGEST != "TO_BE_FILLED" and digest != EXPECTED_DESCRIPTOR_DIGEST:
        raise AssertionError(
            f"descriptor digest changed: expected {EXPECTED_DESCRIPTOR_DIGEST}, observed {digest}"
        )

    return {
        "schema": SCHEMA,
        "scope": "synthetic static geometry; no property labels",
        "case_count": EXPECTED_CASES,
        "descriptor_count_per_case": len(DESCRIPTOR_NAMES),
        "finite_descriptor_values": finite_values,
        "expected_dimensionality_counts": expected_counts,
        "observed_dimensionality_counts": observed_counts,
        "exact_dimensionality_matches": exact_matches,
        "layered_positive_clearance_cases": layered_positive_clearance,
        "descriptor_sha256": digest,
        "expected_descriptor_sha256": EXPECTED_DESCRIPTOR_DIGEST,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_validation()
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
