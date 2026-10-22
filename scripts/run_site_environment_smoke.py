#!/usr/bin/env python3
"""Deterministic 1,000-structure target-blind smoke for the descriptor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from site_environment_descriptor import (
    FEATURE_NAMES,
    DescriptorConfig,
    PeriodicStructure,
    featurize_structure,
)
from site_environment_descriptor.elements import get_element


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def _slightly_skewed_lattice(
    rng: np.random.Generator,
    lengths: tuple[float, float, float],
) -> np.ndarray:
    lattice = np.diag(np.asarray(lengths, dtype=float))
    shear = np.eye(3)
    shear[1, 0] = rng.uniform(-0.06, 0.06)
    shear[2, 0] = rng.uniform(-0.06, 0.06)
    shear[2, 1] = rng.uniform(-0.06, 0.06)
    return shear @ lattice


def _single_symbol(index: int) -> str:
    return ("C", "Si", "Ge", "Al", "S")[index % 5]


def cubic_single(rng: np.random.Generator, index: int) -> PeriodicStructure:
    symbol = _single_symbol(index)
    radius = get_element(symbol).covalent_radius_angstrom
    a = 2.0 * radius * rng.uniform(0.98, 1.16)
    return PeriodicStructure(
        _slightly_skewed_lattice(
            rng,
            (a, a * rng.uniform(0.99, 1.01), a * rng.uniform(0.99, 1.01)),
        ),
        (symbol,),
        np.array([[0.0, 0.0, 0.0]]),
        f"cubic-single-{index}",
    )


def chain_single(rng: np.random.Generator, index: int) -> PeriodicStructure:
    symbol = _single_symbol(index)
    radius = get_element(symbol).covalent_radius_angstrom
    a = 2.0 * radius * rng.uniform(0.98, 1.16)
    return PeriodicStructure(
        _slightly_skewed_lattice(rng, (a, rng.uniform(7.0, 10.0), rng.uniform(7.0, 10.0))),
        (symbol,),
        np.array([[rng.uniform(0.0, 1.0), 0.25, 0.75]]),
        f"chain-single-{index}",
    )


def layer_single(rng: np.random.Generator, index: int) -> PeriodicStructure:
    symbol = _single_symbol(index)
    radius = get_element(symbol).covalent_radius_angstrom
    a = 2.0 * radius * rng.uniform(0.98, 1.14)
    b = 2.0 * radius * rng.uniform(0.98, 1.14)
    return PeriodicStructure(
        _slightly_skewed_lattice(rng, (a, b, rng.uniform(8.0, 12.0))),
        (symbol,),
        np.array([[0.37, 0.61, 0.22]]),
        f"layer-single-{index}",
    )


def binary_bcc(rng: np.random.Generator, index: int) -> PeriodicStructure:
    pairs = (("Na", "Cl"), ("Mg", "O"), ("Ca", "F"), ("La", "O"), ("Ti", "N"))
    first, second = pairs[index % len(pairs)]
    radius_sum = (
        get_element(first).covalent_radius_angstrom
        + get_element(second).covalent_radius_angstrom
    )
    nearest_s = rng.uniform(0.98, 1.18)
    a = 2.0 * radius_sum * nearest_s / math.sqrt(3.0)
    lattice = _slightly_skewed_lattice(
        rng,
        (a, a * rng.uniform(0.99, 1.01), a * rng.uniform(0.99, 1.01)),
    )
    coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=float)
    coords[1] += rng.normal(0.0, 0.004, size=3)
    return PeriodicStructure(lattice, (first, second), coords, f"binary-bcc-{index}")


def distorted_cage(rng: np.random.Generator, index: int) -> PeriodicStructure:
    choices = (
        ("La", "O", 6),
        ("Ca", "F", 8),
        ("Ti", "O", 6),
        ("P", "O", 4),
        ("Si", "N", 4),
    )
    center_symbol, ligand_symbol, coordination = choices[index % len(choices)]
    box = rng.uniform(10.5, 13.0)
    lattice = _slightly_skewed_lattice(
        rng,
        (box, box * rng.uniform(0.97, 1.03), box * rng.uniform(0.97, 1.03)),
    )
    center_frac = np.array([0.5, 0.5, 0.5])
    center_cart = center_frac @ lattice
    if coordination == 4:
        directions = np.array(
            [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
            dtype=float,
        )
    elif coordination == 6:
        directions = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=float,
        )
    else:
        directions = np.array(
            [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
            dtype=float,
        )
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    directions = directions @ _random_rotation(rng)
    radius_sum = (
        get_element(center_symbol).covalent_radius_angstrom
        + get_element(ligand_symbol).covalent_radius_angstrom
    )
    cart = [center_cart]
    for direction in directions:
        distance = radius_sum * rng.uniform(0.96, 1.17)
        cart.append(center_cart + direction * distance)
    frac = np.asarray(cart) @ np.linalg.inv(lattice)
    species = (center_symbol,) + (ligand_symbol,) * coordination
    return PeriodicStructure(lattice, species, frac, f"distorted-cage-{index}")


GENERATORS: tuple[
    tuple[str, Callable[[np.random.Generator, int], PeriodicStructure]], ...
] = (
    ("cubic_single", cubic_single),
    ("chain_single", chain_single),
    ("layer_single", layer_single),
    ("binary_bcc", binary_bcc),
    ("distorted_cage", distorted_cage),
)


def _compare_results(left, right) -> tuple[float, int]:
    max_error = 0.0
    missing_mismatch = 0
    for a, b in zip(left.values, right.values):
        if a is None or b is None:
            missing_mismatch += int((a is None) != (b is None))
        else:
            max_error = max(max_error, abs(float(a) - float(b)))
    return max_error, missing_mismatch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument(
        "--output-dir",
        default="artifacts/paper-7a2c33e715c47a2c191a0574",
    )
    args = parser.parse_args()
    if args.n < 1:
        raise SystemExit("--n must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    config = DescriptorConfig()
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    family_counts: Counter[str] = Counter()
    null_counts = np.zeros(len(FEATURE_NAMES), dtype=int)
    numeric_columns: list[list[float]] = [[] for _ in FEATURE_NAMES]
    max_permutation_error = 0.0
    max_translation_error = 0.0
    max_rotation_error = 0.0
    max_supercell_error = 0.0
    missingness_mismatches = 0

    for case_id in range(args.n):
        family, generator = GENERATORS[case_id % len(GENERATORS)]
        structure = generator(rng, case_id)
        result = featurize_structure(structure, config)
        if result.feature_names != FEATURE_NAMES:
            raise RuntimeError(f"feature contract changed at case {case_id}")
        family_counts[family] += 1
        for column, value in enumerate(result.values):
            if value is None:
                null_counts[column] += 1
            else:
                if not np.isfinite(float(value)):
                    raise RuntimeError(f"non-finite value at case {case_id}, column {column}")
                numeric_columns[column].append(float(value))
        rows.append(
            {
                "case_id": case_id,
                "family": family,
                "n_sites": structure.n_sites,
                "values": list(result.values),
                "n_valid_environments": result.metadata["n_valid_environments"],
            }
        )

        if case_id % 20 == 0:
            permuted = featurize_structure(
                structure.reordered(rng.permutation(structure.n_sites)), config
            )
            error, mismatch = _compare_results(result, permuted)
            max_permutation_error = max(max_permutation_error, error)
            missingness_mismatches += mismatch
            translated = featurize_structure(
                structure.translated(rng.uniform(-3.0, 3.0, size=3)), config
            )
            error, mismatch = _compare_results(result, translated)
            max_translation_error = max(max_translation_error, error)
            missingness_mismatches += mismatch

        if case_id % 40 == 0:
            rotated = featurize_structure(
                structure.rotated(_random_rotation(rng)), config
            )
            error, mismatch = _compare_results(result, rotated)
            max_rotation_error = max(max_rotation_error, error)
            missingness_mismatches += mismatch

        if case_id % 100 == 0:
            expanded = featurize_structure(structure.supercell((2, 1, 1)), config)
            error, mismatch = _compare_results(result, expanded)
            max_supercell_error = max(max_supercell_error, error)
            missingness_mismatches += mismatch

    runtime = time.perf_counter() - started
    canonical_rows = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for row in rows
    ).encode("utf-8")
    column_summary: dict[str, dict[str, float | int | None]] = {}
    for name, null_count, data in zip(FEATURE_NAMES, null_counts, numeric_columns):
        arr = np.asarray(data, dtype=float)
        column_summary[name] = {
            "defined_count": int(arr.size),
            "null_count": int(null_count),
            "min": float(np.min(arr)) if arr.size else None,
            "max": float(np.max(arr)) if arr.size else None,
            "mean": float(np.mean(arr)) if arr.size else None,
            "std": float(np.std(arr)) if arr.size else None,
        }

    max_error = max(
        max_permutation_error,
        max_translation_error,
        max_rotation_error,
        max_supercell_error,
    )
    if missingness_mismatches:
        raise RuntimeError(f"representation changed typed missingness {missingness_mismatches} times")
    if max_error > 2.0e-9:
        raise RuntimeError(f"representation invariance error {max_error:.6g}")

    summary = {
        "schema": "paper-descriptor-smoke/1",
        "paper_id": "paper-7a2c33e715c47a2c191a0574",
        "doi": "10.1039/d2ra06962h",
        "source_commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "n_structures": args.n,
        "n_features": len(FEATURE_NAMES),
        "seed": args.seed,
        "family_counts": dict(sorted(family_counts.items())),
        "runtime_seconds": runtime,
        "structures_per_second": args.n / runtime,
        "defined_fraction": float(
            (args.n * len(FEATURE_NAMES) - int(np.sum(null_counts)))
            / (args.n * len(FEATURE_NAMES))
        ),
        "columns_with_any_defined_value": int(sum(bool(column) for column in numeric_columns)),
        "columns_with_any_null": int(np.sum(null_counts > 0)),
        "max_permutation_error": max_permutation_error,
        "max_translation_error": max_translation_error,
        "max_rotation_error": max_rotation_error,
        "max_supercell_error": max_supercell_error,
        "missingness_mismatches": missingness_mismatches,
        "rows_sha256": hashlib.sha256(canonical_rows).hexdigest(),
        "column_summary": column_summary,
        "config": {
            "normalized_inner": config.normalized_inner,
            "normalized_outer": config.normalized_outer,
            "max_image_range": config.max_image_range,
            "max_neighbor_evaluations": config.max_neighbor_evaluations,
        },
        "scope_note": (
            "Target-blind synthetic structure verification only. No oxidation state, "
            "dopant site/occupancy, optical or thermal property, energy, or synthesis "
            "history is used or validated."
        ),
    }

    (output_dir / "feature_names.json").write_text(
        json.dumps(list(FEATURE_NAMES), indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "features.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "n_structures",
                    "n_features",
                    "runtime_seconds",
                    "structures_per_second",
                    "defined_fraction",
                    "columns_with_any_defined_value",
                    "columns_with_any_null",
                    "max_permutation_error",
                    "max_translation_error",
                    "max_rotation_error",
                    "max_supercell_error",
                    "rows_sha256",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
