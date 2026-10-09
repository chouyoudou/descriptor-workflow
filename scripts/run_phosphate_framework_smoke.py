#!/usr/bin/env python3
"""Deterministic thousand-structure smoke test for the descriptor family."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from phosphate_framework_descriptor import DescriptorConfig, PeriodicStructure, featurize_structure


def _random_lattice(rng: np.random.Generator, lengths: tuple[float, float, float]) -> np.ndarray:
    lattice = np.diag(np.asarray(lengths, dtype=float))
    shear = rng.uniform(-0.08, 0.08, size=(3, 3))
    shear = np.tril(shear, -1)
    return (np.eye(3) + shear) @ lattice


def _orthogonal_frame(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.array([1.0, 0.0, 0.0]) if abs(unit[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(unit, trial)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(unit, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def phosphate_bridge_structure(rng: np.random.Generator, index: int) -> PeriodicStructure:
    length = float(rng.uniform(11.5, 13.5))
    lattice = _random_lattice(rng, (length, length * rng.uniform(0.94, 1.06), length * rng.uniform(0.94, 1.06)))
    center_cart = np.array([0.5, 0.5, 0.5]) @ lattice
    high_symbol = ("La", "Pr", "Sm", "Y")[index % 4]
    low_symbol = ("P", "Si", "V", "S")[index % 4]
    tetra_dirs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    tetra_dirs /= np.linalg.norm(tetra_dirs, axis=1)[:, None]
    other_dirs = -tetra_dirs
    species: list[str] = [high_symbol]
    cart: list[np.ndarray] = [center_cart]
    p_distance = rng.uniform(3.65, 3.95)
    bridge_distance = rng.uniform(2.25, 2.45)
    po = rng.uniform(1.48, 1.62)
    for unit in tetra_dirs:
        p_cart = center_cart + unit * p_distance
        bridge = center_cart + unit * bridge_distance
        species.extend([low_symbol, "O"])
        cart.extend([p_cart, bridge])
        e1, e2 = _orthogonal_frame(unit)
        for k in range(3):
            phi = 2.0 * math.pi * k / 3.0 + rng.normal(0.0, 0.025)
            direction = (unit / 3.0) + math.sqrt(8.0 / 9.0) * (math.cos(phi) * e1 + math.sin(phi) * e2)
            direction /= np.linalg.norm(direction)
            species.append("O")
            cart.append(p_cart + direction * po * rng.uniform(0.97, 1.03))
    for unit in other_dirs:
        species.append("O")
        cart.append(center_cart + unit * rng.uniform(2.25, 2.50))
    frac = np.asarray(cart) @ np.linalg.inv(lattice)
    return PeriodicStructure(lattice, tuple(species), frac, f"synthetic phosphate bridge {index}")


def tetra_structure(rng: np.random.Generator, index: int) -> PeriodicStructure:
    lengths = tuple(float(x) for x in rng.uniform(6.5, 10.0, size=3))
    lattice = _random_lattice(rng, lengths)
    center = rng.uniform(0.25, 0.75, size=3)
    center_cart = center @ lattice
    directions = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    directions = directions @ q
    bond = rng.uniform(1.35, 1.85)
    cart = [center_cart]
    for vector in directions:
        cart.append(center_cart + vector * bond * rng.uniform(0.94, 1.06))
    center_symbol = ("P", "Si", "Ge", "V", "Mo")[index % 5]
    ligand = ("O", "N", "S")[index % 3]
    frac = np.asarray(cart) @ np.linalg.inv(lattice)
    return PeriodicStructure(lattice, (center_symbol,) + (ligand,) * 4, frac, f"tetra {index}")


def octa_structure(rng: np.random.Generator, index: int) -> PeriodicStructure:
    lengths = tuple(float(x) for x in rng.uniform(7.0, 10.5, size=3))
    lattice = _random_lattice(rng, lengths)
    center = rng.uniform(0.25, 0.75, size=3)
    center_cart = center @ lattice
    directions = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    directions = directions @ q
    bond = rng.uniform(1.75, 2.25)
    cart = [center_cart] + [center_cart + v * bond * rng.uniform(0.93, 1.07) for v in directions]
    center_symbol = ("Ti", "Zr", "Hf", "Fe", "Mn")[index % 5]
    ligand = ("O", "F", "N")[index % 3]
    frac = np.asarray(cart) @ np.linalg.inv(lattice)
    return PeriodicStructure(lattice, (center_symbol,) + (ligand,) * 6, frac, f"octa {index}")


def dense_structure(rng: np.random.Generator, index: int) -> PeriodicStructure:
    a = float(rng.uniform(3.6, 6.0))
    lattice = _random_lattice(rng, (a, a * rng.uniform(0.95, 1.05), a * rng.uniform(0.95, 1.05)))
    families = [("Na", "Cl"), ("Mg", "O"), ("Ca", "F"), ("Zn", "S"), ("Li", "O")]
    a_symbol, b_symbol = families[index % len(families)]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=float,
    )
    species = (a_symbol, a_symbol, a_symbol, a_symbol, b_symbol, b_symbol, b_symbol, b_symbol)
    coords = np.mod(coords + rng.normal(0.0, 0.004, size=coords.shape), 1.0)
    return PeriodicStructure(lattice, species, coords, f"dense binary {index}")


def open_channel_structure(rng: np.random.Generator, index: int) -> PeriodicStructure:
    lengths = (float(rng.uniform(5.0, 8.0)), float(rng.uniform(9.0, 16.0)), float(rng.uniform(9.0, 16.0)))
    lattice = _random_lattice(rng, lengths)
    symbols = ("C", "Si", "Al", "Xe", "La")
    n = 1 + index % 4
    species = tuple(symbols[(index + k) % len(symbols)] for k in range(n))
    coords = np.zeros((n, 3), dtype=float)
    for k in range(n):
        coords[k] = [((k + 0.5) / n) % 1.0, 0.15 + 0.07 * (k % 2), 0.15 + 0.06 * ((k // 2) % 2)]
    coords = np.mod(coords + rng.normal(0.0, 0.005, size=coords.shape), 1.0)
    return PeriodicStructure(lattice, species, coords, f"open channel {index}")


GENERATORS = (
    ("phosphate_bridge", phosphate_bridge_structure),
    ("tetrahedral", tetra_structure),
    ("octahedral", octa_structure),
    ("dense_binary", dense_structure),
    ("open_channel", open_channel_structure),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20250115)
    parser.add_argument("--output-dir", default="artifacts/paper-5a4d4d2640ecebb85628abb0")
    parser.add_argument("--grid-spacing", type=float, default=2.5)
    parser.add_argument("--max-grid-points", type=int, default=512)
    args = parser.parse_args()
    if args.n < 1:
        raise SystemExit("--n must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    config = DescriptorConfig(grid_spacing=args.grid_spacing, max_grid_points=args.max_grid_points)
    started = time.perf_counter()
    rows = []
    family_counts: dict[str, int] = {}
    max_permutation_error = 0.0
    max_translation_error = 0.0
    feature_names: tuple[str, ...] | None = None

    for i in range(args.n):
        family, generator = GENERATORS[i % len(GENERATORS)]
        structure = generator(rng, i)
        result = featurize_structure(structure, config)
        values = np.asarray(result.values, dtype=float)
        if feature_names is None:
            feature_names = result.feature_names
        if result.feature_names != feature_names or not np.all(np.isfinite(values)):
            raise RuntimeError(f"invalid fixed vector at case {i}")
        family_counts[family] = family_counts.get(family, 0) + 1
        rows.append({"case_id": i, "family": family, "values": values.tolist()})

        if i % 25 == 0:
            order = rng.permutation(structure.n_sites)
            permuted = featurize_structure(structure.reordered(order), config)
            translated = featurize_structure(structure.translated(rng.uniform(-2.0, 2.0, size=3)), config)
            max_permutation_error = max(max_permutation_error, float(np.max(np.abs(values - permuted.values))))
            max_translation_error = max(max_translation_error, float(np.max(np.abs(values - translated.values))))

    matrix = np.asarray([row["values"] for row in rows], dtype=float)
    runtime = time.perf_counter() - started
    assert feature_names is not None
    summary = {
        "schema": "paper-descriptor-smoke/1",
        "paper_id": "paper-5a4d4d2640ecebb85628abb0",
        "doi": "10.3390/molecules30020331",
        "n_structures": args.n,
        "n_features": len(feature_names),
        "seed": args.seed,
        "family_counts": family_counts,
        "runtime_seconds": runtime,
        "structures_per_second": args.n / runtime,
        "finite_fraction": float(np.mean(np.isfinite(matrix))),
        "max_permutation_error": max_permutation_error,
        "max_translation_error": max_translation_error,
        "feature_min": dict(zip(feature_names, np.min(matrix, axis=0).tolist())),
        "feature_max": dict(zip(feature_names, np.max(matrix, axis=0).tolist())),
        "feature_mean": dict(zip(feature_names, np.mean(matrix, axis=0).tolist())),
        "feature_std": dict(zip(feature_names, np.std(matrix, axis=0).tolist())),
        "config": {
            "grid_spacing": args.grid_spacing,
            "max_grid_points": args.max_grid_points,
            "bond_scale": config.bond_scale,
        },
        "scope_note": (
            "Synthetic structural smoke coverage only. It is not a catalytic benchmark and does not validate "
            "macroscopic porosity, defects, activity, durability, or reaction history."
        ),
    }
    if max_permutation_error > 1e-8 or max_translation_error > 1e-8:
        raise RuntimeError(
            f"invariance failure: permutation={max_permutation_error:g}, translation={max_translation_error:g}"
        )

    (output_dir / "feature_names.json").write_text(json.dumps(list(feature_names), indent=2) + "\n", encoding="utf-8")
    (output_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "features.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "n_structures", "n_features", "runtime_seconds", "structures_per_second",
        "finite_fraction", "max_permutation_error", "max_translation_error"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
