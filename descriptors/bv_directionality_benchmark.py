"""Deterministic synthetic qualification for directional response."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import pathlib
import random
from typing import Any, Final

try:
    from .bv_directionality_model import (
        BENCHMARK_SCHEMA, COVALENT_RADII_ANGSTROM, INTERPRETATION_SCOPE, SCHEMA,
        ContactConfig, DescriptorError, Mat3, Structure, Vec3, _cart_to_frac,
        _cross, _dot, _frac_to_cart, _mean, _pearson, _percentile, _unit,
        _vadd, _vscale, _wrap_frac,
    )
    from .bv_directionality_core import analyze_structure
except ImportError:  # direct script sibling import
    from bv_directionality_model import (
        BENCHMARK_SCHEMA, COVALENT_RADII_ANGSTROM, INTERPRETATION_SCOPE, SCHEMA,
        ContactConfig, DescriptorError, Mat3, Structure, Vec3, _cart_to_frac,
        _cross, _dot, _frac_to_cart, _mean, _pearson, _percentile, _unit,
        _vadd, _vscale, _wrap_frac,
    )
    from bv_directionality_core import analyze_structure

_FAMILY_DIRECTIONS: Final[dict[str, tuple[Vec3, ...]]] = {
    "octahedral": (
        (1, 0, 0), (-1, 0, 0), (0, 1, 0),
        (0, -1, 0), (0, 0, 1), (0, 0, -1),
    ),
    "tetrahedral": (
        (1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1),
    ),
    "cubic": tuple(
        (float(x), float(y), float(z))
        for x in (-1, 1)
        for y in (-1, 1)
        for z in (-1, 1)
    ),
    "hexagonal_planar": tuple(
        (
            math.cos(index * math.pi / 3.0),
            math.sin(index * math.pi / 3.0),
            0.0,
        )
        for index in range(6)
    ),
    "square_planar": (
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
    ),
}


def _rodrigues(vector: Vec3, axis: Vec3, angle: float) -> Vec3:
    axis = _unit(axis)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return _vadd(
        _vadd(_vscale(vector, cosine), _vscale(_cross(axis, vector), sine)),
        _vscale(axis, _dot(axis, vector) * (1.0 - cosine)),
    )


def synthetic_structure(
    family: str,
    severity: float,
    variant: int,
    rng: random.Random,
) -> Structure:
    if family not in _FAMILY_DIRECTIONS:
        raise DescriptorError(f"unknown synthetic family {family!r}")
    if not 0 <= severity <= 1:
        raise DescriptorError("severity must be in [0, 1]")
    center_options = ("Sn", "Pb", "Bi", "Sb", "Ca", "Na", "La", "Zr")
    ligand_options = ("F", "O", "S", "Cl", "Se")
    center = center_options[(variant + len(family)) % len(center_options)]
    ligand = ligand_options[(variant * 3 + len(family)) % len(ligand_options)]
    radius_sum = (
        COVALENT_RADII_ANGSTROM[center]
        + COVALENT_RADII_ANGSTROM[ligand]
    )
    base_distance = radius_sum * (1.02 + 0.08 * rng.random())
    raw_directions = _FAMILY_DIRECTIONS[family]
    axis = _unit(
        (0.4 + rng.random(), 0.5 + rng.random(), 0.6 + rng.random())
    )
    angle = 2.0 * math.pi * rng.random()
    directions = [
        _unit(_rodrigues(_unit(direction), axis, angle))
        for direction in raw_directions
    ]

    cell = 14.0 + 1.5 * rng.random()
    lattice: Mat3 = (
        (cell, 0.0, 0.0),
        (0.25 * rng.random(), cell + 0.3, 0.0),
        (0.20 * rng.random(), 0.15 * rng.random(), cell + 0.6),
    )
    center_fractional: Vec3 = (0.5, 0.5, 0.5)
    center_cartesian = _frac_to_cart(center_fractional, lattice)
    coordinates = [center_fractional]
    distorted_direction = directions[0]
    center_shift = _vscale(distorted_direction, 0.18 * severity)
    shifted_center_cartesian = _vadd(center_cartesian, center_shift)
    coordinates[0] = _wrap_frac(
        _cart_to_frac(shifted_center_cartesian, lattice)
    )
    for index, direction in enumerate(directions):
        distance = base_distance * (
            1.0 + (0.70 * severity if index == 0 else 0.0)
        )
        jitter = 1.0 + 0.01 * (rng.random() - 0.5)
        ligand_cartesian = _vadd(
            center_cartesian, _vscale(direction, distance * jitter)
        )
        coordinates.append(
            _wrap_frac(_cart_to_frac(ligand_cartesian, lattice))
        )
    return Structure(
        lattice=lattice,
        species=(center,) + (ligand,) * len(directions),
        frac_coords=tuple(coordinates),
        comment=f"synthetic-{family}-{variant:04d}",
    )


def run_synthetic_benchmark(
    *,
    count_per_family: int,
    seed: int,
    output_dir: str | pathlib.Path,
    config: ContactConfig | None = None,
) -> dict[str, Any]:
    if count_per_family < 2:
        raise DescriptorError("count_per_family must be at least 2")
    config = config or ContactConfig()
    config.validate()
    destination = pathlib.Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sample_map: list[dict[str, Any]] = []
    digest = hashlib.sha256()

    for family in _FAMILY_DIRECTIONS:
        for variant in range(count_per_family):
            severity = variant / (count_per_family - 1)
            record_id = f"{family}-{variant:04d}"
            try:
                structure = synthetic_structure(
                    family, severity, variant, rng
                )
                result = analyze_structure(
                    structure,
                    contact_config=config,
                    site_indices=(0,),
                )
                site = result["radius_contact"]["sites"][0]
                scalar = result["radius_contact"]["scalars"]
                row = {
                    "id": record_id,
                    "family": family,
                    "severity": severity,
                    "center_element": structure.species[0],
                    "ligand_element": structure.species[1],
                    "contact_vector_imbalance": (
                        site["contact_vector_imbalance"]
                    ),
                    "effective_coordination": site["effective_coordination"],
                    "radius_fit_mismatch": site["radius_fit_mismatch"],
                    "virtual_point_clearance_angstrom": (
                        site["virtual_point_clearance_angstrom"]
                    ),
                    "opposed_void_support": site["opposed_void_support"],
                    "fallback_used": site["fallback_used"],
                    "structure_scalar_snapshot": scalar,
                }
                encoded = json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                digest.update(encoded.encode("utf-8"))
                digest.update(b"\n")
                rows.append(row)
                sample_map.append(
                    {
                        "id": record_id,
                        "family": family,
                        "severity": f"{severity:.8f}",
                        "center_element": structure.species[0],
                        "ligand_element": structure.species[1],
                    }
                )
            except Exception as exc:  # preserve every benchmark failure
                failures.append(
                    {
                        "id": record_id,
                        "family": family,
                        "severity": severity,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

    expected_count = len(_FAMILY_DIRECTIONS) * count_per_family
    finite_rows = sum(
        all(
            math.isfinite(float(row[key]))
            for key in (
                "contact_vector_imbalance",
                "effective_coordination",
                "radius_fit_mismatch",
                "virtual_point_clearance_angstrom",
                "opposed_void_support",
            )
        )
        for row in rows
    )
    imbalance_values = [
        float(row["contact_vector_imbalance"]) for row in rows
    ]
    family_summaries: dict[str, Any] = {}
    for family in _FAMILY_DIRECTIONS:
        subset = [row for row in rows if row["family"] == family]
        values = [
            float(row["contact_vector_imbalance"]) for row in subset
        ]
        severities = [float(row["severity"]) for row in subset]
        family_summaries[family] = {
            "count": len(subset),
            "imbalance_min": min(values) if values else None,
            "imbalance_p50": (
                _percentile(values, 0.50) if values else None
            ),
            "imbalance_p95": (
                _percentile(values, 0.95) if values else None
            ),
            "imbalance_max": max(values) if values else None,
            "severity_imbalance_pearson": _pearson(
                severities, values
            ),
            "fallback_count": sum(
                bool(row["fallback_used"]) for row in subset
            ),
        }
    outliers = sorted(
        rows,
        key=lambda row: float(row["contact_vector_imbalance"]),
        reverse=True,
    )[:10]
    summary = {
        "schema": BENCHMARK_SCHEMA,
        "descriptor_schema": SCHEMA,
        "interpretation_scope": INTERPRETATION_SCOPE,
        "seed": seed,
        "count_per_family": count_per_family,
        "family_count": len(_FAMILY_DIRECTIONS),
        "expected_structure_count": expected_count,
        "successful_structure_count": len(rows),
        "failure_count": len(failures),
        "finite_row_count": finite_rows,
        "row_sha256": digest.hexdigest(),
        "contact_vector_imbalance": {
            "min": min(imbalance_values) if imbalance_values else None,
            "p05": (
                _percentile(imbalance_values, 0.05)
                if imbalance_values else None
            ),
            "p50": (
                _percentile(imbalance_values, 0.50)
                if imbalance_values else None
            ),
            "p95": (
                _percentile(imbalance_values, 0.95)
                if imbalance_values else None
            ),
            "max": max(imbalance_values) if imbalance_values else None,
            "mean": _mean(imbalance_values),
        },
        "fallback_row_count": sum(
            bool(row["fallback_used"]) for row in rows
        ),
        "negative_virtual_clearance_count": sum(
            float(row["virtual_point_clearance_angstrom"]) < 0
            for row in rows
        ),
        "family_summaries": family_summaries,
        "top_imbalance_outliers": [
            {
                "id": row["id"],
                "family": row["family"],
                "severity": row["severity"],
                "contact_vector_imbalance": (
                    row["contact_vector_imbalance"]
                ),
                "effective_coordination": row["effective_coordination"],
                "virtual_point_clearance_angstrom": (
                    row["virtual_point_clearance_angstrom"]
                ),
            }
            for row in outliers
        ],
        "protocol": dataclasses.asdict(config),
    }

    (destination / "thousand-run-results.json").write_text(
        json.dumps(
            {"summary": summary, "rows": rows},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with (destination / "thousand-run-failures.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("id", "family", "severity", "error_type", "message"),
        )
        writer.writeheader()
        writer.writerows(failures)
    with (destination / "thousand-run-sample-map.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "id", "family", "severity",
                "center_element", "ligand_element",
            ),
        )
        writer.writeheader()
        writer.writerows(sample_map)

    summary_lines = [
        "# Thousand-run summary",
        "",
        f"- Descriptor schema: `{SCHEMA}`",
        f"- Seed: `{seed}`",
        f"- Families: `{len(_FAMILY_DIRECTIONS)}`",
        f"- Expected structures: `{expected_count}`",
        f"- Successful structures: `{len(rows)}`",
        f"- Failures: `{len(failures)}`",
        f"- Finite rows: `{finite_rows}`",
        f"- Row SHA-256: `{summary['row_sha256']}`",
        "",
        (
            "The run is target-blind synthetic qualification. It tests "
            "numerical stability, determinism, directional response, and "
            "element-size coverage; it is not a validation against ionic "
            "conductivity, electron density, or migration barriers."
        ),
        "",
        "## Family response",
        "",
        (
            "| family | count | p50 imbalance | p95 imbalance | "
            "severity correlation | fallback |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family, family_summary in family_summaries.items():
        summary_lines.append(
            f"| {family} | {family_summary['count']} | "
            f"{family_summary['imbalance_p50']:.6f} | "
            f"{family_summary['imbalance_p95']:.6f} | "
            f"{family_summary['severity_imbalance_pearson']:.6f} | "
            f"{family_summary['fallback_count']} |"
        )
    (destination / "thousand-run-summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    analysis_lines = [
        "# Cross-material synthetic analysis",
        "",
        "## Distribution and pathologies",
        "",
        (
            "The global contact-vector imbalance spans "
            f"{summary['contact_vector_imbalance']['min']:.6f} to "
            f"{summary['contact_vector_imbalance']['max']:.6f}; the median is "
            f"{summary['contact_vector_imbalance']['p50']:.6f}."
        ),
        (
            "Fallback neighbor selection occurred in "
            f"{summary['fallback_row_count']} rows. Negative geometric "
            "virtual-point clearance occurred in "
            f"{summary['negative_virtual_clearance_count']} rows."
        ),
        "",
        "## Interpretation",
        "",
        (
            "The severity correlations measure whether a deliberately "
            "lengthened contact and small center shift increase the "
            "normalized first contact moment within each synthetic family. "
            "They are implementation diagnostics, not material-property "
            "correlations."
        ),
        "",
        "## Largest directional outliers",
        "",
        (
            "| id | family | severity | imbalance | effective coordination | "
            "virtual clearance (Å) |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["top_imbalance_outliers"]:
        analysis_lines.append(
            f"| {row['id']} | {row['family']} | {row['severity']:.6f} | "
            f"{row['contact_vector_imbalance']:.6f} | "
            f"{row['effective_coordination']:.6f} | "
            f"{row['virtual_point_clearance_angstrom']:.6f} |"
        )
    (destination / "thousand-run-analysis.md").write_text(
        "\n".join(analysis_lines) + "\n", encoding="utf-8"
    )
    return summary
