#!/usr/bin/env python3
"""Static, interpretable site--void coupling descriptors for POSCAR structures.

The descriptor links three quantities available from one periodic structure:

* displacement of a site from the centroid of its automatically selected local
  coordination shell;
* the least-blocked local axis and the forward/backward free-path contrast
  along that axis; and
* an optional element-radius fit of the explicit site or hypothetical
  substitutions to the unchanged local cage.

The implementation is dependency-free and deterministic. It is a geometric
screening tool. It does not calculate a pressure derivative, force constant,
bond stiffness, polarization, piezoelectric tensor, migration barrier, or a
relaxed substitution response.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

try:
    from .site_void_atoms import (
        COVALENT_RADII_ANGSTROM, SCHEMA, STATIC_SCOPE, DescriptorError,
    )
    from .site_void_core import analyze_site, analyze_structure
    from .site_void_io import parse_poscar, read_poscar, structure_from_json_record
    from .site_void_math import Mat3, Vec3, _cart_to_frac, _inverse3, _vadd, _wrap_frac
    from .site_void_model import (
        DescriptorConfig, NeighborImage, RadiusProjection, SiteDescriptor, Structure,
    )
    from .site_void_neighbors import enumerate_neighbor_images, select_coordination_shell
except ImportError:  # flat private task bundle
    from site_void_atoms import COVALENT_RADII_ANGSTROM, SCHEMA, STATIC_SCOPE, DescriptorError
    from site_void_core import analyze_site, analyze_structure
    from site_void_io import parse_poscar, read_poscar, structure_from_json_record
    from site_void_math import Mat3, Vec3, _cart_to_frac, _inverse3, _vadd, _wrap_frac
    from site_void_model import DescriptorConfig, NeighborImage, RadiusProjection, SiteDescriptor, Structure
    from site_void_neighbors import enumerate_neighbor_images, select_coordination_shell


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def synthetic_cage_structure(index: int) -> Structure:
    """Create one deterministic periodic cage for public qualification.

    The family spans center species, ligand species, anisotropic cage radii,
    outward displacement of one ligand, and an independently shifted center.
    It is synthetic and carries no material-property labels.
    """

    if index < 0:
        raise DescriptorError("synthetic structure index must be non-negative")
    center_symbols = ("Ca", "Sr", "Ba", "La", "Na", "K", "Ga", "Ta")
    ligand_symbols = ("O", "F", "S", "Se")
    center_symbol = center_symbols[index % len(center_symbols)]
    ligand_symbol = ligand_symbols[(index // len(center_symbols)) % len(ligand_symbols)]
    phase = (index * 0.6180339887498949) % 1.0
    phase2 = (index * 0.4142135623730950) % 1.0
    phase3 = (index * 0.7320508075688772) % 1.0
    base_radius = 2.05 + 0.85 * phase
    anisotropy_y = 0.88 + 0.24 * phase2
    anisotropy_z = 0.90 + 0.20 * phase3
    open_extension = 0.15 + 2.35 * ((index % 41) / 40.0)
    shift_magnitude = 0.32 * (((index * 17) % 37) / 36.0)
    azimuth = 2.0 * math.pi * phase2
    polar = math.pi * (0.25 + 0.5 * phase3)
    shift = (
        shift_magnitude * math.sin(polar) * math.cos(azimuth),
        shift_magnitude * math.sin(polar) * math.sin(azimuth),
        shift_magnitude * math.cos(polar),
    )

    lattice_length = 17.0 + 0.6 * ((index % 11) / 10.0)
    lattice: Mat3 = (
        (lattice_length, 0.0, 0.0),
        (0.15 * math.sin(azimuth), lattice_length + 0.2, 0.0),
        (0.12 * math.cos(azimuth), 0.08 * math.sin(polar), lattice_length + 0.4),
    )
    inverse = _inverse3(lattice)
    cage_center_cart = (0.5 * lattice_length, 0.5 * lattice_length, 0.5 * lattice_length)
    center_cart = _vadd(cage_center_cart, shift)
    ligand_vectors = (
        (base_radius + open_extension, 0.0, 0.0),
        (-base_radius, 0.0, 0.0),
        (0.0, base_radius * anisotropy_y, 0.0),
        (0.0, -base_radius * anisotropy_y, 0.0),
        (0.0, 0.0, base_radius * anisotropy_z),
        (0.0, 0.0, -base_radius * anisotropy_z),
    )
    cart_coords = [center_cart] + [
        _vadd(cage_center_cart, vector) for vector in ligand_vectors
    ]
    frac_coords = tuple(_wrap_frac(_cart_to_frac(cart, inverse)) for cart in cart_coords)
    return Structure(
        lattice=lattice,
        species=(center_symbol,) + (ligand_symbol,) * len(ligand_vectors),
        frac_coords=frac_coords,
        comment=f"synthetic-site-void-cage-{index:04d}",
    )


def run_synthetic_benchmark(
    count: int,
    *,
    output_jsonl: str | pathlib.Path | None = None,
    summary_path: str | pathlib.Path | None = None,
    config: DescriptorConfig | None = None,
) -> dict[str, Any]:
    if count < 1:
        raise DescriptorError("benchmark count must be positive")
    config = config or DescriptorConfig()
    digest = hashlib.sha256()
    scalar_rows: list[dict[str, Any]] = []
    output_handle = None
    try:
        if output_jsonl is not None:
            output_path = pathlib.Path(output_jsonl)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open("w", encoding="utf-8", newline="\n")
        for index in range(count):
            structure = synthetic_cage_structure(index)
            result = analyze_structure(
                structure,
                center_indices=(0,),
                candidate_elements=("Al", "Ga", "Ca"),
                config=config,
            )
            row = {
                "id": structure.comment,
                "index": index,
                "center_symbol": structure.species[0],
                "ligand_symbol": structure.species[1],
                "scalars": result["scalars"],
                "site": result["sites"][0],
            }
            encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            digest.update(encoded.encode("utf-8"))
            digest.update(b"\n")
            if output_handle is not None:
                output_handle.write(encoded + "\n")
            scalar_rows.append(row["scalars"])
    finally:
        if output_handle is not None:
            output_handle.close()

    coupling_values = [
        float(row["site_void_coupling_signed_mean"]) for row in scalar_rows
    ]
    support_values = [float(row["void_axis_uniqueness_mean"]) for row in scalar_rows]
    summary = {
        "schema": "site-void-coupling-benchmark/1",
        "descriptor_schema": SCHEMA,
        "interpretation_scope": STATIC_SCOPE,
        "structure_count": count,
        "row_sha256": digest.hexdigest(),
        "finite_scalar_rows": sum(
            1
            for row in scalar_rows
            if all(
                math.isfinite(float(value))
                for value in row.values()
                if isinstance(value, (int, float))
            )
        ),
        "site_void_coupling_signed_min": min(coupling_values),
        "site_void_coupling_signed_max": max(coupling_values),
        "site_void_coupling_signed_mean": _mean(coupling_values),
        "void_axis_uniqueness_min": min(support_values),
        "void_axis_uniqueness_max": max(support_values),
        "protocol": dataclasses.asdict(config),
    }
    if summary_path is not None:
        destination = pathlib.Path(summary_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return summary


# ----------------------------------- CLI -------------------------------------


def _load_radius_overrides(path: str | None) -> dict[str, float] | None:
    if path is None:
        return None
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DescriptorError("radius override JSON must be an object")
    return {str(key): float(value) for key, value in data.items()}


def _write_json(data: Any, destination: str) -> None:
    encoded = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination == "-":
        sys.stdout.write(encoded)
    else:
        path = pathlib.Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")


def _config_from_args(args: argparse.Namespace) -> DescriptorConfig:
    return DescriptorConfig(
        search_radius_angstrom=args.search_radius,
        contact_scale=args.contact_scale,
        shell_gap_ratio=args.shell_gap_ratio,
        min_neighbors=args.min_neighbors,
        max_neighbors=args.max_neighbors,
        clearance_cap_angstrom=args.clearance_cap,
        obstacle_radius_scale=args.obstacle_radius_scale,
    )


def _add_protocol_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--search-radius", type=float, default=8.0)
    parser.add_argument("--contact-scale", type=float, default=1.35)
    parser.add_argument("--shell-gap-ratio", type=float, default=1.18)
    parser.add_argument("--min-neighbors", type=int, default=4)
    parser.add_argument("--max-neighbors", type=int, default=16)
    parser.add_argument("--clearance-cap", type=float, default=5.0)
    parser.add_argument("--obstacle-radius-scale", type=float, default=1.0)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Static site--void coupling descriptors for periodic POSCAR structures"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="analyze one POSCAR")
    analyze_parser.add_argument("poscar")
    analyze_parser.add_argument("--output", default="-")
    analyze_parser.add_argument("--center-element", action="append", default=[])
    analyze_parser.add_argument("--center-index", action="append", type=int, default=[])
    analyze_parser.add_argument("--candidate-element", action="append", default=[])
    analyze_parser.add_argument("--radii-json")
    _add_protocol_arguments(analyze_parser)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="run deterministic public synthetic qualification structures"
    )
    benchmark_parser.add_argument("--count", type=int, default=1200)
    benchmark_parser.add_argument("--output-jsonl", required=True)
    benchmark_parser.add_argument("--summary", required=True)
    _add_protocol_arguments(benchmark_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        if args.command == "analyze":
            structure = read_poscar(args.poscar)
            result = analyze_structure(
                structure,
                center_elements=args.center_element,
                center_indices=args.center_index,
                candidate_elements=args.candidate_element,
                config=config,
                radii_overrides=_load_radius_overrides(args.radii_json),
            )
            _write_json(result, args.output)
        elif args.command == "benchmark":
            summary = run_synthetic_benchmark(
                args.count,
                output_jsonl=args.output_jsonl,
                summary_path=args.summary,
                config=config,
            )
            sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
        else:  # pragma: no cover - argparse enforces known commands
            parser.error(f"unknown command {args.command!r}")
    except (DescriptorError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
