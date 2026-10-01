#!/usr/bin/env python3
"""Bond-valence-aligned and radius-contact directionality descriptors.

The default radius-contact layer is an original geometric descriptor for an
ordered POSCAR.  The opt-in explicit layer evaluates local BVS, absolute BVSM,
and bond-valence vector sums only from caller-supplied oxidation states and
pair parameters.  Neither layer is electron density, a true lone-pair site, a
BVSE barrier, or ionic conductivity.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence

try:
    from .bv_directionality_model import (
        ATOMIC_NUMBERS, BENCHMARK_SCHEMA, COVALENT_RADII_ANGSTROM,
        EXPLICIT_PROTOCOL_SCHEMA, INTERPRETATION_SCOPE, SCHEMA, ContactConfig,
        DescriptorError, ExplicitBondValenceProtocol, PairParameter, Structure,
        _cart_to_frac, _frac_to_cart,
    )
    from .bv_directionality_io import parse_poscar, read_poscar
    from .bv_directionality_core import (
        analyze_explicit_bond_valence_site, analyze_radius_contact_site,
        analyze_structure, analyze_structure_file, load_explicit_protocol,
    )
    from .bv_directionality_benchmark import run_synthetic_benchmark, synthetic_structure
except ImportError:  # executed as ``python descriptors/bv_directionality.py``
    from bv_directionality_model import (
        ATOMIC_NUMBERS, BENCHMARK_SCHEMA, COVALENT_RADII_ANGSTROM,
        EXPLICIT_PROTOCOL_SCHEMA, INTERPRETATION_SCOPE, SCHEMA, ContactConfig,
        DescriptorError, ExplicitBondValenceProtocol, PairParameter, Structure,
        _cart_to_frac, _frac_to_cart,
    )
    from bv_directionality_io import parse_poscar, read_poscar
    from bv_directionality_core import (
        analyze_explicit_bond_valence_site, analyze_radius_contact_site,
        analyze_structure, analyze_structure_file, load_explicit_protocol,
    )
    from bv_directionality_benchmark import run_synthetic_benchmark, synthetic_structure

__all__ = [
    "ATOMIC_NUMBERS", "BENCHMARK_SCHEMA", "COVALENT_RADII_ANGSTROM",
    "EXPLICIT_PROTOCOL_SCHEMA", "INTERPRETATION_SCOPE", "SCHEMA",
    "ContactConfig", "DescriptorError", "ExplicitBondValenceProtocol",
    "PairParameter", "Structure", "analyze_explicit_bond_valence_site",
    "analyze_radius_contact_site", "analyze_structure", "analyze_structure_file",
    "load_explicit_protocol", "parse_poscar", "read_poscar",
    "run_synthetic_benchmark", "synthetic_structure",
]


def _parse_indices(text: str | None) -> tuple[int, ...] | None:
    if text is None:
        return None
    values = tuple(int(token.strip()) for token in text.split(",") if token.strip())
    if not values:
        raise DescriptorError("site index list is empty")
    return values


def _config_from_args(args: argparse.Namespace) -> ContactConfig:
    return ContactConfig(
        contact_cutoff_ratio=args.contact_cutoff_ratio,
        contact_softness_ratio=args.contact_softness_ratio,
        minimum_neighbors=args.minimum_neighbors,
        maximum_neighbors=args.maximum_neighbors,
        fallback_search_radius_angstrom=args.fallback_search_radius,
        virtual_offset_angstrom=args.virtual_offset,
        virtual_clearance_cap_angstrom=args.virtual_clearance_cap,
    )


def _add_contact_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = ContactConfig()
    parser.add_argument("--contact-cutoff-ratio", type=float, default=defaults.contact_cutoff_ratio)
    parser.add_argument("--contact-softness-ratio", type=float, default=defaults.contact_softness_ratio)
    parser.add_argument("--minimum-neighbors", type=int, default=defaults.minimum_neighbors)
    parser.add_argument("--maximum-neighbors", type=int, default=defaults.maximum_neighbors)
    parser.add_argument("--fallback-search-radius", type=float, default=defaults.fallback_search_radius_angstrom)
    parser.add_argument("--virtual-offset", type=float, default=defaults.virtual_offset_angstrom)
    parser.add_argument("--virtual-clearance-cap", type=float, default=defaults.virtual_clearance_cap_angstrom)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicit BVS-vector and general radius-contact directionality descriptors"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="analyze one ordered VASP 5 POSCAR")
    analyze.add_argument("structure_file")
    analyze.add_argument("--explicit-bv-protocol")
    analyze.add_argument("--site-indices", help="comma-separated zero-based site indices")
    analyze.add_argument("--output", default="-")
    _add_contact_arguments(analyze)

    benchmark = subparsers.add_parser("benchmark", help="run deterministic five-family synthetic qualification")
    benchmark.add_argument("--count-per-family", type=int, default=200)
    benchmark.add_argument("--seed", type=int, default=20251001)
    benchmark.add_argument("--output-dir", required=True)
    _add_contact_arguments(benchmark)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        if args.command == "analyze":
            result = analyze_structure_file(
                args.structure_file,
                explicit_protocol_file=args.explicit_bv_protocol,
                contact_config=config,
                site_indices=_parse_indices(args.site_indices),
            )
            encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
            if args.output == "-":
                sys.stdout.write(encoded)
            else:
                output = pathlib.Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(encoded, encoding="utf-8")
        else:
            summary = run_synthetic_benchmark(
                count_per_family=args.count_per_family,
                seed=args.seed,
                output_dir=args.output_dir,
                config=config,
            )
            sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return 0
    except (DescriptorError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
