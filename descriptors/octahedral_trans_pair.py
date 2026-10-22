#!/usr/bin/env python3
"""Octahedral trans-pair deformation (OTPD) descriptor.

This facade exposes the public API and command-line interface. OTPD is a
static geometry descriptor: it does not infer oxidation states, ligand-field
levels, magnetic anisotropy, zero-field splitting, relaxation barriers, or
single-ion-magnet behaviour.
"""

from __future__ import annotations

import argparse
import json

try:  # Package import in the public repository.
    from .otpd_types import (
        ALL_SCALARS, CORE_SCALARS, DEFAULT_COVALENT_RADII, ELEMENT_PROTOCOL,
        ELEMENT_SYMBOLS, EXTENSION_SCALARS, SCHEMA, STATIC_SCOPE, DescriptorError,
        Neighbor, Settings, SiteResult, StructureData, canonical_symbol,
        load_project_element_table, parse_poscar, radius_table, structure_from_record,
    )
    from .otpd_geometry import periodic_neighbors, select_six_shell
    from .otpd_vectors import analyze_six_vectors
    from .otpd_structure import analyze_site, analyze_structure
except ImportError:  # Flat source directory in the private bundle runtime.
    from otpd_types import (
        ALL_SCALARS, CORE_SCALARS, DEFAULT_COVALENT_RADII, ELEMENT_PROTOCOL,
        ELEMENT_SYMBOLS, EXTENSION_SCALARS, SCHEMA, STATIC_SCOPE, DescriptorError,
        Neighbor, Settings, SiteResult, StructureData, canonical_symbol,
        load_project_element_table, parse_poscar, radius_table, structure_from_record,
    )
    from otpd_geometry import periodic_neighbors, select_six_shell
    from otpd_vectors import analyze_six_vectors
    from otpd_structure import analyze_site, analyze_structure

__all__ = [
    "ALL_SCALARS", "CORE_SCALARS", "DEFAULT_COVALENT_RADII",
    "ELEMENT_PROTOCOL", "ELEMENT_SYMBOLS", "EXTENSION_SCALARS", "SCHEMA",
    "STATIC_SCOPE", "DescriptorError", "Neighbor", "Settings", "SiteResult",
    "StructureData", "canonical_symbol", "load_project_element_table",
    "parse_poscar", "radius_table", "structure_from_record",
    "analyze_six_vectors", "analyze_site", "analyze_structure",
    "periodic_neighbors", "select_six_shell",
]

def _main() -> int:
    parser = argparse.ArgumentParser(description="Compute octahedral trans-pair deformation scalars")
    parser.add_argument("structure", help="VASP 5 POSCAR")
    parser.add_argument("--center-element", action="append", default=[])
    parser.add_argument("--element-table", help="project element table JSON")
    parser.add_argument("--sites", action="store_true", help="include site-level details")
    parser.add_argument("--no-plane-extension", action="store_true")
    args = parser.parse_args()

    table = None
    table_meta = None
    if args.element_table:
        table, table_meta = load_project_element_table(args.element_table)
    structure = parse_poscar(args.structure)
    result = analyze_structure(
        structure,
        center_elements=set(args.center_element) if args.center_element else None,
        radii=table,
        include_sites=args.sites,
        include_plane_extension=not args.no_plane_extension,
    )
    if table_meta is not None:
        result["element_table"] = table_meta
    print(json.dumps(result, sort_keys=True, allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
