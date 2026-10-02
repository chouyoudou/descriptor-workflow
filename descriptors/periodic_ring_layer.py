#!/usr/bin/env python3
"""Periodic Ring–Layer Topology (PRLT) descriptor.

PRLT builds a neutral-radius periodic contact graph, reports its translation rank,
enumerates bounded zero-winding shortest-path rings up to a fixed length, measures
ring/layer geometry, and couples those structural channels to neutral element identity.
It is a static structure discriminator, not a bond-order, electronic, energetic,
kinetic, biological, antimicrobial, or application-performance model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .prlt_graph import analyze_components, build_periodic_graph
    from .prlt_metrics import aggregate_scalars, categorical_assortativity
    from .prlt_rings import enumerate_shortest_path_rings
    from .prlt_types import (
        COLUMNS,
        COLUMN_DEFINITIONS,
        IMPLEMENTATION_VERSION,
        SCHEMA,
        STATIC_SCOPE,
        DescriptorError,
        ElementLookup,
        InputError,
        ResourceLimitError,
        Settings,
        StructureData,
        parse_poscar,
        structure_from_record,
    )
except ImportError:  # Flat source layout in the private Actions bundle.
    from prlt_graph import analyze_components, build_periodic_graph
    from prlt_metrics import aggregate_scalars, categorical_assortativity
    from prlt_rings import enumerate_shortest_path_rings
    from prlt_types import (
        COLUMNS,
        COLUMN_DEFINITIONS,
        IMPLEMENTATION_VERSION,
        SCHEMA,
        STATIC_SCOPE,
        DescriptorError,
        ElementLookup,
        InputError,
        ResourceLimitError,
        Settings,
        StructureData,
        parse_poscar,
        structure_from_record,
    )

__all__ = [
    "COLUMNS",
    "COLUMN_DEFINITIONS",
    "IMPLEMENTATION_VERSION",
    "SCHEMA",
    "STATIC_SCOPE",
    "DescriptorError",
    "ElementLookup",
    "InputError",
    "ResourceLimitError",
    "Settings",
    "StructureData",
    "parse_poscar",
    "structure_from_record",
    "categorical_assortativity",
    "analyze_structure",
]


def analyze_structure(
    structure: StructureData,
    *,
    settings: Settings | None = None,
    lookup: ElementLookup | None = None,
) -> dict[str, Any]:
    settings = settings or Settings()
    lookup = lookup or ElementLookup()
    edges, adjacency, graph_diagnostics = build_periodic_graph(structure, lookup, settings)
    components, atom_component = analyze_components(structure, edges, adjacency)
    rings, ring_diagnostics, ring_incomplete = enumerate_shortest_path_rings(
        structure,
        edges,
        adjacency,
        components,
        atom_component,
        settings,
    )
    scalars, unavailable_reasons, status, aggregate_diagnostics = aggregate_scalars(
        structure,
        lookup,
        edges,
        components,
        rings,
        ring_incomplete,
    )
    component_summaries = [
        {
            "component_id": component.component_id,
            "atom_count": len(component.atoms),
            "edge_count": len(component.edge_ids),
            "translation_rank": component.rank,
            "cycle_translation_count": len(component.cycle_translations),
            "cycle_translations": [list(item) for item in component.cycle_translations],
            "normal": component.normal.tolist() if component.normal is not None else None,
        }
        for component in components
    ]
    return {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "static_scope": STATIC_SCOPE,
        "status": status,
        "scalars": scalars,
        "unavailable_reasons": unavailable_reasons,
        "diagnostics": {
            "structure_name": structure.name,
            "atom_count": structure.atom_count,
            "composition": {
                symbol: structure.species.count(symbol) for symbol in sorted(set(structure.species))
            },
            "settings": settings.as_dict(),
            "element_protocol": lookup.protocol,
            "element_table": lookup.snapshot(structure.species),
            "graph": graph_diagnostics,
            "components": component_summaries,
            "rings": ring_diagnostics,
            "aggregate": aggregate_diagnostics,
        },
    }


def _read_json_structure(path: str | Path) -> StructureData:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InputError("JSON structure file must contain one object")
    return structure_from_record(value)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Compute periodic ring-layer topology scalars")
    parser.add_argument("structure", help="VASP 5 POSCAR or one JSON structure object")
    parser.add_argument("--json", action="store_true", help="read a JSON structure object")
    parser.add_argument("--max-ring-size", type=int, default=12)
    parser.add_argument("--bond-ratio", type=float, default=1.20)
    args = parser.parse_args()
    settings = Settings(max_ring_size=args.max_ring_size, bond_ratio=args.bond_ratio)
    structure = _read_json_structure(args.structure) if args.json else parse_poscar(args.structure)
    result = analyze_structure(structure, settings=settings)
    print(json.dumps(result, sort_keys=True, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
