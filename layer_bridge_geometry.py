"""Low-cost descriptors for layers, metal coordination, and ligand bridges.

The implementation is deliberately dependency-free. It infers a periodic bond
multigraph from covalent radii, determines component dimensionality from the rank
of graph-cycle translations, and summarizes static geometry only. It does not
predict pressure response, magnetic order, elasticity, stability, or synthesis
history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from _layer_bridge_model import (
    DEFAULT_BOND_SCALE, DEFAULT_MAX_BRIDGE_ATOMS, DESCRIPTOR_NAMES, SCHEMA,
    Edge, ElementData, Structure, default_element_table, load_element_table,
    parse_poscar, read_poscar,
)
from _layer_bridge_graph import build_periodic_bonds, component_dimensionalities
from _layer_bridge_features import analyze_cutoff_ensemble, analyze_structure

__all__ = [
    "DEFAULT_BOND_SCALE", "DEFAULT_MAX_BRIDGE_ATOMS", "DESCRIPTOR_NAMES", "SCHEMA",
    "Edge", "ElementData", "Structure", "analyze_cutoff_ensemble",
    "analyze_structure", "build_periodic_bonds", "component_dimensionalities",
    "default_element_table", "load_element_table", "parse_poscar", "read_poscar",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute static layer, coordination, and bridge descriptors from a POSCAR."
    )
    parser.add_argument("poscar", type=Path)
    parser.add_argument("--bond-scale", type=float, default=DEFAULT_BOND_SCALE)
    parser.add_argument("--minimum-distance", type=float, default=0.35)
    parser.add_argument("--max-bridge-atoms", type=int, default=DEFAULT_MAX_BRIDGE_ATOMS)
    parser.add_argument("--max-bridge-paths", type=int, default=10000)
    parser.add_argument("--element-table", type=Path, default=None)
    parser.add_argument(
        "--cutoff-ensemble",
        action="store_true",
        help="also evaluate fixed scales 1.10, 1.18, and 1.26",
    )
    args = parser.parse_args(argv)
    try:
        structure = read_poscar(args.poscar)
        table = load_element_table(args.element_table)
        if args.cutoff_ensemble:
            result = analyze_cutoff_ensemble(
                structure,
                element_table=table,
                minimum_distance=args.minimum_distance,
                max_bridge_atoms=args.max_bridge_atoms,
                max_bridge_paths=args.max_bridge_paths,
            )
        else:
            result = analyze_structure(
                structure,
                element_table=table,
                bond_scale=args.bond_scale,
                minimum_distance=args.minimum_distance,
                max_bridge_atoms=args.max_bridge_atoms,
                max_bridge_paths=args.max_bridge_paths,
            )
    except FileNotFoundError:
        print("error: input file not found", file=sys.stderr)
        return 2
    except json.JSONDecodeError:
        print("error: invalid element-table JSON", file=sys.stderr)
        return 2
    except OSError:
        print("error: unable to read input", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
