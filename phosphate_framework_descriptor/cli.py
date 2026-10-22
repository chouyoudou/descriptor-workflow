"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .descriptor import DescriptorConfig, featurize_structure
from .poscar import read_poscar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phosphate-framework-descriptor",
        description=(
            "Compute fixed periodic framework/polyhedron/atomistic-void descriptors from a fully occupied POSCAR."
        ),
    )
    parser.add_argument("poscar", help="input POSCAR path")
    parser.add_argument("-o", "--output", help="JSON output path; stdout when omitted")
    parser.add_argument("--species", nargs="+", help="element symbols for a VASP-4 POSCAR")
    parser.add_argument("--bond-scale", type=float, default=1.25)
    parser.add_argument("--grid-spacing", type=float, default=0.80)
    parser.add_argument("--max-grid-points", type=int, default=32768)
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    structure = read_poscar(args.poscar, species_override=args.species)
    config = DescriptorConfig(
        bond_scale=args.bond_scale,
        grid_spacing=args.grid_spacing,
        max_grid_points=args.max_grid_points,
    )
    payload = featurize_structure(structure, config).to_jsonable()
    text = json.dumps(payload, indent=args.indent, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
