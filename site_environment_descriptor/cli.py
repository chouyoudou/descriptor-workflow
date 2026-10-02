"""Command-line interface for the periodic site-environment descriptor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .descriptor import DescriptorConfig, evaluate_probe_elements, featurize_structure
from .poscar import read_poscar


def _split_tokens(text: str | None) -> list[str]:
    if not text:
        return []
    return [token.strip() for token in text.replace(",", " ").split() if token.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute fixed periodic site-environment scalars from a fully occupied "
            "POSCAR. Optional probes are explicit frozen-cage neutral-radius checks."
        )
    )
    parser.add_argument("poscar", help="VASP POSCAR/CONTCAR path")
    parser.add_argument("-o", "--output", help="write JSON to this path; stdout otherwise")
    parser.add_argument(
        "--vasp4-species",
        help="comma/space-separated symbols for a VASP-4 POSCAR without names",
    )
    parser.add_argument(
        "--probes",
        help="comma/space-separated explicit probe elements; never inferred",
    )
    parser.add_argument("--normalized-inner", type=float, default=0.95)
    parser.add_argument("--normalized-outer", type=float, default=1.35)
    parser.add_argument("--max-image-range", type=int, default=6)
    parser.add_argument("--max-neighbor-evaluations", type=int, default=8_000_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DescriptorConfig(
        normalized_inner=args.normalized_inner,
        normalized_outer=args.normalized_outer,
        max_image_range=args.max_image_range,
        max_neighbor_evaluations=args.max_neighbor_evaluations,
    )
    species_override = _split_tokens(args.vasp4_species) or None
    structure = read_poscar(args.poscar, species_override=species_override)
    result = featurize_structure(structure, config)
    payload: dict[str, object] = {"descriptor": result.to_jsonable()}
    probes = _split_tokens(args.probes)
    if probes:
        payload["explicit_probes"] = evaluate_probe_elements(
            structure, probes, config
        ).to_jsonable()
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
