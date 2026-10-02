"""Public scalar API and command-line interface."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys
from collections.abc import Callable, Sequence

from .analysis import DESCRIPTOR_NAMES, DESCRIPTOR_UNITS, analyze_structure
from .model import DescriptorConfig, DescriptorError
from .poscar import read_poscar

TASK_ID = "paper-03aef80c9d0a380d998e06d1"
SCHEMA = "molecular-packing-descriptor/result-v1"
VERSION = "1.0.0"


def compute_descriptor(
    poscar_path: str | pathlib.Path,
    descriptor: str,
    *,
    config: DescriptorConfig | None = None,
) -> float:
    """Compute exactly one finite scalar from one POSCAR path."""

    if descriptor not in DESCRIPTOR_NAMES:
        raise DescriptorError(
            f"unknown descriptor {descriptor!r}; choose one of {', '.join(DESCRIPTOR_NAMES)}"
        )
    structure = read_poscar(poscar_path)
    value = float(analyze_structure(structure, config).values[descriptor])
    if not math.isfinite(value):
        raise DescriptorError(f"descriptor {descriptor} is not finite")
    return value


def compute_all(
    poscar_path: str | pathlib.Path,
    *,
    config: DescriptorConfig | None = None,
) -> dict[str, float]:
    """Convenience analysis returning the complete named scalar family."""

    structure = read_poscar(poscar_path)
    return dict(analyze_structure(structure, config).values)


def _scalar_function(name: str) -> Callable[[str | pathlib.Path], float]:
    def function(poscar_path: str | pathlib.Path) -> float:
        return compute_descriptor(poscar_path, name)

    function.__name__ = name
    function.__qualname__ = name
    function.__doc__ = f"Return the {name!r} finite scalar for one POSCAR path."
    return function


spacer_path_extension = _scalar_function("spacer_path_extension")
spacer_atom_fraction = _scalar_function("spacer_atom_fraction")
component_shape_anisotropy = _scalar_function("component_shape_anisotropy")
coordination_shell_radial_distortion = _scalar_function(
    "coordination_shell_radial_distortion"
)
coordination_shell_orientation_anisotropy = _scalar_function(
    "coordination_shell_orientation_anisotropy"
)
interfragment_contact_compactness = _scalar_function(
    "interfragment_contact_compactness"
)
interfragment_contact_coordination = _scalar_function(
    "interfragment_contact_coordination"
)
interfragment_contact_anisotropy = _scalar_function(
    "interfragment_contact_anisotropy"
)
heteroelement_contact_fraction = _scalar_function("heteroelement_contact_fraction")
contact_pair_specificity = _scalar_function("contact_pair_specificity")

DESCRIPTOR_FUNCTIONS: dict[str, Callable[[str | pathlib.Path], float]] = {
    name: globals()[name] for name in DESCRIPTOR_NAMES
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute one static molecular-crystal packing scalar from a POSCAR. "
            "No charge, spin, phase history, liquid property, or response is inferred."
        )
    )
    parser.add_argument("poscar", help="input POSCAR/CONTCAR path")
    parser.add_argument("--descriptor", required=True, choices=DESCRIPTOR_NAMES)
    parser.add_argument(
        "--output",
        default=f"output/{TASK_ID}/result.json",
        help="JSON output path, or '-' for stdout",
    )
    parser.add_argument("--bond-scale", type=float, default=DescriptorConfig.bond_scale)
    parser.add_argument(
        "--contact-ratio-full", type=float, default=DescriptorConfig.contact_ratio_full
    )
    parser.add_argument(
        "--contact-ratio-max", type=float, default=DescriptorConfig.contact_ratio_max
    )
    parser.add_argument(
        "--contact-ratio-min", type=float, default=DescriptorConfig.contact_ratio_min
    )
    parser.add_argument(
        "--min-spacer-bonds", type=int, default=DescriptorConfig.min_spacer_bonds
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config = DescriptorConfig(
        bond_scale=args.bond_scale,
        contact_ratio_full=args.contact_ratio_full,
        contact_ratio_max=args.contact_ratio_max,
        contact_ratio_min=args.contact_ratio_min,
        min_spacer_bonds=args.min_spacer_bonds,
    )
    try:
        structure = read_poscar(args.poscar)
        analysis = analyze_structure(structure, config)
        value = analysis.values[args.descriptor]
        result = {
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "version": VERSION,
            "descriptor": args.descriptor,
            "value": value,
            "units": DESCRIPTOR_UNITS[args.descriptor],
            "input": str(pathlib.Path(args.poscar)),
            "parameters": dataclasses.asdict(config),
            "diagnostics": {
                "n_atoms": structure.n_atoms,
                "n_bonds": analysis.n_bonds,
                "n_components": analysis.n_components,
                "n_periodic_components": analysis.n_periodic_components,
                "n_contacts": analysis.n_contacts,
            },
            "interpretation_scope": (
                "static geometry and element-radius lookup only; not charge, oxidation state, "
                "magnetism, liquid behavior, solubility, thermal stability, or processing history"
            ),
        }
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output == "-":
            sys.stdout.write(encoded)
        else:
            output = pathlib.Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded, encoding="utf-8")
    except (DescriptorError, OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
