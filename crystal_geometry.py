"""Small geometric utilities for periodic unit cells."""
import argparse
import json
import math
from pathlib import Path


def cell_volume(lattice):
    """Return the volume in cubic angstroms for three row lattice vectors."""
    if len(lattice) != 3 or any(len(row) != 3 for row in lattice):
        raise ValueError("lattice must contain three 3D vectors")
    if any(type(x) not in (int, float) or not math.isfinite(x)
           for row in lattice for x in row):
        raise ValueError("lattice entries must be finite real numbers")
    a, b, c = lattice
    determinant = (a[0] * (b[1] * c[2] - b[2] * c[1])
                   - a[1] * (b[0] * c[2] - b[2] * c[0])
                   + a[2] * (b[0] * c[1] - b[1] * c[0]))
    volume = abs(determinant)
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("cell must have a finite positive volume")
    return volume


def volume_per_atom(lattice, atom_count):
    """Return cubic angstroms per atom for a fully occupied unit cell."""
    if type(atom_count) is not int or atom_count < 1:
        raise ValueError("atom_count must be a positive integer")
    return cell_volume(lattice) / atom_count


def main():
    parser = argparse.ArgumentParser(description="Compute unit-cell geometry.")
    parser.add_argument("input", type=Path, help="JSON file containing cells")
    args = parser.parse_args()
    cells = json.loads(args.input.read_text(encoding="utf-8"))
    results = [
        {"id": cell["id"],
         "volume_angstrom3": cell_volume(cell["lattice"]),
         "volume_per_atom_angstrom3": volume_per_atom(cell["lattice"], cell["atom_count"])}
        for cell in cells
    ]
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
