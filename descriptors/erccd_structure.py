"""Periodic structure model plus POSCAR and JARVIS JSON adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .erccd_elements import DescriptorError, canonical_symbol
from .erccd_geometry import Matrix, Vector, as_vector, determinant, inverse, row_times_matrix


@dataclass(frozen=True)
class Structure:
    lattice: Matrix
    species: tuple[str, ...]
    frac_coords: tuple[Vector, ...]
    identifier: str = ""

    def __post_init__(self) -> None:
        lattice = tuple(as_vector(row, name="lattice vector") for row in self.lattice)
        if len(lattice) != 3:
            raise DescriptorError("lattice must contain three vectors")
        if abs(determinant(lattice)) < 1.0e-10:
            raise DescriptorError("lattice volume must be non-zero")
        species = tuple(canonical_symbol(item) for item in self.species)
        coordinates = tuple(as_vector(row, name="fractional coordinate") for row in self.frac_coords)
        if not species:
            raise DescriptorError("structure must contain at least one atom")
        if len(species) != len(coordinates):
            raise DescriptorError("species and coordinate counts differ")
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "frac_coords", coordinates)

    @property
    def atom_count(self) -> int:
        return len(self.species)

    @property
    def volume_angstrom3(self) -> float:
        return abs(determinant(self.lattice))

    def repeated(self, repeats: tuple[int, int, int]) -> "Structure":
        if any(value <= 0 for value in repeats):
            raise DescriptorError("supercell repeats must be positive")
        lattice: Matrix = tuple(
            tuple(component * repeats[index] for component in self.lattice[index])
            for index in range(3)
        )  # type: ignore[assignment]
        species: list[str] = []
        coords: list[Vector] = []
        for a in range(repeats[0]):
            for b in range(repeats[1]):
                for c in range(repeats[2]):
                    for symbol, frac in zip(self.species, self.frac_coords):
                        species.append(symbol)
                        coords.append(
                            ((frac[0] + a) / repeats[0],
                             (frac[1] + b) / repeats[1],
                             (frac[2] + c) / repeats[2])
                        )
        return Structure(lattice, tuple(species), tuple(coords), self.identifier + " supercell")


@dataclass(frozen=True)
class DescriptorConfig:
    contact_factor: float = 1.25
    cutoff_tolerance_angstrom: float = 1.0e-9
    near_cutoff_fraction: float = 0.02

    def validate(self) -> "DescriptorConfig":
        if not (1.0 <= self.contact_factor <= 2.0):
            raise DescriptorError("contact_factor must be in [1.0, 2.0]")
        if self.cutoff_tolerance_angstrom < 0.0:
            raise DescriptorError("cutoff tolerance cannot be negative")
        if not (0.0 <= self.near_cutoff_fraction <= 0.2):
            raise DescriptorError("near_cutoff_fraction must be in [0, 0.2]")
        return self


def parse_poscar(text: str, *, identifier: str = "") -> Structure:
    """Parse VASP 5+ POSCAR text, including Cartesian and negative scales."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        raise DescriptorError("POSCAR is too short")
    try:
        scale_raw = float(lines[1].split()[0])
        raw_lattice: Matrix = tuple(
            as_vector([float(item) for item in lines[index].split()[:3]], name="lattice vector")
            for index in (2, 3, 4)
        )  # type: ignore[assignment]
    except (IndexError, ValueError) as exc:
        raise DescriptorError("invalid POSCAR scale or lattice") from exc
    raw_volume = abs(determinant(raw_lattice))
    if raw_volume <= 1.0e-12 or scale_raw == 0.0:
        raise DescriptorError("POSCAR lattice/scale gives zero volume")
    linear_scale = scale_raw if scale_raw > 0.0 else (abs(scale_raw) / raw_volume) ** (1.0 / 3.0)
    lattice: Matrix = tuple(
        tuple(linear_scale * component for component in row) for row in raw_lattice
    )  # type: ignore[assignment]

    symbols_line = lines[5].split()
    try:
        [int(item) for item in symbols_line]
    except ValueError:
        symbols = [canonical_symbol(item) for item in symbols_line]
    else:
        raise DescriptorError("VASP 4 POSCAR without element symbols is unsupported")
    try:
        counts = [int(item) for item in lines[6].split()]
    except ValueError as exc:
        raise DescriptorError("invalid POSCAR element counts") from exc
    if len(symbols) != len(counts) or any(count < 0 for count in counts):
        raise DescriptorError("POSCAR symbols/counts mismatch")
    species = tuple(symbol for symbol, count in zip(symbols, counts) for _ in range(count))
    cursor = 7
    if lines[cursor].strip().lower().startswith("s"):
        cursor += 1
    mode = lines[cursor].strip().lower()
    cursor += 1
    cartesian = mode.startswith(("c", "k"))
    direct = mode.startswith("d")
    if not (cartesian or direct):
        raise DescriptorError(f"unsupported POSCAR coordinate mode {lines[cursor - 1]!r}")
    if len(lines) < cursor + len(species):
        raise DescriptorError("POSCAR has fewer coordinates than atoms")
    lattice_inverse = inverse(lattice)
    frac_coords: list[Vector] = []
    for index in range(len(species)):
        try:
            values = as_vector(
                [float(item) for item in lines[cursor + index].split()[:3]], name="coordinate"
            )
        except ValueError as exc:
            raise DescriptorError("invalid POSCAR coordinate") from exc
        if direct:
            frac = values
        else:
            cart = tuple(linear_scale * component for component in values)
            frac = row_times_matrix(cart, lattice_inverse)  # type: ignore[arg-type]
        frac_coords.append(frac)
    return Structure(lattice, species, tuple(frac_coords), identifier or lines[0].strip())


def structure_from_jarvis_record(record: Mapping[str, Any]) -> Structure:
    """Convert a JARVIS-DFT JSON row without importing jarvis-tools."""
    atoms = record.get("atoms", record)
    if not isinstance(atoms, Mapping):
        raise DescriptorError("JARVIS record lacks an atoms mapping")
    try:
        lattice: Matrix = tuple(
            as_vector(row, name="lattice vector") for row in atoms["lattice_mat"]
        )  # type: ignore[assignment,index]
        species = tuple(str(item) for item in atoms["elements"])  # type: ignore[index]
        coordinates = tuple(as_vector(row, name="coordinate") for row in atoms["coords"])  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise DescriptorError("malformed JARVIS atoms mapping") from exc
    if bool(atoms.get("cartesian", True)):
        lattice_inverse = inverse(lattice)
        frac = tuple(row_times_matrix(row, lattice_inverse) for row in coordinates)
    else:
        frac = coordinates
    identifier = str(record.get("jid") or record.get("id") or record.get("name") or "")
    return Structure(lattice, species, frac, identifier)


__all__ = [
    "DescriptorConfig",
    "Structure",
    "parse_poscar",
    "structure_from_jarvis_record",
]
