"""Small, dependency-light parser for fully occupied VASP POSCAR structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .elements import normalize_symbol


@dataclass(frozen=True)
class PeriodicStructure:
    """A fully occupied periodic structure in row-vector lattice convention."""

    lattice: np.ndarray
    species: tuple[str, ...]
    frac_coords: np.ndarray
    comment: str = ""

    def __post_init__(self) -> None:
        lattice = np.asarray(self.lattice, dtype=float)
        frac = np.asarray(self.frac_coords, dtype=float)
        if lattice.shape != (3, 3):
            raise ValueError("lattice must have shape (3, 3)")
        if frac.ndim != 2 or frac.shape[1] != 3:
            raise ValueError("frac_coords must have shape (n, 3)")
        if frac.shape[0] != len(self.species):
            raise ValueError("species and coordinate lengths differ")
        if not len(self.species):
            raise ValueError("structure contains no sites")
        if not np.all(np.isfinite(lattice)) or not np.all(np.isfinite(frac)):
            raise ValueError("structure contains non-finite numbers")
        volume = float(abs(np.linalg.det(lattice)))
        if volume <= 1e-8:
            raise ValueError("lattice volume must be positive")
        object.__setattr__(self, "lattice", lattice.copy())
        object.__setattr__(self, "frac_coords", np.mod(frac, 1.0))
        object.__setattr__(self, "species", tuple(normalize_symbol(x) for x in self.species))

    @property
    def n_sites(self) -> int:
        return len(self.species)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    @property
    def cart_coords(self) -> np.ndarray:
        return self.frac_coords @ self.lattice

    def translated(self, shift: Sequence[float]) -> "PeriodicStructure":
        arr = np.asarray(shift, dtype=float)
        if arr.shape != (3,):
            raise ValueError("shift must have length three")
        return PeriodicStructure(self.lattice, self.species, self.frac_coords + arr, self.comment)

    def reordered(self, order: Sequence[int]) -> "PeriodicStructure":
        idx = np.asarray(order, dtype=int)
        if sorted(idx.tolist()) != list(range(self.n_sites)):
            raise ValueError("order must be a permutation")
        return PeriodicStructure(
            self.lattice,
            tuple(self.species[int(i)] for i in idx),
            self.frac_coords[idx],
            self.comment,
        )

    def supercell(self, repeats: Sequence[int]) -> "PeriodicStructure":
        reps = np.asarray(repeats, dtype=int)
        if reps.shape != (3,) or np.any(reps < 1):
            raise ValueError("repeats must be three positive integers")
        species: list[str] = []
        coords: list[np.ndarray] = []
        for ia in range(int(reps[0])):
            for ib in range(int(reps[1])):
                for ic in range(int(reps[2])):
                    offset = np.array([ia, ib, ic], dtype=float)
                    coords.extend((self.frac_coords + offset) / reps)
                    species.extend(self.species)
        lattice = np.diag(reps.astype(float)) @ self.lattice
        return PeriodicStructure(lattice, tuple(species), np.asarray(coords), self.comment)


def _clean_lines(text: str) -> list[str]:
    # Blank coordinate lines are invalid and therefore discarded consistently.
    return [line.strip() for line in text.replace("\ufeff", "").splitlines() if line.strip()]


def _parse_scaling(line: str, raw_lattice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = [float(x) for x in line.split()]
    if len(values) == 1:
        scale = values[0]
        if scale == 0.0:
            raise ValueError("POSCAR scale cannot be zero")
        if scale < 0.0:
            target_volume = abs(scale)
            raw_volume = abs(float(np.linalg.det(raw_lattice)))
            if raw_volume <= 1e-12:
                raise ValueError("invalid raw lattice")
            factor = (target_volume / raw_volume) ** (1.0 / 3.0)
        else:
            factor = scale
        cart_scale = np.array([factor, factor, factor], dtype=float)
        return raw_lattice * factor, cart_scale
    if len(values) == 3:
        factors = np.asarray(values, dtype=float)
        if np.any(factors <= 0.0):
            raise ValueError("three POSCAR scale factors must be positive")
        # Each factor scales the corresponding lattice vector.  Cartesian input
        # is scaled component-wise, matching VASP's three-factor convention.
        return factors[:, None] * raw_lattice, factors
    raise ValueError("POSCAR scale line must contain one or three numbers")


def parse_poscar_text(text: str, *, species_override: Sequence[str] | None = None) -> PeriodicStructure:
    """Parse a VASP 4/5 POSCAR.

    VASP-4 files do not contain element symbols.  They are accepted only when
    ``species_override`` supplies one symbol per element count, preventing silent
    chemistry guesses.
    """

    lines = _clean_lines(text)
    if len(lines) < 8:
        raise ValueError("POSCAR is too short")
    comment = lines[0]
    raw_lattice = np.asarray([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=float)
    if raw_lattice.shape != (3, 3):
        raise ValueError("invalid POSCAR lattice")
    lattice, cart_scale = _parse_scaling(lines[1], raw_lattice)

    cursor = 5
    tokens = lines[cursor].split()
    has_symbols = not all(_is_int_token(x) for x in tokens)
    if has_symbols:
        symbols = [normalize_symbol(x) for x in tokens]
        cursor += 1
        count_tokens = lines[cursor].split()
    else:
        count_tokens = tokens
        if species_override is None:
            raise ValueError("VASP-4 POSCAR needs species_override; chemistry is not guessed")
        symbols = [normalize_symbol(x) for x in species_override]
    try:
        counts = [int(x) for x in count_tokens]
    except ValueError as exc:
        raise ValueError("invalid POSCAR species counts") from exc
    if len(symbols) != len(counts) or any(x <= 0 for x in counts):
        raise ValueError("POSCAR symbols/counts mismatch or non-positive count")
    cursor += 1

    if lines[cursor].lower().startswith("s"):
        cursor += 1
    mode = lines[cursor].lower()
    if not (mode.startswith("d") or mode.startswith("c") or mode.startswith("k")):
        raise ValueError("POSCAR coordinates must be Direct or Cartesian")
    direct = mode.startswith("d")
    cursor += 1

    n_sites = sum(counts)
    if len(lines) < cursor + n_sites:
        raise ValueError("POSCAR does not contain all coordinates")
    raw_coords = []
    for line in lines[cursor : cursor + n_sites]:
        fields = line.split()
        if len(fields) < 3:
            raise ValueError("invalid POSCAR coordinate line")
        raw_coords.append([float(fields[0]), float(fields[1]), float(fields[2])])
    coords = np.asarray(raw_coords, dtype=float)
    if direct:
        frac = coords
    else:
        cart = coords * cart_scale[None, :]
        frac = cart @ np.linalg.inv(lattice)
    species = tuple(symbol for symbol, count in zip(symbols, counts) for _ in range(count))
    return PeriodicStructure(lattice=lattice, species=species, frac_coords=frac, comment=comment)


def _is_int_token(token: str) -> bool:
    try:
        int(token)
    except ValueError:
        return False
    return True


def read_poscar(path: str | Path, *, species_override: Sequence[str] | None = None) -> PeriodicStructure:
    return parse_poscar_text(Path(path).read_text(encoding="utf-8"), species_override=species_override)


def write_poscar(structure: PeriodicStructure, path: str | Path | None = None) -> str:
    """Serialize a structure using grouped VASP-5 symbols and Direct coordinates."""

    order: list[str] = []
    for symbol in structure.species:
        if symbol not in order:
            order.append(symbol)
    indices = [i for symbol in order for i, current in enumerate(structure.species) if current == symbol]
    counts = [sum(current == symbol for current in structure.species) for symbol in order]
    lines = [structure.comment or "generated by phosphate_framework_descriptor", "1.0"]
    lines.extend("  " + "  ".join(f"{x:.12f}" for x in row) for row in structure.lattice)
    lines.append("  " + "  ".join(order))
    lines.append("  " + "  ".join(str(x) for x in counts))
    lines.append("Direct")
    lines.extend("  " + "  ".join(f"{x:.12f}" for x in structure.frac_coords[i]) for i in indices)
    text = "\n".join(lines) + "\n"
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text
