"""Input, lookup, and POSCAR contracts for PIAN."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

TABLE_SCHEMA = "magpie-neutral-atom-ie1-ea/1"
SHELL_FACTOR = 1.25
DISTANCE_TOLERANCE_ANGSTROM = 1.0e-9
MIN_SEPARATION_ANGSTROM = 0.10
MAX_ATOMS = 512
MAX_FIRST_SHELL_EDGES = 500_000
DEFAULT_TABLE = Path(__file__).with_name("neutral_atom_ie_ea_magpie_v1.csv")


class DescriptorUnavailable(RuntimeError):
    """Typed non-result, never silently replaced by zero."""

    def __init__(self, category: str, code: str, message: str):
        super().__init__(message)
        self.category = category
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {"category": self.category, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class StructureData:
    lattice: np.ndarray
    species: tuple[str, ...]
    fractional_coordinates: np.ndarray


@dataclass(frozen=True)
class ElementRecord:
    atomic_number: int
    symbol: str
    first_ionization_energy_ev: float | None
    electron_affinity_ev: float | None
    ie1_source_state: str
    ea_source_state: str


def _finite_array(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise DescriptorUnavailable("input", f"invalid_{name}", f"{name} must be finite with shape {shape}")
    return array


def normalize_structure(lattice: Any, species: Sequence[Any], fractional_coordinates: Any) -> StructureData:
    lattice_array = _finite_array(lattice, shape=(3, 3), name="lattice")
    determinant = float(np.linalg.det(lattice_array))
    if not math.isfinite(determinant) or abs(determinant) <= 1.0e-10:
        raise DescriptorUnavailable("input", "singular_lattice", "lattice volume is zero or non-finite")
    species_tuple = tuple(str(item).strip() for item in species)
    if not species_tuple or any(not item for item in species_tuple):
        raise DescriptorUnavailable("input", "invalid_species", "species must contain non-empty element symbols")
    if len(species_tuple) > MAX_ATOMS:
        raise DescriptorUnavailable("resource", "atom_limit", f"structure has {len(species_tuple)} sites; limit is {MAX_ATOMS}")
    fractions = _finite_array(fractional_coordinates, shape=(len(species_tuple), 3), name="fractional_coordinates")
    return StructureData(lattice_array.copy(), species_tuple, np.mod(fractions, 1.0))


def structure_from_record(record: Mapping[str, Any]) -> StructureData:
    try:
        return normalize_structure(record["lattice"], record["species"], record["fractional_coordinates"])
    except KeyError as exc:
        raise DescriptorUnavailable("input", "missing_structure_field", "record must contain lattice, species, and fractional_coordinates") from exc


def _parse_float_triplet(line: str, *, name: str) -> list[float]:
    fields = line.split()
    if len(fields) < 3:
        raise DescriptorUnavailable("input", "invalid_poscar", f"{name} requires three numbers")
    try:
        values = [float(fields[i]) for i in range(3)]
    except ValueError as exc:
        raise DescriptorUnavailable("input", "invalid_poscar", f"{name} contains a non-number") from exc
    if not all(math.isfinite(x) for x in values):
        raise DescriptorUnavailable("input", "invalid_poscar", f"{name} contains a non-finite number")
    return values


def parse_poscar_text(text: str, *, vasp4_symbols: Sequence[str] | None = None) -> StructureData:
    raw_lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(raw_lines) < 8:
        raise DescriptorUnavailable("input", "invalid_poscar", "POSCAR is too short")
    try:
        scale = float(raw_lines[1].split()[0])
    except (ValueError, IndexError) as exc:
        raise DescriptorUnavailable("input", "invalid_poscar", "invalid POSCAR scale") from exc
    if not math.isfinite(scale) or scale == 0.0:
        raise DescriptorUnavailable("input", "invalid_poscar", "POSCAR scale must be finite and non-zero")
    unscaled = np.asarray([_parse_float_triplet(raw_lines[i], name=f"lattice row {i-1}") for i in range(2, 5)], dtype=np.float64)
    if scale > 0:
        effective_scale = scale
    else:
        raw_volume = abs(float(np.linalg.det(unscaled)))
        if raw_volume <= 1.0e-14:
            raise DescriptorUnavailable("input", "singular_lattice", "negative scale used with singular lattice")
        effective_scale = (abs(scale) / raw_volume) ** (1.0 / 3.0)
    lattice = unscaled * effective_scale
    line5 = raw_lines[5].split()
    if not line5:
        raise DescriptorUnavailable("input", "invalid_poscar", "missing species/count line")
    is_vasp4 = all(token.lstrip("+-").isdigit() for token in line5)
    if is_vasp4:
        if vasp4_symbols is None:
            raise DescriptorUnavailable("input", "vasp4_symbols_required", "VASP4 POSCAR requires an explicit ordered symbol override")
        symbols = [str(x) for x in vasp4_symbols]
        count_tokens = line5
        cursor = 6
    else:
        symbols = line5
        if len(raw_lines) <= 6:
            raise DescriptorUnavailable("input", "invalid_poscar", "missing POSCAR counts")
        count_tokens = raw_lines[6].split()
        cursor = 7
    try:
        counts = [int(x) for x in count_tokens]
    except ValueError as exc:
        raise DescriptorUnavailable("input", "invalid_poscar", "site counts must be integers") from exc
    if len(symbols) != len(counts) or any(count <= 0 for count in counts):
        raise DescriptorUnavailable("input", "invalid_poscar", "species and positive site counts do not match")
    site_count = sum(counts)
    if cursor >= len(raw_lines):
        raise DescriptorUnavailable("input", "invalid_poscar", "missing coordinate mode")
    if raw_lines[cursor].strip().lower().startswith("s"):
        cursor += 1
        if cursor >= len(raw_lines):
            raise DescriptorUnavailable("input", "invalid_poscar", "missing coordinate mode after selective dynamics")
    mode = raw_lines[cursor].strip().lower()
    cursor += 1
    if cursor + site_count > len(raw_lines):
        raise DescriptorUnavailable("input", "invalid_poscar", "not enough site-coordinate lines")
    coords = np.asarray([_parse_float_triplet(raw_lines[cursor+i], name=f"site coordinate {i}") for i in range(site_count)], dtype=np.float64)
    if mode.startswith("d"):
        fractions = coords
    elif mode.startswith("c") or mode.startswith("k"):
        try:
            fractions = (coords * effective_scale) @ np.linalg.inv(lattice)
        except np.linalg.LinAlgError as exc:
            raise DescriptorUnavailable("input", "singular_lattice", "cannot invert POSCAR lattice") from exc
    else:
        raise DescriptorUnavailable("input", "invalid_poscar", f"unknown coordinate mode: {raw_lines[cursor-1]}")
    expanded = tuple(symbol for symbol, count in zip(symbols, counts) for _ in range(count))
    return normalize_structure(lattice, expanded, fractions)


def structure_from_poscar(path: str | Path, *, vasp4_symbols: Sequence[str] | None = None) -> StructureData:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise DescriptorUnavailable("input", "poscar_read_failed", f"could not read POSCAR: {exc}") from exc
    return parse_poscar_text(text, vasp4_symbols=vasp4_symbols)


def _canonical_table_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


@lru_cache(maxsize=16)
def _load_element_table_cached(canonical_path: str) -> dict[str, ElementRecord]:
    source = Path(canonical_path)
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise DescriptorUnavailable("lookup", "element_table_read_failed", f"could not read element table: {exc}") from exc
    required = {"atomic_number", "symbol", "first_ionization_energy_eV", "electron_affinity_eV", "ie1_source_state", "ea_source_state"}
    if not rows or not required.issubset(rows[0]):
        raise DescriptorUnavailable("lookup", "invalid_element_table", "element table schema mismatch")
    records: dict[str, ElementRecord] = {}
    for row in rows:
        symbol = row["symbol"].strip()
        try:
            atomic_number = int(row["atomic_number"])
            def optional_float(value: str) -> float | None:
                if not value.strip():
                    return None
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("non-finite")
                return number
            ie1 = optional_float(row["first_ionization_energy_eV"])
            ea = optional_float(row["electron_affinity_eV"])
        except ValueError as exc:
            raise DescriptorUnavailable("lookup", "invalid_element_table", f"invalid table row for {symbol}") from exc
        if symbol in records:
            raise DescriptorUnavailable("lookup", "invalid_element_table", f"duplicate symbol {symbol}")
        records[symbol] = ElementRecord(atomic_number, symbol, ie1, ea, row["ie1_source_state"].strip(), row["ea_source_state"].strip())
    return records


def load_element_table(path: str | Path = DEFAULT_TABLE) -> dict[str, ElementRecord]:
    """Load a frozen table once per process; mapping is read-only by contract."""
    return _load_element_table_cached(_canonical_table_path(path))


@lru_cache(maxsize=16)
def table_sha256(canonical_path: str) -> str:
    digest = hashlib.sha256()
    with Path(canonical_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
