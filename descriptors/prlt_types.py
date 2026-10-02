"""Types, schema, POSCAR parsing, and neutral element lookup for PRLT."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "periodic-ring-layer-topology/1"
IMPLEMENTATION_VERSION = "prlt-1.0.0"
STATIC_SCOPE = (
    "Static periodic coordinates and neutral-element lookup only. The descriptor does "
    "not infer bond order, oxidation state, charge, electronic structure, energy, "
    "kinetics, morphology outside the supplied cell, or biological/application response."
)

COLUMNS = (
    "prlt_bonded_atom_fraction",
    "prlt_max_translation_rank",
    "prlt_rank2_atom_fraction",
    "prlt_ring_atom_fraction",
    "prlt_ring_count_per_atom",
    "prlt_shortest_ring_size",
    "prlt_ring_size_mean",
    "prlt_ring_size_std",
    "prlt_ring_even_fraction",
    "prlt_ring_edge_coverage",
    "prlt_shared_ring_edge_fraction",
    "prlt_ring_planarity_rms_norm_mean",
    "prlt_ring_planarity_rms_norm_max",
    "prlt_rank2_ring_fraction",
    "prlt_rank2_bond_out_of_plane_rms",
    "prlt_rank2_bond_out_of_plane_p90",
    "prlt_rank2_ring_normal_alignment_mean",
    "prlt_rank2_ring_normal_alignment_min",
    "prlt_element_assortativity_all_bonds",
    "prlt_element_assortativity_ring_bonds",
    "prlt_ring_heteroedge_fraction",
    "prlt_ring_species_diversity_mean",
    "prlt_ring_atomic_number_contrast_mean",
    "prlt_ring_radius_contrast_mean",
)

COLUMN_DEFINITIONS: dict[str, dict[str, Any]] = {
    "prlt_bonded_atom_fraction": {
        "units": "dimensionless",
        "definition": "Fraction of supplied atoms incident to at least one admitted periodic geometric bond.",
    },
    "prlt_max_translation_rank": {
        "units": "integer 0..3",
        "definition": "Maximum integer translation rank among connected components of the periodic bond graph.",
    },
    "prlt_rank2_atom_fraction": {
        "units": "dimensionless",
        "definition": "Fraction of supplied atoms belonging to periodic bond-graph components of translation rank two.",
    },
    "prlt_ring_atom_fraction": {
        "units": "dimensionless",
        "definition": "Fraction of supplied atoms represented in at least one admitted bounded zero-winding shortest-path ring.",
    },
    "prlt_ring_count_per_atom": {
        "units": "rings/atom",
        "definition": "Number of distinct admitted periodic ring instances divided by supplied atom count.",
    },
    "prlt_shortest_ring_size": {
        "units": "edges",
        "definition": "Minimum admitted ring length.",
    },
    "prlt_ring_size_mean": {
        "units": "edges",
        "definition": "Arithmetic mean admitted ring length.",
    },
    "prlt_ring_size_std": {
        "units": "edges",
        "definition": "Population standard deviation of admitted ring lengths.",
    },
    "prlt_ring_even_fraction": {
        "units": "dimensionless",
        "definition": "Fraction of admitted rings with even length.",
    },
    "prlt_ring_edge_coverage": {
        "units": "dimensionless",
        "definition": "Fraction of unique periodic quotient edges participating in at least one admitted ring.",
    },
    "prlt_shared_ring_edge_fraction": {
        "units": "dimensionless",
        "definition": "Fraction of ring-participating quotient edges occurring in at least two admitted rings.",
    },
    "prlt_ring_planarity_rms_norm_mean": {
        "units": "dimensionless",
        "definition": "Mean ring RMS distance from its least-squares plane, normalized by that ring's mean edge length.",
    },
    "prlt_ring_planarity_rms_norm_max": {
        "units": "dimensionless",
        "definition": "Maximum normalized ring RMS distance from its least-squares plane.",
    },
    "prlt_rank2_ring_fraction": {
        "units": "dimensionless",
        "definition": "Fraction of admitted rings belonging to translation-rank-two components.",
    },
    "prlt_rank2_bond_out_of_plane_rms": {
        "units": "dimensionless",
        "definition": "RMS absolute direction cosine of bonds relative to the component layer normal, over rank-two components.",
    },
    "prlt_rank2_bond_out_of_plane_p90": {
        "units": "dimensionless",
        "definition": "90th percentile absolute out-of-plane bond direction cosine over rank-two components.",
    },
    "prlt_rank2_ring_normal_alignment_mean": {
        "units": "dimensionless",
        "definition": "Mean absolute cosine between rank-two component normal and least-squares ring normal.",
    },
    "prlt_rank2_ring_normal_alignment_min": {
        "units": "dimensionless",
        "definition": "Minimum absolute cosine between rank-two component normal and least-squares ring normal.",
    },
    "prlt_element_assortativity_all_bonds": {
        "units": "dimensionless",
        "definition": "Newman categorical assortativity of element symbols on all admitted quotient edges.",
    },
    "prlt_element_assortativity_ring_bonds": {
        "units": "dimensionless",
        "definition": "Newman categorical assortativity restricted to unique quotient edges participating in admitted rings.",
    },
    "prlt_ring_heteroedge_fraction": {
        "units": "dimensionless",
        "definition": "Fraction of unique ring-participating quotient edges joining unlike element symbols.",
    },
    "prlt_ring_species_diversity_mean": {
        "units": "dimensionless",
        "definition": "Mean per-ring Simpson element diversity, one minus the sum of squared species fractions.",
    },
    "prlt_ring_atomic_number_contrast_mean": {
        "units": "dimensionless",
        "definition": "Mean |Zi-Zj|/(Zi+Zj) over unique ring-participating quotient edges.",
    },
    "prlt_ring_radius_contrast_mean": {
        "units": "dimensionless",
        "definition": "Mean |ri-rj|/(ri+rj) using neutral RDKit covalent radii over unique ring-participating quotient edges.",
    },
}


class DescriptorError(RuntimeError):
    """Base descriptor exception."""


class InputError(DescriptorError):
    """Malformed or unsupported structure input."""


class ResourceLimitError(DescriptorError):
    """Declared deterministic resource guard was exceeded."""


@dataclass(frozen=True)
class Settings:
    """Frozen geometric protocol and deterministic resource guards."""

    bond_ratio: float = 1.20
    min_distance: float = 0.45
    max_search_radius: float = 7.50
    max_atoms: int = 1500
    max_translation_images: int = 12000
    max_image_atoms: int = 800_000
    max_neighbor_candidates: int = 4_000_000
    max_edges: int = 30_000
    max_component_edges_for_rings: int = 4000
    max_degree_for_ring_search: int = 12
    max_ring_size: int = 12
    max_paths_per_edge: int = 12
    max_bfs_states_per_edge: int = 30_000
    max_ring_candidates: int = 30_000
    max_rings: int = 10_000
    distance_tolerance: float = 1.0e-8

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class StructureData:
    lattice: np.ndarray
    species: tuple[str, ...]
    fractional: np.ndarray
    name: str = "structure"

    def __post_init__(self) -> None:
        lattice = np.asarray(self.lattice, dtype=float)
        fractional = np.asarray(self.fractional, dtype=float)
        if lattice.shape != (3, 3):
            raise InputError("lattice must have shape (3, 3)")
        if not np.all(np.isfinite(lattice)):
            raise InputError("lattice contains non-finite values")
        determinant = float(np.linalg.det(lattice))
        if not math.isfinite(determinant) or abs(determinant) <= 1.0e-8:
            raise InputError("lattice is singular or nearly singular")
        if fractional.ndim != 2 or fractional.shape[1] != 3:
            raise InputError("fractional coordinates must have shape (N, 3)")
        if len(self.species) != fractional.shape[0]:
            raise InputError("species and coordinate counts differ")
        if fractional.shape[0] == 0:
            raise InputError("structure has no atoms")
        if not np.all(np.isfinite(fractional)):
            raise InputError("fractional coordinates contain non-finite values")
        canonical = tuple(canonical_symbol(value) for value in self.species)
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "species", canonical)
        object.__setattr__(self, "fractional", np.mod(fractional, 1.0))

    @property
    def atom_count(self) -> int:
        return len(self.species)


_SYMBOL_RE = re.compile(r"^[A-Z][a-z]?$", re.ASCII)


def canonical_symbol(value: Any) -> str:
    text = str(value).strip()
    if text in {"D", "T"}:
        return "H"
    if not _SYMBOL_RE.fullmatch(text):
        raise InputError(f"unsupported element symbol {text!r}; neutral symbols such as C or Fe are required")
    return text


class ElementLookup:
    """Thin, inspectable adapter over RDKit's neutral periodic table API."""

    protocol = "rdkit-periodic-table/GetRCovalent+GetAtomicNumber"

    def __init__(self) -> None:
        try:
            from rdkit import Chem  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on runtime installation.
            raise InputError("RDKit is required for neutral covalent radii and atomic numbers") from exc
        self._table = Chem.GetPeriodicTable()
        self._cache: dict[str, tuple[int, float]] = {}

    def values(self, symbol: str) -> tuple[int, float]:
        symbol = canonical_symbol(symbol)
        if symbol not in self._cache:
            try:
                atomic_number = int(self._table.GetAtomicNumber(symbol))
                radius = float(self._table.GetRcovalent(atomic_number))
            except Exception as exc:
                raise InputError(f"periodic-table lookup failed for {symbol}") from exc
            if atomic_number <= 0 or not math.isfinite(radius) or radius <= 0.0:
                raise InputError(f"invalid periodic-table lookup for {symbol}")
            self._cache[symbol] = (atomic_number, radius)
        return self._cache[symbol]

    def atomic_number(self, symbol: str) -> int:
        return self.values(symbol)[0]

    def covalent_radius(self, symbol: str) -> float:
        return self.values(symbol)[1]

    def snapshot(self, symbols: Sequence[str]) -> dict[str, dict[str, float | int]]:
        return {
            symbol: {
                "atomic_number": self.atomic_number(symbol),
                "covalent_radius_angstrom": self.covalent_radius(symbol),
            }
            for symbol in sorted(set(symbols), key=lambda item: self.atomic_number(item))
        }


def structure_from_record(record: Mapping[str, Any]) -> StructureData:
    if not isinstance(record, Mapping):
        raise InputError("JSON structure record must be an object")
    lattice = record.get("lattice")
    species = record.get("species")
    fractional = record.get("fractional_coordinates", record.get("fractional"))
    if lattice is None or species is None or fractional is None:
        raise InputError("record requires lattice, species, and fractional_coordinates")
    if not isinstance(species, Sequence) or isinstance(species, (str, bytes)):
        raise InputError("species must be a sequence")
    return StructureData(
        np.asarray(lattice, dtype=float),
        tuple(str(item) for item in species),
        np.asarray(fractional, dtype=float),
        str(record.get("id", record.get("name", "json-structure"))),
    )


def _parse_float_triplet(line: str, context: str) -> list[float]:
    fields = line.split()
    if len(fields) < 3:
        raise InputError(f"{context} requires three numeric fields")
    try:
        return [float(fields[0]), float(fields[1]), float(fields[2])]
    except ValueError as exc:
        raise InputError(f"invalid number in {context}") from exc


def parse_poscar(path_or_text: str | Path, *, is_text: bool = False) -> StructureData:
    """Parse ordinary VASP 5 POSCAR text, including selective-dynamics columns."""

    if is_text:
        text = str(path_or_text)
    else:
        text = Path(path_or_text).read_text(encoding="utf-8")
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        raise InputError("POSCAR is too short")
    name = lines[0].strip() or "POSCAR"
    try:
        scale_value = float(lines[1].split()[0])
    except (ValueError, IndexError) as exc:
        raise InputError("invalid POSCAR scale") from exc
    raw_lattice = np.asarray(
        [_parse_float_triplet(lines[index], f"lattice line {index - 1}") for index in range(2, 5)],
        dtype=float,
    )
    raw_volume = abs(float(np.linalg.det(raw_lattice)))
    if raw_volume <= 1.0e-12:
        raise InputError("POSCAR lattice is singular")
    if scale_value > 0.0:
        linear_scale = scale_value
    elif scale_value < 0.0:
        linear_scale = ((-scale_value) / raw_volume) ** (1.0 / 3.0)
    else:
        raise InputError("POSCAR scale cannot be zero")
    lattice = raw_lattice * linear_scale

    symbols = lines[5].split()
    if not symbols or all(token.lstrip("+-").isdigit() for token in symbols):
        raise InputError("VASP 4 POSCAR without an element-symbol line is unsupported")
    symbols = [canonical_symbol(token) for token in symbols]
    try:
        counts = [int(token) for token in lines[6].split()]
    except ValueError as exc:
        raise InputError("invalid POSCAR element counts") from exc
    if len(counts) != len(symbols) or any(value < 0 for value in counts):
        raise InputError("POSCAR symbols and counts differ")
    atom_count = sum(counts)
    cursor = 7
    if cursor >= len(lines):
        raise InputError("POSCAR lacks coordinate mode")
    if lines[cursor].strip().lower().startswith("s"):
        cursor += 1
    if cursor >= len(lines):
        raise InputError("POSCAR lacks coordinate mode after selective dynamics")
    mode = lines[cursor].strip().lower()
    cursor += 1
    if len(lines) < cursor + atom_count:
        raise InputError("POSCAR has fewer coordinate rows than declared atoms")
    coordinates = np.asarray(
        [_parse_float_triplet(lines[cursor + index], f"coordinate row {index + 1}") for index in range(atom_count)],
        dtype=float,
    )
    if mode.startswith("d"):
        fractional = coordinates
    elif mode.startswith("c") or mode.startswith("k"):
        cartesian = coordinates * linear_scale
        fractional = cartesian @ np.linalg.inv(lattice)
    else:
        raise InputError(f"unsupported POSCAR coordinate mode {lines[cursor - 1]!r}")
    species: list[str] = []
    for symbol, count in zip(symbols, counts, strict=True):
        species.extend([symbol] * count)
    return StructureData(lattice, tuple(species), fractional, name)
