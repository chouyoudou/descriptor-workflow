#!/usr/bin/env python3
"""POSCAR and workflow-JSON adapters for site--void coupling."""

from __future__ import annotations

import math
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from .site_void_atoms import DescriptorError, _canonical_symbol
    from .site_void_math import (
        Mat3, Vec3, _cart_to_frac, _det3, _inverse3, _vscale, _wrap_frac,
    )
    from .site_void_model import Structure
except ImportError:  # flat private task bundle
    from site_void_atoms import DescriptorError, _canonical_symbol
    from site_void_math import (
        Mat3, Vec3, _cart_to_frac, _det3, _inverse3, _vscale, _wrap_frac,
    )
    from site_void_model import Structure

def _parse_float_triplet(line: str, context: str) -> Vec3:
    fields = line.split()
    if len(fields) < 3:
        raise DescriptorError(f"{context} must contain three numbers")
    try:
        values = tuple(float(field) for field in fields[:3])
    except ValueError as exc:
        raise DescriptorError(f"invalid number in {context}") from exc
    if not all(math.isfinite(value) for value in values):
        raise DescriptorError(f"non-finite number in {context}")
    return values  # type: ignore[return-value]


def parse_poscar(text: str) -> Structure:
    """Parse an ordinary VASP 5-style POSCAR string.

    Supported: one positive scale factor, the negative target-volume convention,
    element and count lines, optional Selective Dynamics, and Direct or Cartesian
    coordinates.  VASP 4 files without element symbols are rejected because an
    element lookup would otherwise be guessed.
    """

    raw_lines = text.splitlines()
    if len(raw_lines) < 8:
        raise DescriptorError("POSCAR is too short")
    comment = raw_lines[0].strip()
    lines = [line.strip() for line in raw_lines[1:] if line.strip()]
    if len(lines) < 7:
        raise DescriptorError("POSCAR is incomplete")

    scale_fields = lines[0].split()
    if len(scale_fields) != 1:
        raise DescriptorError("only the ordinary one-value POSCAR scale is supported")
    try:
        requested_scale = float(scale_fields[0])
    except ValueError as exc:
        raise DescriptorError("invalid POSCAR scale") from exc
    if not math.isfinite(requested_scale) or requested_scale == 0:
        raise DescriptorError("POSCAR scale must be finite and nonzero")

    raw_lattice: Mat3 = (
        _parse_float_triplet(lines[1], "lattice vector 1"),
        _parse_float_triplet(lines[2], "lattice vector 2"),
        _parse_float_triplet(lines[3], "lattice vector 3"),
    )
    raw_volume = abs(_det3(raw_lattice))
    if raw_volume <= 1.0e-15:
        raise DescriptorError("POSCAR lattice is singular")
    if requested_scale > 0:
        length_scale = requested_scale
    else:
        length_scale = (abs(requested_scale) / raw_volume) ** (1.0 / 3.0)
    lattice: Mat3 = tuple(
        tuple(length_scale * component for component in vector) for vector in raw_lattice
    )  # type: ignore[assignment]

    symbol_fields = lines[4].split()
    try:
        [int(field) for field in symbol_fields]
    except ValueError:
        symbols = tuple(_canonical_symbol(field) for field in symbol_fields)
    else:
        raise DescriptorError(
            "VASP 4 POSCAR without an element-symbol line is unsupported; "
            "supply a VASP 5-style POSCAR"
        )

    try:
        counts = tuple(int(field) for field in lines[5].split())
    except ValueError as exc:
        raise DescriptorError("invalid POSCAR atom counts") from exc
    if len(counts) != len(symbols) or any(count <= 0 for count in counts):
        raise DescriptorError("element and positive count lines must have equal length")
    atom_count = sum(counts)

    cursor = 6
    if lines[cursor].lower().startswith("s"):
        cursor += 1
    if cursor >= len(lines):
        raise DescriptorError("missing POSCAR coordinate mode")
    coordinate_mode = lines[cursor].lower()
    cursor += 1
    if not coordinate_mode or coordinate_mode[0] not in {"d", "c", "k"}:
        raise DescriptorError("coordinate mode must be Direct or Cartesian")
    if len(lines) < cursor + atom_count:
        raise DescriptorError("fewer coordinate rows than atom counts require")

    lattice_inverse = _inverse3(lattice)
    frac_coords: list[Vec3] = []
    for atom_index in range(atom_count):
        raw = _parse_float_triplet(lines[cursor + atom_index], f"coordinate {atom_index}")
        if coordinate_mode[0] == "d":
            frac = raw
        else:
            # VASP applies the universal scale to Cartesian coordinates too.
            cart = _vscale(raw, length_scale)
            frac = _cart_to_frac(cart, lattice_inverse)
        frac_coords.append(_wrap_frac(frac))

    species: list[str] = []
    for symbol, count in zip(symbols, counts, strict=True):
        species.extend([symbol] * count)
    return Structure(
        lattice=lattice,
        species=tuple(species),
        frac_coords=tuple(frac_coords),
        comment=comment,
    )


def read_poscar(path: str | pathlib.Path) -> Structure:
    return parse_poscar(pathlib.Path(path).read_text(encoding="utf-8"))


def structure_from_json_record(record: Mapping[str, Any]) -> Structure:
    """Build a :class:`Structure` from the workflow JSONL structure schema.

    The transport schema uses ``lattice`` (row vectors), ``species``, and
    ``fractional_coordinates``.  ``frac_coords`` is accepted as a compact alias
    for callers outside the workflow.  No oxidation state, occupancy, or target
    property is inferred.
    """

    try:
        raw_lattice = record["lattice"]
        raw_species = record["species"]
        raw_coords = record.get("fractional_coordinates", record.get("frac_coords"))
    except (KeyError, TypeError) as exc:
        raise DescriptorError("JSON structure record lacks lattice/species/coordinates") from exc
    if raw_coords is None:
        raise DescriptorError("JSON structure record lacks fractional_coordinates")
    if not isinstance(raw_lattice, Sequence) or len(raw_lattice) != 3:
        raise DescriptorError("JSON lattice must contain three row vectors")
    if not isinstance(raw_species, Sequence) or isinstance(raw_species, (str, bytes)):
        raise DescriptorError("JSON species must be a sequence")
    if not isinstance(raw_coords, Sequence) or isinstance(raw_coords, (str, bytes)):
        raise DescriptorError("JSON fractional coordinates must be a sequence")

    def triple(value: Any, label: str) -> Vec3:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
            raise DescriptorError(f"{label} must be a numeric triple")
        try:
            result = (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError) as exc:
            raise DescriptorError(f"{label} must be a numeric triple") from exc
        if not all(math.isfinite(x) for x in result):
            raise DescriptorError(f"{label} must contain finite numbers")
        return result

    lattice = tuple(triple(row, "JSON lattice row") for row in raw_lattice)
    coords = tuple(triple(row, "JSON fractional coordinate") for row in raw_coords)
    species = tuple(_canonical_symbol(str(symbol)) for symbol in raw_species)
    comment = str(record.get("id", record.get("comment", "")))
    return Structure(
        lattice=lattice,  # type: ignore[arg-type]
        species=species,
        frac_coords=coords,
        comment=comment,
    )
