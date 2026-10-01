"""POSCAR parsing and explicit periodic-image neighbour geometry."""

from __future__ import annotations

import math
import pathlib
from collections.abc import Iterable

try:
    from .bv_directionality_model import (
        COVALENT_RADII_ANGSTROM, DescriptorError, Int3, Mat3, NeighborImage,
        Structure, Vec3, _cart_to_frac, _cross, _det3, _frac_to_cart, _norm, _vscale, _wrap_frac,
    )
except ImportError:  # direct script sibling import
    from bv_directionality_model import (
        COVALENT_RADII_ANGSTROM, DescriptorError, Int3, Mat3, NeighborImage,
        Structure, Vec3, _cart_to_frac, _cross, _det3, _frac_to_cart, _norm, _vscale, _wrap_frac,
    )


def _clean_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_poscar(text: str) -> Structure:
    """Parse an ordered VASP 4/5 POSCAR with Direct or Cartesian coordinates."""

    lines = _clean_lines(text)
    if len(lines) < 8:
        raise DescriptorError("POSCAR is too short")
    comment = lines[0]
    try:
        scale = float(lines[1].split()[0])
    except (ValueError, IndexError) as exc:
        raise DescriptorError("invalid POSCAR scale") from exc
    if scale == 0:
        raise DescriptorError("POSCAR scale cannot be zero")
    try:
        raw_lattice: Mat3 = tuple(
            tuple(float(value) for value in lines[row].split()[:3])  # type: ignore[misc]
            for row in range(2, 5)
        )  # type: ignore[assignment]
    except (ValueError, IndexError) as exc:
        raise DescriptorError("invalid POSCAR lattice") from exc
    if any(len(vector) != 3 for vector in raw_lattice):
        raise DescriptorError("each lattice vector needs three values")

    if scale < 0:
        target_volume = abs(scale)
        raw_volume = abs(_det3(raw_lattice))
        if raw_volume <= 0:
            raise DescriptorError("invalid lattice volume")
        linear_scale = (target_volume / raw_volume) ** (1.0 / 3.0)
    else:
        linear_scale = scale
    lattice: Mat3 = tuple(_vscale(vector, linear_scale) for vector in raw_lattice)  # type: ignore[assignment]

    token_line = lines[5].split()
    vasp5 = not all(token.lstrip("+-").isdigit() for token in token_line)
    if vasp5:
        symbols = tuple(token_line)
        count_line_index = 6
    else:
        symbols = tuple(f"X{index + 1}" for index in range(len(token_line)))
        count_line_index = 5
    try:
        counts = tuple(int(token) for token in lines[count_line_index].split())
    except ValueError as exc:
        raise DescriptorError("invalid POSCAR atom counts") from exc
    if len(counts) != len(symbols) or any(count <= 0 for count in counts):
        raise DescriptorError("POSCAR symbols/counts mismatch or non-positive count")
    if not vasp5:
        raise DescriptorError("VASP 4 POSCAR requires explicit element symbols; use VASP 5 format")
    unknown = sorted({symbol for symbol in symbols if symbol not in COVALENT_RADII_ANGSTROM})
    if unknown:
        raise DescriptorError(f"unsupported POSCAR symbols: {', '.join(unknown)}")

    cursor = count_line_index + 1
    if cursor >= len(lines):
        raise DescriptorError("POSCAR has no coordinate mode")
    if lines[cursor].lower().startswith("s"):
        cursor += 1
    if cursor >= len(lines):
        raise DescriptorError("POSCAR has no coordinate mode")
    mode = lines[cursor].lower()
    direct = mode.startswith("d")
    cartesian = mode.startswith("c") or mode.startswith("k")
    if not (direct or cartesian):
        raise DescriptorError("POSCAR coordinate mode must be Direct or Cartesian")
    cursor += 1

    atom_count = sum(counts)
    if len(lines) < cursor + atom_count:
        raise DescriptorError("POSCAR has fewer coordinates than declared atoms")
    species: list[str] = []
    for symbol, count in zip(symbols, counts, strict=True):
        species.extend([symbol] * count)
    frac_coords: list[Vec3] = []
    for atom_index in range(atom_count):
        tokens = lines[cursor + atom_index].split()
        try:
            values: Vec3 = tuple(float(token) for token in tokens[:3])  # type: ignore[assignment]
        except ValueError as exc:
            raise DescriptorError(f"invalid coordinate at atom {atom_index}") from exc
        if len(values) != 3:
            raise DescriptorError(f"coordinate {atom_index} needs three values")
        if direct:
            frac = values
        else:
            # Positive POSCAR scale multiplies Cartesian coordinates as well.
            cart = _vscale(values, linear_scale)
            frac = _cart_to_frac(cart, lattice)
        frac_coords.append(_wrap_frac(frac))
    return Structure(lattice=lattice, species=tuple(species), frac_coords=tuple(frac_coords), comment=comment)


def read_poscar(path: str | pathlib.Path) -> Structure:
    return parse_poscar(pathlib.Path(path).read_text(encoding="utf-8"))


def _translation_bounds(lattice: Mat3, cutoff_angstrom: float) -> Int3:
    volume = abs(_det3(lattice))
    heights = (
        volume / max(_norm(_cross(lattice[1], lattice[2])), 1e-15),
        volume / max(_norm(_cross(lattice[2], lattice[0])), 1e-15),
        volume / max(_norm(_cross(lattice[0], lattice[1])), 1e-15),
    )
    return tuple(max(1, int(math.ceil(cutoff_angstrom / height)) + 1) for height in heights)  # type: ignore[return-value]


def _integer_translations(bounds: Int3) -> Iterable[Int3]:
    for tx in range(-bounds[0], bounds[0] + 1):
        for ty in range(-bounds[1], bounds[1] + 1):
            for tz in range(-bounds[2], bounds[2] + 1):
                yield (tx, ty, tz)


def _neighbor_images(
    structure: Structure,
    center_index: int,
    cutoff_angstrom: float,
) -> list[NeighborImage]:
    center_frac = structure.frac_coords[center_index]
    center_radius = COVALENT_RADII_ANGSTROM[structure.species[center_index]]
    bounds = _translation_bounds(structure.lattice, cutoff_angstrom)
    output: list[NeighborImage] = []
    for target_index, target_frac in enumerate(structure.frac_coords):
        target_radius = COVALENT_RADII_ANGSTROM[structure.species[target_index]]
        radius_sum = center_radius + target_radius
        for translation in _integer_translations(bounds):
            if target_index == center_index and translation == (0, 0, 0):
                continue
            delta_frac = (
                target_frac[0] + translation[0] - center_frac[0],
                target_frac[1] + translation[1] - center_frac[1],
                target_frac[2] + translation[2] - center_frac[2],
            )
            vector = _frac_to_cart(delta_frac, structure.lattice)
            distance = _norm(vector)
            if 1e-10 < distance <= cutoff_angstrom + 1e-10:
                output.append(
                    NeighborImage(
                        site_index=target_index,
                        translation=translation,
                        vector_cart=vector,
                        distance_angstrom=distance,
                        normalized_distance=distance / radius_sum,
                        radius_angstrom=target_radius,
                    )
                )
    output.sort(
        key=lambda item: (
            item.normalized_distance,
            item.distance_angstrom,
            item.site_index,
            item.translation,
        )
    )
    return output


def _point_clearance(
    structure: Structure,
    point_cart: Vec3,
    *,
    excluded_site_index: int | None,
    search_radius_angstrom: float,
) -> tuple[float, int | None]:
    point_frac = _cart_to_frac(point_cart, structure.lattice)
    bounds = _translation_bounds(structure.lattice, search_radius_angstrom)
    best = math.inf
    best_index: int | None = None
    for target_index, target_frac in enumerate(structure.frac_coords):
        radius = COVALENT_RADII_ANGSTROM[structure.species[target_index]]
        for translation in _integer_translations(bounds):
            if target_index == excluded_site_index and translation == (0, 0, 0):
                continue
            delta_frac = (
                target_frac[0] + translation[0] - point_frac[0],
                target_frac[1] + translation[1] - point_frac[1],
                target_frac[2] + translation[2] - point_frac[2],
            )
            distance = _norm(_frac_to_cart(delta_frac, structure.lattice))
            if distance <= search_radius_angstrom + radius:
                surface_clearance = distance - radius
                if surface_clearance < best:
                    best = surface_clearance
                    best_index = target_index
    if math.isinf(best):
        return (search_radius_angstrom, None)
    return (best, best_index)
