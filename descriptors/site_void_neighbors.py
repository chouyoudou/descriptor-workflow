#!/usr/bin/env python3
"""Periodic-neighbor, shell-selection, and ray-clearance primitives."""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Mapping, Sequence

try:
    from .site_void_atoms import COVALENT_RADII_ANGSTROM, DescriptorError, _canonical_symbol
    from .site_void_math import (
        Int3, Mat3, Vec3, _dot, _frac_to_cart,
        _inverse3, _norm, _unit, _vsub,
    )
    from .site_void_model import DescriptorConfig, NeighborImage, RadiusProjection, Structure
except ImportError:  # flat private task bundle
    from site_void_atoms import COVALENT_RADII_ANGSTROM, DescriptorError, _canonical_symbol
    from site_void_math import (
        Int3, Mat3, Vec3, _dot, _frac_to_cart,
        _inverse3, _norm, _unit, _vsub,
    )
    from site_void_model import DescriptorConfig, NeighborImage, RadiusProjection, Structure

def _radius_table(overrides: Mapping[str, float] | None = None) -> dict[str, float]:
    table = dict(COVALENT_RADII_ANGSTROM)
    if overrides:
        for raw_symbol, raw_radius in overrides.items():
            symbol = _canonical_symbol(raw_symbol)
            radius = float(raw_radius)
            if not math.isfinite(radius) or radius <= 0:
                raise DescriptorError(f"radius for {symbol} must be finite and positive")
            table[symbol] = radius
    return table


def _dual_norms(lattice_inverse: Mat3) -> Vec3:
    # Fractional component i equals cart dot column i of inverse(lattice).
    return tuple(
        math.sqrt(sum(lattice_inverse[row][column] ** 2 for row in range(3)))
        for column in range(3)
    )  # type: ignore[return-value]


def _integer_range_for_fractional_component(
    base_component: float, dual_norm: float, cutoff: float
) -> range:
    bound = cutoff * dual_norm
    lower = math.ceil(-base_component - bound - 1.0e-12)
    upper = math.floor(-base_component + bound + 1.0e-12)
    return range(lower, upper + 1)


def enumerate_neighbor_images(
    structure: Structure,
    site_index: int,
    cutoff_angstrom: float,
    radii: Mapping[str, float] | None = None,
) -> list[NeighborImage]:
    """Enumerate all atom images within a Cartesian cutoff of one site.

    Image bounds are derived from reciprocal/dual-vector norms, so the result is
    valid for skew cells and does not rely on component-wise fractional rounding.
    """

    if not 0 <= site_index < structure.atom_count:
        raise DescriptorError("site index is out of range")
    if cutoff_angstrom <= 0:
        raise DescriptorError("cutoff must be positive")
    table = _radius_table(radii)
    lattice_inverse = _inverse3(structure.lattice)
    dual_norms = _dual_norms(lattice_inverse)
    center_frac = structure.frac_coords[site_index]
    center_symbol = structure.species[site_index]
    center_radius = table[center_symbol]
    neighbors: list[NeighborImage] = []

    for atom_index, (symbol, frac) in enumerate(
        zip(structure.species, structure.frac_coords, strict=True)
    ):
        base = _vsub(frac, center_frac)
        ranges = tuple(
            _integer_range_for_fractional_component(base[i], dual_norms[i], cutoff_angstrom)
            for i in range(3)
        )
        for translation in itertools.product(*ranges):
            int_translation: Int3 = (
                int(translation[0]),
                int(translation[1]),
                int(translation[2]),
            )
            if atom_index == site_index and int_translation == (0, 0, 0):
                continue
            delta_frac = (
                base[0] + int_translation[0],
                base[1] + int_translation[1],
                base[2] + int_translation[2],
            )
            vector = _frac_to_cart(delta_frac, structure.lattice)
            distance = _norm(vector)
            if distance <= 1.0e-10 or distance > cutoff_angstrom + 1.0e-10:
                continue
            normalized = distance / (center_radius + table[symbol])
            neighbors.append(
                NeighborImage(
                    atom_index=atom_index,
                    symbol=symbol,
                    translation=int_translation,
                    vector=vector,
                    distance=distance,
                    normalized_distance=normalized,
                )
            )
    neighbors.sort(
        key=lambda row: (
            row.normalized_distance,
            row.distance,
            row.atom_index,
            row.translation,
        )
    )
    return neighbors


def select_coordination_shell(
    candidates: Sequence[NeighborImage], config: DescriptorConfig
) -> tuple[list[NeighborImage], str]:
    """Select an adaptive first shell from radius-normalized distances.

    Contacts at ``d/(r_i+r_j) <= contact_scale`` form the initial shell.  When
    fewer contacts exist, nearest images are added up to ``min_neighbors``.
    A large multiplicative gap after at least ``min_neighbors`` trims likely
    second-shell contacts, and ``max_neighbors`` is a hard deterministic cap.
    """

    if not candidates:
        raise DescriptorError("no periodic neighbors found inside the search radius")
    config.validate()
    contact_count = 0
    for row in candidates:
        if row.normalized_distance <= config.contact_scale + 1.0e-12:
            contact_count += 1
        else:
            break
    initial_count = max(config.min_neighbors, contact_count)
    initial_count = min(initial_count, len(candidates), config.max_neighbors)
    shell = list(candidates[:initial_count])
    selection = "contact-cutoff"
    if contact_count < config.min_neighbors:
        selection = "nearest-fallback"

    if len(shell) > config.min_neighbors:
        best_ratio = 1.0
        best_index: int | None = None
        for index in range(config.min_neighbors, len(shell)):
            previous = shell[index - 1].normalized_distance
            current = shell[index].normalized_distance
            ratio = current / max(previous, 1.0e-15)
            if ratio > best_ratio:
                best_ratio = ratio
                best_index = index
        if best_index is not None and best_ratio >= config.shell_gap_ratio:
            shell = shell[:best_index]
            selection += "+gap-trim"

    if not shell:
        raise DescriptorError("coordination-shell selection produced no neighbors")
    return shell, selection


def _ray_sphere_clearance(
    direction: Vec3,
    obstacles: Sequence[NeighborImage],
    radii: Mapping[str, float],
    cap: float,
    radius_scale: float,
) -> float:
    direction = _unit(direction)
    clearance = cap
    for obstacle in obstacles:
        center = obstacle.vector
        projection = _dot(direction, center)
        if projection <= 0:
            continue
        obstacle_radius = radius_scale * radii[obstacle.symbol]
        perpendicular_squared = max(0.0, _dot(center, center) - projection * projection)
        radius_squared = obstacle_radius * obstacle_radius
        if perpendicular_squared >= radius_squared:
            continue
        near_intersection = projection - math.sqrt(max(0.0, radius_squared - perpendicular_squared))
        if near_intersection > 1.0e-12:
            clearance = min(clearance, near_intersection)
    return clearance


def _median(values: Sequence[float]) -> float:
    if not values:
        raise DescriptorError("median requires at least one value")
    return float(statistics.median(values))


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _radius_projection(
    candidate: str,
    candidate_radius: float,
    shell: Sequence[NeighborImage],
    neighbor_radii: Mapping[str, float],
    effective_cavity_radius: float,
    void_axis_uniqueness: float,
    clearance_contrast: float,
) -> RadiusProjection:
    margins = [
        row.distance - neighbor_radii[row.symbol] - candidate_radius for row in shell
    ]
    mismatch = (candidate_radius - effective_cavity_radius) / max(
        effective_cavity_radius, 1.0e-12
    )
    return RadiusProjection(
        candidate=candidate,
        candidate_radius_angstrom=candidate_radius,
        mean_radial_margin_angstrom=sum(margins) / len(margins),
        minimum_radial_margin_angstrom=min(margins),
        radius_mismatch_fraction=mismatch,
        projected_size_void_coupling=(
            abs(mismatch) * void_axis_uniqueness * clearance_contrast
        ),
    )
