#!/usr/bin/env python3
"""Validated structure/configuration and output records for site--void coupling."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

try:
    from .site_void_atoms import DescriptorError, _canonical_symbol
    from .site_void_math import Mat3, Int3, Vec3, _det3, _frac_to_cart, _vadd, _wrap_frac
except ImportError:  # flat private task bundle
    from site_void_atoms import DescriptorError, _canonical_symbol
    from site_void_math import Mat3, Int3, Vec3, _det3, _frac_to_cart, _vadd, _wrap_frac

@dataclass(frozen=True)
class Structure:
    """Minimal periodic structure with lattice vectors stored as rows."""

    lattice: Mat3
    species: tuple[str, ...]
    frac_coords: tuple[Vec3, ...]
    comment: str = ""

    def __post_init__(self) -> None:
        if len(self.species) != len(self.frac_coords):
            raise DescriptorError("species and coordinate counts differ")
        if not self.species:
            raise DescriptorError("structure contains no atoms")
        volume = abs(_det3(self.lattice))
        if not math.isfinite(volume) or volume <= 1.0e-10:
            raise DescriptorError("lattice is singular or has zero volume")
        for symbol in self.species:
            _canonical_symbol(symbol)
        for coord in self.frac_coords:
            if len(coord) != 3 or not all(math.isfinite(x) for x in coord):
                raise DescriptorError("fractional coordinates must be finite triples")

    @property
    def atom_count(self) -> int:
        return len(self.species)

    @property
    def volume_angstrom3(self) -> float:
        return abs(_det3(self.lattice))

    def cart_coords(self) -> tuple[Vec3, ...]:
        return tuple(_frac_to_cart(frac, self.lattice) for frac in self.frac_coords)

    def translated(self, delta_frac: Vec3) -> "Structure":
        return Structure(
            lattice=self.lattice,
            species=self.species,
            frac_coords=tuple(_wrap_frac(_vadd(x, delta_frac)) for x in self.frac_coords),
            comment=self.comment,
        )


@dataclass(frozen=True)
class DescriptorConfig:
    """Protocol-defining parameters for deterministic descriptor evaluation."""

    search_radius_angstrom: float = 8.0
    contact_scale: float = 1.35
    shell_gap_ratio: float = 1.18
    min_neighbors: int = 4
    max_neighbors: int = 16
    clearance_cap_angstrom: float = 5.0
    obstacle_radius_scale: float = 1.0
    minimum_clearance_contrast: float = 1.0e-12

    def validate(self) -> None:
        if self.search_radius_angstrom <= 0:
            raise DescriptorError("search radius must be positive")
        if self.contact_scale <= 0:
            raise DescriptorError("contact scale must be positive")
        if self.shell_gap_ratio < 1:
            raise DescriptorError("shell gap ratio must be at least 1")
        if self.min_neighbors < 1:
            raise DescriptorError("min_neighbors must be positive")
        if self.max_neighbors < self.min_neighbors:
            raise DescriptorError("max_neighbors must be >= min_neighbors")
        if self.clearance_cap_angstrom <= 0:
            raise DescriptorError("clearance cap must be positive")
        if self.obstacle_radius_scale <= 0:
            raise DescriptorError("obstacle radius scale must be positive")
        if self.search_radius_angstrom <= self.clearance_cap_angstrom:
            raise DescriptorError(
                "search radius must exceed clearance cap so ray blockers are observable"
            )


@dataclass(frozen=True)
class NeighborImage:
    atom_index: int
    symbol: str
    translation: Int3
    vector: Vec3
    distance: float
    normalized_distance: float


@dataclass(frozen=True)
class RadiusProjection:
    candidate: str
    candidate_radius_angstrom: float
    mean_radial_margin_angstrom: float
    minimum_radial_margin_angstrom: float
    radius_mismatch_fraction: float
    projected_size_void_coupling: float


@dataclass(frozen=True)
class SiteDescriptor:
    site_index: int
    symbol: str
    fractional_coordinate: Vec3
    neighbor_count: int
    neighbor_indices: tuple[int, ...]
    neighbor_symbols: tuple[str, ...]
    neighbor_translations: tuple[Int3, ...]
    neighbor_distances_angstrom: tuple[float, ...]
    normalized_neighbor_distances: tuple[float, ...]
    mean_neighbor_distance_angstrom: float
    shell_cutoff_normalized_distance: float
    shell_selection: str
    offcenter_vector_angstrom: Vec3
    offcenter_angstrom: float
    offcenter_fraction: float
    blocking_tensor_eigenvalues: Vec3
    void_axis: Vec3
    angular_void_anisotropy: float
    void_axis_uniqueness: float
    forward_clearance_angstrom: float
    backward_clearance_angstrom: float
    clearance_contrast: float
    offcenter_void_alignment: float
    site_void_coupling_signed: float
    site_void_coupling_positive: float
    coupling_support: float
    center_covalent_radius_angstrom: float
    median_neighbor_covalent_radius_angstrom: float
    center_to_neighbor_radius_ratio: float
    locally_large_radius_site: bool
    effective_cavity_radius_angstrom: float
    mean_radial_margin_angstrom: float
    minimum_radial_margin_angstrom: float
    radius_mismatch_fraction: float
    size_void_coupling: float
    candidate_radius_projections: tuple[RadiusProjection, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["candidate_radius_projections"] = {
            row.candidate: dataclasses.asdict(row)
            for row in self.candidate_radius_projections
        }
        return data
