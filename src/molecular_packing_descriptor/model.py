"""Data models for the molecular packing descriptor family."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DescriptorError(ValueError):
    """Raised when a structure cannot be interpreted deterministically."""


@dataclass(frozen=True)
class DescriptorConfig:
    """Fixed, transparent geometry protocol.

    ``bond_scale`` multiplies sums of covalent radii when constructing the
    periodic bond graph. ``contact_ratio_max`` is the largest accepted
    distance divided by a sum of van-der-Waals radii. A contact receives full
    closeness at or below ``contact_ratio_full`` and is linearly tapered to
    zero at ``contact_ratio_max``.
    """

    bond_scale: float = 1.18
    contact_ratio_full: float = 0.75
    contact_ratio_max: float = 1.05
    contact_ratio_min: float = 0.45
    min_spacer_bonds: int = 3
    max_image_repetitions: int = 8
    max_unique_pairs: int = 2_000_000

    def validate(self) -> None:
        if not (1.0 <= self.bond_scale <= 1.6):
            raise DescriptorError("bond_scale must be in [1.0, 1.6]")
        if not (0.2 <= self.contact_ratio_min < self.contact_ratio_full < self.contact_ratio_max <= 1.5):
            raise DescriptorError(
                "contact ratios must satisfy 0.2 <= min < full < max <= 1.5"
            )
        if self.min_spacer_bonds < 2:
            raise DescriptorError("min_spacer_bonds must be at least 2")
        if self.max_image_repetitions < 1:
            raise DescriptorError("max_image_repetitions must be positive")
        if self.max_unique_pairs < 1:
            raise DescriptorError("max_unique_pairs must be positive")


@dataclass(frozen=True)
class Structure:
    """Minimal periodic structure in Cartesian and fractional coordinates."""

    comment: str
    lattice: np.ndarray
    species: tuple[str, ...]
    frac_coords: np.ndarray
    source: Path | None = None

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    @property
    def cart_coords(self) -> np.ndarray:
        return self.frac_coords @ self.lattice

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))


@dataclass(frozen=True)
class BondEdge:
    """One canonical periodic covalent-radius edge."""

    i: int
    j: int
    shift: tuple[int, int, int]
    vector: np.ndarray
    distance: float
    normalized_distance: float


@dataclass
class Component:
    """One connected bond-graph component in the reference cell."""

    index: int
    nodes: tuple[int, ...]
    offsets: dict[int, tuple[int, int, int]]
    periodic: bool


@dataclass(frozen=True)
class Contact:
    """One canonical non-covalent inter-fragment contact image."""

    i: int
    j: int
    shift: tuple[int, int, int]
    vector: np.ndarray
    distance: float
    normalized_vdw_distance: float
    closeness: float
    component_i: int
    component_j: int
    component_shift: tuple[int, int, int]
