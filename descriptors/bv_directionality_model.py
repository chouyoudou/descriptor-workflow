"""Data contracts, radius table, and dependency-free vector helpers."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Final

SCHEMA: Final[str] = "bond-valence-directionality/1"
BENCHMARK_SCHEMA: Final[str] = "bond-valence-directionality-benchmark/1"
EXPLICIT_PROTOCOL_SCHEMA: Final[str] = "explicit-bond-valence-protocol/1"
INTERPRETATION_SCOPE: Final[str] = (
    "Static structure descriptors only. Radius-contact quantities are not BVS/BVSM/BVSE. "
    "Bond-valence quantities require explicit oxidation states and pair parameters. "
    "No output is electron density, a true lone-pair position, a migration barrier, "
    "carrier concentration, ionic conductivity, or a relaxed response."
)

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]
Int3 = tuple[int, int, int]


class DescriptorError(ValueError):
    """Raised for invalid structures or protocols."""


# Cordero-style single-bond covalent radii in angstrom.  Values are numerical
# reference data, not copied executable code.  See the example README for the
# source and limitations.  The superheavy-element values are conservative
# tabulated estimates and are reported through the same fixed protocol.
_ELEMENT_SYMBOLS: Final[tuple[str, ...]] = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
_COVALENT_RADIUS_VALUES: Final[tuple[float, ...]] = (
    0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76,
    1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22,
    1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75,
    1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39,
    1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01,
    1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87,
    1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32,
    1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06,
    2.00, 1.96, 1.90, 1.87, 1.80, 1.69, 1.68, 1.68, 1.65, 1.67,
    1.73, 1.76, 1.61, 1.57, 1.49, 1.43, 1.41, 1.34, 1.29, 1.28,
    1.21, 1.22, 1.36, 1.43, 1.62, 1.75, 1.65, 1.57,
)
COVALENT_RADII_ANGSTROM: Final[dict[str, float]] = dict(
    zip(_ELEMENT_SYMBOLS, _COVALENT_RADIUS_VALUES, strict=True)
)
ATOMIC_NUMBERS: Final[dict[str, int]] = {
    symbol: index + 1 for index, symbol in enumerate(_ELEMENT_SYMBOLS)
}


def _vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vscale(a: Vec3, scalar: float) -> Vec3:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Vec3) -> Vec3:
    length = _norm(a)
    if length <= 0:
        raise DescriptorError("cannot normalize a zero vector")
    return _vscale(a, 1.0 / length)


def _det3(matrix: Mat3) -> float:
    return _dot(matrix[0], _cross(matrix[1], matrix[2]))


def _inverse3(matrix: Mat3) -> Mat3:
    determinant = _det3(matrix)
    if abs(determinant) < 1e-14:
        raise DescriptorError("lattice is singular")
    a, b, c = matrix
    # For a row-vector lattice, inverse columns are reciprocal dual vectors.
    bc = _cross(b, c)
    ca = _cross(c, a)
    ab = _cross(a, b)
    inv_det = 1.0 / determinant
    return (
        (bc[0] * inv_det, ca[0] * inv_det, ab[0] * inv_det),
        (bc[1] * inv_det, ca[1] * inv_det, ab[1] * inv_det),
        (bc[2] * inv_det, ca[2] * inv_det, ab[2] * inv_det),
    )


def _frac_to_cart(frac: Vec3, lattice: Mat3) -> Vec3:
    return (
        frac[0] * lattice[0][0] + frac[1] * lattice[1][0] + frac[2] * lattice[2][0],
        frac[0] * lattice[0][1] + frac[1] * lattice[1][1] + frac[2] * lattice[2][1],
        frac[0] * lattice[0][2] + frac[1] * lattice[1][2] + frac[2] * lattice[2][2],
    )


def _cart_to_frac(cart: Vec3, lattice: Mat3) -> Vec3:
    inverse = _inverse3(lattice)
    return (
        cart[0] * inverse[0][0] + cart[1] * inverse[1][0] + cart[2] * inverse[2][0],
        cart[0] * inverse[0][1] + cart[1] * inverse[1][1] + cart[2] * inverse[2][1],
        cart[0] * inverse[0][2] + cart[1] * inverse[1][2] + cart[2] * inverse[2][2],
    )


def _wrap_frac(frac: Vec3) -> Vec3:
    return tuple(value - math.floor(value) for value in frac)  # type: ignore[return-value]


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    if not 0 <= fraction <= 1:
        raise DescriptorError("percentile fraction must be in [0, 1]")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    mean_x = _mean(x_values)
    mean_y = _mean(y_values)
    dx = [value - mean_x for value in x_values]
    dy = [value - mean_y for value in y_values]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denominator <= 1e-15:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / denominator


@dataclasses.dataclass(frozen=True)
class Structure:
    lattice: Mat3
    species: tuple[str, ...]
    frac_coords: tuple[Vec3, ...]
    comment: str = ""

    def __post_init__(self) -> None:
        if len(self.species) == 0:
            raise DescriptorError("structure contains no atoms")
        if len(self.species) != len(self.frac_coords):
            raise DescriptorError("species and coordinate counts differ")
        if abs(_det3(self.lattice)) < 1e-10:
            raise DescriptorError("lattice is singular")
        unknown = sorted({symbol for symbol in self.species if symbol not in COVALENT_RADII_ANGSTROM})
        if unknown:
            raise DescriptorError(f"unsupported element symbols: {', '.join(unknown)}")


@dataclasses.dataclass(frozen=True)
class ContactConfig:
    contact_cutoff_ratio: float = 1.45
    contact_softness_ratio: float = 0.20
    minimum_neighbors: int = 3
    maximum_neighbors: int = 20
    fallback_search_radius_angstrom: float = 8.0
    virtual_offset_angstrom: float = 1.0
    virtual_clearance_cap_angstrom: float = 2.0
    vector_epsilon: float = 1e-12

    def validate(self) -> None:
        if self.contact_cutoff_ratio <= 1.0:
            raise DescriptorError("contact_cutoff_ratio must exceed 1")
        if self.contact_softness_ratio <= 0:
            raise DescriptorError("contact_softness_ratio must be positive")
        if self.minimum_neighbors < 1:
            raise DescriptorError("minimum_neighbors must be positive")
        if self.maximum_neighbors < self.minimum_neighbors:
            raise DescriptorError("maximum_neighbors must be >= minimum_neighbors")
        if self.fallback_search_radius_angstrom <= 0:
            raise DescriptorError("fallback search radius must be positive")
        if self.virtual_offset_angstrom <= 0:
            raise DescriptorError("virtual offset must be positive")
        if self.virtual_clearance_cap_angstrom <= 0:
            raise DescriptorError("virtual clearance cap must be positive")


@dataclasses.dataclass(frozen=True)
class PairParameter:
    cation: str
    cation_oxidation: int
    anion: str
    anion_oxidation: int
    r0_angstrom: float
    b_angstrom: float
    source: str

    def __post_init__(self) -> None:
        if self.cation not in COVALENT_RADII_ANGSTROM or self.anion not in COVALENT_RADII_ANGSTROM:
            raise DescriptorError("bond-valence pair contains an unknown element")
        if self.cation_oxidation <= 0 or self.anion_oxidation >= 0:
            raise DescriptorError("bond-valence pair must name positive cation and negative anion states")
        if self.r0_angstrom <= 0 or self.b_angstrom <= 0:
            raise DescriptorError("R0 and B must be positive")
        if not self.source.strip():
            raise DescriptorError("each pair parameter requires a non-empty source string")

    @property
    def key(self) -> tuple[str, int, str, int]:
        return (self.cation, self.cation_oxidation, self.anion, self.anion_oxidation)


@dataclasses.dataclass(frozen=True)
class ExplicitBondValenceProtocol:
    site_oxidation_states: tuple[int, ...]
    pair_parameters: tuple[PairParameter, ...]
    candidate_site_indices: tuple[int, ...] = ()
    minimum_bond_valence: float = 1e-4
    maximum_distance_angstrom: float = 8.0
    vector_cutoff: float = 0.5
    dummy_offset_angstrom: float = 1.0
    note: str = ""

    def validate(self, structure: Structure) -> None:
        if len(self.site_oxidation_states) != len(structure.species):
            raise DescriptorError("site_oxidation_states must contain one integer per atom")
        if any(state == 0 for state in self.site_oxidation_states):
            raise DescriptorError("zero oxidation states are not allowed in the explicit BVS layer")
        if not 0 < self.minimum_bond_valence < 1:
            raise DescriptorError("minimum_bond_valence must lie between 0 and 1")
        if self.maximum_distance_angstrom <= 0:
            raise DescriptorError("maximum_distance_angstrom must be positive")
        if self.vector_cutoff < 0:
            raise DescriptorError("vector_cutoff cannot be negative")
        if self.dummy_offset_angstrom <= 0:
            raise DescriptorError("dummy_offset_angstrom must be positive")
        if len({parameter.key for parameter in self.pair_parameters}) != len(self.pair_parameters):
            raise DescriptorError("duplicate bond-valence pair parameters")
        bad = [index for index in self.candidate_site_indices if index < 0 or index >= len(structure.species)]
        if bad:
            raise DescriptorError(f"candidate site indices out of range: {bad}")

    @property
    def parameter_map(self) -> dict[tuple[str, int, str, int], PairParameter]:
        return {parameter.key: parameter for parameter in self.pair_parameters}


@dataclasses.dataclass(frozen=True)
class NeighborImage:
    site_index: int
    translation: Int3
    vector_cart: Vec3
    distance_angstrom: float
    normalized_distance: float
    radius_angstrom: float
