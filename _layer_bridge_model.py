"""Internal data model, element table, math, and POSCAR parsing."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence


SCHEMA = "layer-bridge-geometry/1"
DEFAULT_BOND_SCALE = 1.18
DEFAULT_MAX_BRIDGE_ATOMS = 4

# Cordero-style single-bond covalent radii in angstrom. Values through Cm are
# provided; later elements need an explicit user override because their tabulated
# radii are prediction-dependent. Atomic numbers follow IUPAC element order.
_ELEMENT_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl "
    "Mc Lv Ts Og"
).split()

_RADIUS_VALUES = (
    0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76,
    1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22,
    1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75,
    1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39,
    1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01,
    1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87,
    1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32,
    1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06,
    2.00, 1.96, 1.90, 1.87, 1.80, 1.69,
)
_COVALENT_RADII = {
    symbol: _RADIUS_VALUES[index]
    for index, symbol in enumerate(_ELEMENT_SYMBOLS[: len(_RADIUS_VALUES)])
}
_ATOMIC_NUMBERS = {symbol: index + 1 for index, symbol in enumerate(_ELEMENT_SYMBOLS)}

# Treat metalloids, nonmetals, halogens and noble gases as ligand-like for this
# structural role classifier. Polonium is intentionally left metal-like.
_NONMETAL_OR_METALLOID = frozenset(
    "H He B C N O F Ne Si P S Cl Ar Ge As Se Br Kr Sb Te I Xe At Rn Ts Og".split()
)

DESCRIPTOR_NAMES = (
    "framework_dimensionality_max",
    "framework_dimensionality_mean",
    "layered_atom_fraction",
    "mixed_dimensionality_entropy",
    "layer_repeat_angstrom",
    "interlayer_clearance_angstrom",
    "interlayer_clearance_fraction",
    "bond_direction_anisotropy",
    "metal_fraction",
    "metal_coordination_mean",
    "metal_coordination_std",
    "metal_ligand_bond_cv_mean",
    "metal_coordination_anisotropy_mean",
    "metal_ligand_hinge_angle_mean_deg",
    "metal_ligand_hinge_cos_half_mean",
    "metal_ligand_radius_mismatch_mean",
    "metal_ligand_atomic_number_contrast_mean",
    "bridge_path_incidence_per_metal",
    "bridge_internal_atoms_mean",
    "bridge_straightness_mean",
    "single_atom_bridge_angle_mean_deg",
)


@dataclass(frozen=True)
class Structure:
    """A fully occupied periodic structure with row lattice vectors."""

    lattice: tuple[tuple[float, float, float], ...]
    species: tuple[str, ...]
    frac_coords: tuple[tuple[float, float, float], ...]
    title: str = ""

    def __post_init__(self) -> None:
        _validate_lattice(self.lattice)
        if not self.species:
            raise ValueError("structure must contain at least one atom")
        if len(self.species) != len(self.frac_coords):
            raise ValueError("species and coordinates must have the same length")
        for symbol in self.species:
            if symbol not in _ATOMIC_NUMBERS:
                raise ValueError(f"unknown element symbol: {symbol}")
        for coord in self.frac_coords:
            if len(coord) != 3 or any(not _is_finite_real(value) for value in coord):
                raise ValueError("fractional coordinates must be finite 3D vectors")


@dataclass(frozen=True)
class ElementData:
    atomic_number: int
    covalent_radius: float
    is_metal: bool


@dataclass(frozen=True)
class Edge:
    """One canonical undirected periodic bond, i <= j."""

    i: int
    j: int
    shift: tuple[int, int, int]
    distance: float
    vector: tuple[float, float, float]


def _is_finite_real(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _validate_lattice(lattice: Sequence[Sequence[float]]) -> None:
    if len(lattice) != 3 or any(len(row) != 3 for row in lattice):
        raise ValueError("lattice must contain three 3D row vectors")
    if any(not _is_finite_real(value) for row in lattice for value in row):
        raise ValueError("lattice entries must be finite real numbers")
    if abs(_determinant(lattice)) <= 1e-12:
        raise ValueError("lattice must have a finite nonzero volume")


def _determinant(matrix: Sequence[Sequence[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _inverse3(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    a, b, c = matrix
    det = _determinant(matrix)
    if abs(det) <= 1e-12:
        raise ValueError("matrix is singular")
    return (
        (
            (b[1] * c[2] - b[2] * c[1]) / det,
            (a[2] * c[1] - a[1] * c[2]) / det,
            (a[1] * b[2] - a[2] * b[1]) / det,
        ),
        (
            (b[2] * c[0] - b[0] * c[2]) / det,
            (a[0] * c[2] - a[2] * c[0]) / det,
            (a[2] * b[0] - a[0] * b[2]) / det,
        ),
        (
            (b[0] * c[1] - b[1] * c[0]) / det,
            (a[1] * c[0] - a[0] * c[1]) / det,
            (a[0] * b[1] - a[1] * b[0]) / det,
        ),
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _add_int(left: Sequence[int], right: Sequence[int]) -> tuple[int, int, int]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _sub_int(left: Sequence[int], right: Sequence[int]) -> tuple[int, int, int]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _frac_vector_to_cart(
    fractional: Sequence[float], lattice: Sequence[Sequence[float]]
) -> tuple[float, float, float]:
    return tuple(
        sum(float(fractional[row]) * float(lattice[row][column]) for row in range(3))
        for column in range(3)
    )


def _cart_vector_to_frac(
    cartesian: Sequence[float], lattice: Sequence[Sequence[float]]
) -> tuple[float, float, float]:
    inverse = _inverse3(lattice)
    # Fractional row vector = Cartesian row vector * inverse(lattice).
    return tuple(
        sum(cartesian[row] * inverse[row][column] for row in range(3))
        for column in range(3)
    )


def _angle_deg(left: Sequence[float], right: Sequence[float]) -> float:
    denom = _norm(left) * _norm(right)
    if denom <= 1e-15:
        return 0.0
    cosine = max(-1.0, min(1.0, _dot(left, right) / denom))
    return math.degrees(math.acos(cosine))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _population_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _coefficient_of_variation(values: Sequence[float]) -> float:
    center = _mean(values)
    return _population_std(values) / center if center > 0 else 0.0


def _direction_anisotropy(vectors: Sequence[Sequence[float]]) -> float:
    """Return a rotation-invariant second-moment anisotropy in [0, 1]."""
    if not vectors:
        return 0.0
    q = [[0.0] * 3 for _ in range(3)]
    count = 0
    for vector in vectors:
        length = _norm(vector)
        if length <= 1e-15:
            continue
        unit = [component / length for component in vector]
        for row in range(3):
            for column in range(3):
                q[row][column] += unit[row] * unit[column]
        count += 1
    if count == 0:
        return 0.0
    for row in range(3):
        for column in range(3):
            q[row][column] /= count
    trace_q_squared = sum(
        q[row][column] * q[column][row]
        for row in range(3)
        for column in range(3)
    )
    value = math.sqrt(max(0.0, 1.5 * (trace_q_squared - 1.0 / 3.0)))
    return max(0.0, min(1.0, value))


def default_element_table() -> dict[str, ElementData]:
    """Return a fresh element table used for bond and role inference."""
    table: dict[str, ElementData] = {}
    for symbol, radius in _COVALENT_RADII.items():
        table[symbol] = ElementData(
            atomic_number=_ATOMIC_NUMBERS[symbol],
            covalent_radius=radius,
            is_metal=symbol not in _NONMETAL_OR_METALLOID,
        )
    return table


def load_element_table(path: Path | None = None) -> dict[str, ElementData]:
    """Load optional JSON overrides on top of the built-in element table."""
    table = default_element_table()
    if path is None:
        return table
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("element table must be a JSON object keyed by symbol")
    for symbol, record in raw.items():
        if symbol not in _ATOMIC_NUMBERS or not isinstance(record, dict):
            raise ValueError(f"invalid element-table record for {symbol!r}")
        current = table.get(symbol)
        atomic_number = record.get("atomic_number", _ATOMIC_NUMBERS[symbol])
        radius = record.get(
            "covalent_radius", current.covalent_radius if current else None
        )
        is_metal = record.get(
            "is_metal",
            current.is_metal if current else symbol not in _NONMETAL_OR_METALLOID,
        )
        if type(atomic_number) is not int or atomic_number < 1:
            raise ValueError(f"invalid atomic_number for {symbol}")
        if not _is_finite_real(radius) or float(radius) <= 0:
            raise ValueError(f"invalid covalent_radius for {symbol}")
        if type(is_metal) is not bool:
            raise ValueError(f"invalid is_metal for {symbol}")
        table[symbol] = ElementData(atomic_number, float(radius), is_metal)
    return table


def parse_poscar(text: str) -> Structure:
    """Parse a VASP 5-style POSCAR with explicit element symbols.

    Selective dynamics flags are ignored. Full occupancy is assumed because a
    POSCAR does not encode site occupancies.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        raise ValueError("POSCAR is too short")
    title = lines[0]
    scale_tokens = lines[1].split()
    if len(scale_tokens) != 1:
        raise ValueError("POSCAR must use one universal scale factor")
    try:
        scale = float(scale_tokens[0])
        raw_lattice = [
            [float(value) for value in lines[index].split()]
            for index in range(2, 5)
        ]
    except ValueError as exc:
        raise ValueError("POSCAR scale and lattice must be numeric") from exc
    if scale == 0 or not math.isfinite(scale):
        raise ValueError("POSCAR scale must be finite and nonzero")
    _validate_lattice(raw_lattice)
    if scale > 0:
        linear_scale = scale
    else:
        target_volume = abs(scale)
        raw_volume = abs(_determinant(raw_lattice))
        linear_scale = (target_volume / raw_volume) ** (1.0 / 3.0)
    lattice = tuple(
        tuple(value * linear_scale for value in row) for row in raw_lattice
    )

    symbols = lines[5].split()
    if not symbols or all(token.lstrip("+-").isdigit() for token in symbols):
        raise ValueError("POSCAR must include explicit VASP 5 element symbols")
    for symbol in symbols:
        if symbol not in _ATOMIC_NUMBERS:
            raise ValueError(f"unknown element symbol: {symbol}")
    try:
        counts = [int(token) for token in lines[6].split()]
    except ValueError as exc:
        raise ValueError("POSCAR atom counts must be integers") from exc
    if len(counts) != len(symbols) or any(count < 1 for count in counts):
        raise ValueError("POSCAR element symbols and positive counts must align")

    cursor = 7
    if lines[cursor].lower().startswith("s"):
        cursor += 1
    if cursor >= len(lines):
        raise ValueError("POSCAR coordinate mode is missing")
    mode = lines[cursor].lower()
    cursor += 1
    if not (mode.startswith("d") or mode.startswith("c") or mode.startswith("k")):
        raise ValueError("POSCAR coordinate mode must be Direct or Cartesian")
    atom_count = sum(counts)
    if len(lines) < cursor + atom_count:
        raise ValueError("POSCAR has fewer coordinate rows than atoms")

    species = tuple(
        symbol for symbol, count in zip(symbols, counts) for _ in range(count)
    )
    fractional: list[tuple[float, float, float]] = []
    for row in lines[cursor : cursor + atom_count]:
        fields = row.split()
        if len(fields) < 3:
            raise ValueError("each POSCAR coordinate row needs three values")
        try:
            coord = tuple(float(value) for value in fields[:3])
        except ValueError as exc:
            raise ValueError("POSCAR coordinates must be numeric") from exc
        if any(not math.isfinite(value) for value in coord):
            raise ValueError("POSCAR coordinates must be finite")
        if mode.startswith("d"):
            frac = coord
        else:
            cart = tuple(value * linear_scale for value in coord)
            frac = _cart_vector_to_frac(cart, lattice)
        fractional.append(tuple(value % 1.0 for value in frac))
    return Structure(
        lattice=lattice,
        species=species,
        frac_coords=tuple(fractional),
        title=title,
    )


def read_poscar(path: Path) -> Structure:
    return parse_poscar(path.read_text(encoding="utf-8"))
