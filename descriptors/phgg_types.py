"""Types, parameter validation, and VASP POSCAR parsing for PHGG."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .phgg_radii import canonical_symbol

Vector3 = tuple[float, float, float]
IntVector3 = tuple[int, int, int]
Matrix3 = tuple[Vector3, Vector3, Vector3]


class DescriptorError(ValueError):
    """Raised when a structure or parameter set cannot be analysed safely."""


def vector_scale(a: Vector3, scale: float) -> Vector3:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Vector3) -> float:
    return math.sqrt(max(0.0, dot(a, a)))


def determinant(matrix: Matrix3) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def inverse(matrix: Matrix3) -> Matrix3:
    a, b, c = matrix
    det = determinant(matrix)
    if not math.isfinite(det) or abs(det) <= 1.0e-12:
        raise DescriptorError("lattice is singular or numerically degenerate")
    inv_det = 1.0 / det
    return (
        (
            (b[1] * c[2] - b[2] * c[1]) * inv_det,
            (a[2] * c[1] - a[1] * c[2]) * inv_det,
            (a[1] * b[2] - a[2] * b[1]) * inv_det,
        ),
        (
            (b[2] * c[0] - b[0] * c[2]) * inv_det,
            (a[0] * c[2] - a[2] * c[0]) * inv_det,
            (a[2] * b[0] - a[0] * b[2]) * inv_det,
        ),
        (
            (b[0] * c[1] - b[1] * c[0]) * inv_det,
            (a[1] * c[0] - a[0] * c[1]) * inv_det,
            (a[0] * b[1] - a[1] * b[0]) * inv_det,
        ),
    )


def row_times_matrix(row: Vector3, matrix: Matrix3) -> Vector3:
    return (
        row[0] * matrix[0][0] + row[1] * matrix[1][0] + row[2] * matrix[2][0],
        row[0] * matrix[0][1] + row[1] * matrix[1][1] + row[2] * matrix[2][1],
        row[0] * matrix[0][2] + row[1] * matrix[1][2] + row[2] * matrix[2][2],
    )


def frac_to_cart(frac: Vector3, lattice: Matrix3) -> Vector3:
    return (
        frac[0] * lattice[0][0] + frac[1] * lattice[1][0] + frac[2] * lattice[2][0],
        frac[0] * lattice[0][1] + frac[1] * lattice[1][1] + frac[2] * lattice[2][1],
        frac[0] * lattice[0][2] + frac[1] * lattice[1][2] + frac[2] * lattice[2][2],
    )


def wrap_fractional(frac: Vector3) -> Vector3:
    return tuple(value - math.floor(value) for value in frac)  # type: ignore[return-value]


@dataclass(frozen=True)
class StructureData:
    comment: str
    lattice: Matrix3
    inverse_lattice: Matrix3
    species: tuple[str, ...]
    fractional: tuple[Vector3, ...]

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    @property
    def volume_A3(self) -> float:
        return abs(determinant(self.lattice))


@dataclass(frozen=True)
class Settings:
    bond_scale: float = 1.18
    bond_offset_A: float = 0.10
    contact_scale: float = 1.05
    grid_spacing_A: float = 1.75
    max_grid_points: int = 512
    probe_radius_A: float = 0.0
    max_atoms: int = 512
    max_edges: int = 100_000

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None) -> "Settings":
        raw = dict(params or {})
        allowed = {
            "bond_scale", "bond_offset_A", "contact_scale", "grid_spacing_A",
            "max_grid_points", "probe_radius_A", "max_atoms", "max_edges",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise DescriptorError(f"unknown parameters: {', '.join(unknown)}")
        values: dict[str, Any] = {}
        for name in ("bond_scale", "bond_offset_A", "contact_scale", "grid_spacing_A", "probe_radius_A"):
            if name in raw:
                try:
                    values[name] = float(raw[name])
                except (TypeError, ValueError) as exc:
                    raise DescriptorError(f"{name} must be numeric") from exc
        for name in ("max_grid_points", "max_atoms", "max_edges"):
            if name in raw:
                value = raw[name]
                if isinstance(value, bool):
                    raise DescriptorError(f"{name} must be an integer")
                try:
                    converted = int(value)
                except (TypeError, ValueError) as exc:
                    raise DescriptorError(f"{name} must be an integer") from exc
                try:
                    if float(value) != converted:
                        raise DescriptorError(f"{name} must be an integer")
                except (TypeError, ValueError) as exc:
                    raise DescriptorError(f"{name} must be an integer") from exc
                values[name] = converted
        result = cls(**values)
        if not 0.8 <= result.bond_scale <= 2.0:
            raise DescriptorError("bond_scale must be in [0.8, 2.0]")
        if not 0.0 <= result.bond_offset_A <= 1.0:
            raise DescriptorError("bond_offset_A must be in [0, 1]")
        if not 0.5 <= result.contact_scale <= 2.0:
            raise DescriptorError("contact_scale must be in [0.5, 2.0]")
        if not 0.25 <= result.grid_spacing_A <= 10.0:
            raise DescriptorError("grid_spacing_A must be in [0.25, 10]")
        if not 8 <= result.max_grid_points <= 100_000:
            raise DescriptorError("max_grid_points must be in [8, 100000]")
        if not 0.0 <= result.probe_radius_A <= 10.0:
            raise DescriptorError("probe_radius_A must be in [0, 10]")
        if not 1 <= result.max_atoms <= 100_000:
            raise DescriptorError("max_atoms must be in [1, 100000]")
        if not 1 <= result.max_edges <= 5_000_000:
            raise DescriptorError("max_edges must be in [1, 5000000]")
        return result


@dataclass(frozen=True)
class FeatureResult:
    features: dict[str, float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"features": self.features, "metadata": self.metadata}


def _parse_vector(line: str, context: str) -> Vector3:
    fields = line.split()
    if len(fields) < 3:
        raise DescriptorError(f"{context} must contain three numbers")
    try:
        values = tuple(float(fields[index]) for index in range(3))
    except ValueError as exc:
        raise DescriptorError(f"{context} contains a non-numeric value") from exc
    if not all(math.isfinite(value) for value in values):
        raise DescriptorError(f"{context} contains a non-finite value")
    return values  # type: ignore[return-value]


def parse_poscar_text(poscar_data: str, *, max_atoms: int = 512) -> StructureData:
    """Parse a VASP 5/6 POSCAR string without inferring missing elements."""
    lines = [line.rstrip() for line in poscar_data.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 8:
        raise DescriptorError("POSCAR is incomplete")
    comment = lines[0].strip()
    try:
        raw_scale = float(lines[1].split()[0])
    except (IndexError, ValueError) as exc:
        raise DescriptorError("invalid POSCAR scale") from exc
    if not math.isfinite(raw_scale) or raw_scale == 0.0:
        raise DescriptorError("POSCAR scale must be finite and nonzero")
    raw_lattice: Matrix3 = (
        _parse_vector(lines[2], "lattice vector a"),
        _parse_vector(lines[3], "lattice vector b"),
        _parse_vector(lines[4], "lattice vector c"),
    )
    raw_volume = abs(determinant(raw_lattice))
    if raw_volume <= 1.0e-12:
        raise DescriptorError("POSCAR lattice is singular")
    factor = raw_scale if raw_scale > 0 else (abs(raw_scale) / raw_volume) ** (1.0 / 3.0)
    lattice: Matrix3 = tuple(vector_scale(vector, factor) for vector in raw_lattice)  # type: ignore[assignment]
    inv_lattice = inverse(lattice)

    symbol_fields = lines[5].split()
    if not symbol_fields:
        raise DescriptorError("POSCAR element-symbol line is empty")
    try:
        symbols = tuple(canonical_symbol(token) for token in symbol_fields)
    except ValueError as exc:
        raise DescriptorError(str(exc)) from exc
    count_fields = lines[6].split()
    if len(count_fields) != len(symbols):
        raise DescriptorError("element and count columns differ; VASP 4 files are unsupported")
    try:
        counts = tuple(int(token) for token in count_fields)
    except ValueError as exc:
        raise DescriptorError("POSCAR atom counts must be integers") from exc
    if any(count < 0 for count in counts):
        raise DescriptorError("POSCAR atom counts must be nonnegative")
    n_atoms = sum(counts)
    if n_atoms <= 0:
        raise DescriptorError("POSCAR has no atoms")
    if n_atoms > max_atoms:
        raise DescriptorError(f"POSCAR has {n_atoms} atoms; max_atoms={max_atoms}")

    cursor = 7
    if cursor >= len(lines):
        raise DescriptorError("POSCAR lacks coordinate mode")
    if lines[cursor].strip().lower().startswith("s"):
        cursor += 1
        if cursor >= len(lines):
            raise DescriptorError("POSCAR lacks coordinate mode after Selective dynamics")
    mode = lines[cursor].strip().lower()
    cursor += 1
    if not mode:
        raise DescriptorError("empty POSCAR coordinate mode")
    direct = mode.startswith("d")
    cartesian = mode.startswith("c") or mode.startswith("k")
    if not (direct or cartesian):
        raise DescriptorError(f"unsupported POSCAR coordinate mode: {lines[cursor - 1]!r}")
    if len(lines) < cursor + n_atoms:
        raise DescriptorError("POSCAR has fewer coordinate rows than atoms")

    species: list[str] = []
    for symbol, count in zip(symbols, counts):
        species.extend([symbol] * count)
    fractional: list[Vector3] = []
    for atom_index in range(n_atoms):
        values = _parse_vector(lines[cursor + atom_index], f"coordinate row {atom_index + 1}")
        frac = values if direct else row_times_matrix(vector_scale(values, factor), inv_lattice)
        fractional.append(wrap_fractional(frac))

    return StructureData(
        comment=comment,
        lattice=lattice,
        inverse_lattice=inv_lattice,
        species=tuple(species),
        fractional=tuple(fractional),
    )
