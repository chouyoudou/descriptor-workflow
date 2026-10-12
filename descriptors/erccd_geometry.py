"""Small dependency-free 3-D geometry helpers for ERCCD."""

from __future__ import annotations

import math
from typing import Sequence

from .erccd_elements import DescriptorError

Vector = tuple[float, float, float]
Matrix = tuple[Vector, Vector, Vector]


def as_vector(values: Sequence[float], *, name: str) -> Vector:
    if len(values) != 3:
        raise DescriptorError(f"{name} must have three components")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise DescriptorError(f"{name} contains a non-finite value")
    return result  # type: ignore[return-value]


def determinant(matrix: Matrix) -> float:
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def inverse(matrix: Matrix) -> Matrix:
    (a, b, c), (d, e, f), (g, h, i) = matrix
    det = determinant(matrix)
    if not math.isfinite(det) or abs(det) < 1.0e-12:
        raise DescriptorError("lattice matrix is singular or nearly singular")
    q = 1.0 / det
    return (
        ((e * i - f * h) * q, (c * h - b * i) * q, (b * f - c * e) * q),
        ((f * g - d * i) * q, (a * i - c * g) * q, (c * d - a * f) * q),
        ((d * h - e * g) * q, (b * g - a * h) * q, (a * e - b * d) * q),
    )


def row_times_matrix(row: Vector, matrix: Matrix) -> Vector:
    return (
        row[0] * matrix[0][0] + row[1] * matrix[1][0] + row[2] * matrix[2][0],
        row[0] * matrix[0][1] + row[1] * matrix[1][1] + row[2] * matrix[2][1],
        row[0] * matrix[0][2] + row[1] * matrix[1][2] + row[2] * matrix[2][2],
    )


def fractional_to_cartesian(frac: Vector, lattice: Matrix) -> Vector:
    return (
        frac[0] * lattice[0][0] + frac[1] * lattice[1][0] + frac[2] * lattice[2][0],
        frac[0] * lattice[0][1] + frac[1] * lattice[1][1] + frac[2] * lattice[2][1],
        frac[0] * lattice[0][2] + frac[1] * lattice[1][2] + frac[2] * lattice[2][2],
    )


def norm(vector: Vector) -> float:
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def _canonical_axis(vector: Vector) -> Vector:
    index = max(range(3), key=lambda item: abs(vector[item]))
    if vector[index] < 0.0:
        return (-vector[0], -vector[1], -vector[2])
    return vector


def symmetric_eigendecomposition(
    matrix: Sequence[Sequence[float]],
) -> tuple[list[float], list[Vector]]:
    """Jacobi eigensolver for a real symmetric 3x3 matrix."""
    a = [[float(matrix[i][j]) for j in range(3)] for i in range(3)]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _ in range(64):
        p, q = max(((0, 1), (0, 2), (1, 2)), key=lambda x: abs(a[x[0]][x[1]]))
        if abs(a[p][q]) < 1.0e-15:
            break
        tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
        t = math.copysign(1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)), tau)
        cosine = 1.0 / math.sqrt(1.0 + t * t)
        sine = t * cosine
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        a[p][p] = app - t * apq
        a[q][q] = aqq + t * apq
        a[p][q] = a[q][p] = 0.0
        for r in range(3):
            if r in (p, q):
                continue
            arp, arq = a[r][p], a[r][q]
            a[r][p] = a[p][r] = cosine * arp - sine * arq
            a[r][q] = a[q][r] = sine * arp + cosine * arq
        for r in range(3):
            vrp, vrq = v[r][p], v[r][q]
            v[r][p] = cosine * vrp - sine * vrq
            v[r][q] = sine * vrp + cosine * vrq
    pairs: list[tuple[float, Vector]] = []
    for column in range(3):
        vector = (v[0][column], v[1][column], v[2][column])
        length = norm(vector)
        normalized = tuple(component / length for component in vector)
        pairs.append((max(0.0, a[column][column]), _canonical_axis(normalized)))  # type: ignore[arg-type]
    pairs.sort(key=lambda item: item[0], reverse=True)
    return [item[0] for item in pairs], [item[1] for item in pairs]


__all__ = [
    "Matrix",
    "Vector",
    "as_vector",
    "determinant",
    "fractional_to_cartesian",
    "inverse",
    "norm",
    "row_times_matrix",
    "symmetric_eigendecomposition",
]
