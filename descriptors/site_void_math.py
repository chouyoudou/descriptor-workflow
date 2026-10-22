#!/usr/bin/env python3
"""Dependency-free vector, lattice, and symmetric 3x3 operations."""

from __future__ import annotations

import math

try:
    from .site_void_atoms import DescriptorError
except ImportError:  # flat private task bundle
    from site_void_atoms import DescriptorError

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]
Int3 = tuple[int, int, int]

def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Vec3) -> float:
    return math.sqrt(max(0.0, _dot(a, a)))


def _vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vscale(a: Vec3, scalar: float) -> Vec3:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def _unit(a: Vec3, *, fallback: Vec3 = (1.0, 0.0, 0.0)) -> Vec3:
    length = _norm(a)
    if length <= 1.0e-15:
        return fallback
    return _vscale(a, 1.0 / length)


def _det3(matrix: Mat3) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _inverse3(matrix: Mat3) -> Mat3:
    a, b, c = matrix
    det = _det3(matrix)
    if abs(det) <= 1.0e-15:
        raise DescriptorError("lattice matrix is singular")
    inv_det = 1.0 / det
    # Matrix is represented by rows.  This is the standard adjugate inverse.
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


def _frac_to_cart(frac: Vec3, lattice: Mat3) -> Vec3:
    return (
        frac[0] * lattice[0][0] + frac[1] * lattice[1][0] + frac[2] * lattice[2][0],
        frac[0] * lattice[0][1] + frac[1] * lattice[1][1] + frac[2] * lattice[2][1],
        frac[0] * lattice[0][2] + frac[1] * lattice[1][2] + frac[2] * lattice[2][2],
    )


def _cart_to_frac(cart: Vec3, lattice_inverse: Mat3) -> Vec3:
    # cart(row) @ inverse(lattice), with lattice vectors stored as rows.
    return (
        cart[0] * lattice_inverse[0][0]
        + cart[1] * lattice_inverse[1][0]
        + cart[2] * lattice_inverse[2][0],
        cart[0] * lattice_inverse[0][1]
        + cart[1] * lattice_inverse[1][1]
        + cart[2] * lattice_inverse[2][1],
        cart[0] * lattice_inverse[0][2]
        + cart[1] * lattice_inverse[1][2]
        + cart[2] * lattice_inverse[2][2],
    )


def _wrap_frac(frac: Vec3) -> Vec3:
    return tuple(x - math.floor(x) for x in frac)  # type: ignore[return-value]


def _outer(a: Vec3) -> Mat3:
    return (
        (a[0] * a[0], a[0] * a[1], a[0] * a[2]),
        (a[1] * a[0], a[1] * a[1], a[1] * a[2]),
        (a[2] * a[0], a[2] * a[1], a[2] * a[2]),
    )


def _symmetric_eigendecomposition(matrix: Mat3) -> tuple[Vec3, tuple[Vec3, Vec3, Vec3]]:
    """Jacobi eigensolver for a real symmetric 3x3 matrix.

    Eigenvalues are returned in ascending order and eigenvectors as matching
    Cartesian unit vectors.  The small fixed-size solver avoids a NumPy runtime
    dependency while retaining exact rotational covariance up to floating-point
    roundoff.
    """

    a = [[float(matrix[i][j]) for j in range(3)] for i in range(3)]
    # Remove tiny asymmetric noise rather than silently accepting a non-symmetric matrix.
    for i in range(3):
        for j in range(i + 1, 3):
            mean = 0.5 * (a[i][j] + a[j][i])
            a[i][j] = a[j][i] = mean
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]

    for _ in range(64):
        p, q = max(((0, 1), (0, 2), (1, 2)), key=lambda ij: abs(a[ij[0]][ij[1]]))
        apq = a[p][q]
        if abs(apq) <= 1.0e-15:
            break
        app, aqq = a[p][p], a[q][q]
        tau = (aqq - app) / (2.0 * apq)
        if tau >= 0:
            t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
        else:
            t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        for k in range(3):
            if k in (p, q):
                continue
            akp, akq = a[k][p], a[k][q]
            a[k][p] = a[p][k] = c * akp - s * akq
            a[k][q] = a[q][k] = s * akp + c * akq
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = a[q][p] = 0.0

        for k in range(3):
            vkp, vkq = v[k][p], v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq

    pairs = []
    for idx in range(3):
        vector = _unit((v[0][idx], v[1][idx], v[2][idx]))
        pairs.append((a[idx][idx], vector))
    pairs.sort(key=lambda item: item[0])
    eigenvalues = tuple(item[0] for item in pairs)
    eigenvectors = tuple(item[1] for item in pairs)
    return eigenvalues, eigenvectors  # type: ignore[return-value]
