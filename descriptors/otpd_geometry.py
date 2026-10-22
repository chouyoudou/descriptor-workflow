"""Periodic geometry primitives and six-neighbour shell selection for OTPD."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

try:
    from .otpd_types import DescriptorError, Neighbor, Settings, StructureData
except ImportError:
    from otpd_types import DescriptorError, Neighbor, Settings, StructureData

def _unit(vector: np.ndarray, tol: float = 1.0e-14) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= tol:
        raise DescriptorError("zero or non-finite vector")
    return np.asarray(vector, dtype=float) / norm


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.dot(_unit(a), _unit(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _canonical_axis(vector: np.ndarray) -> np.ndarray:
    axis = _unit(vector)
    for value in axis:
        if abs(float(value)) > 1.0e-12:
            if value < 0:
                axis = -axis
            break
    return axis


def _perfect_matchings(items: tuple[int, ...]) -> Iterable[tuple[tuple[int, int], ...]]:
    if not items:
        yield ()
        return
    first = items[0]
    for offset in range(1, len(items)):
        second = items[offset]
        rest = items[1:offset] + items[offset + 1 :]
        for tail in _perfect_matchings(rest):
            pair = (min(first, second), max(first, second))
            yield tuple(sorted((pair,) + tail))


def _translation_grid(lattice: np.ndarray, cutoff: float) -> np.ndarray:
    inverse = np.linalg.inv(lattice)
    bounds = [int(math.ceil(cutoff * float(np.linalg.norm(inverse[:, k])))) + 1 for k in range(3)]
    shifts = np.asarray(
        list(
            itertools.product(
                range(-bounds[0], bounds[0] + 1),
                range(-bounds[1], bounds[1] + 1),
                range(-bounds[2], bounds[2] + 1),
            )
        ),
        dtype=int,
    )
    return shifts


def periodic_neighbors(
    structure: StructureData,
    center_index: int,
    radii: Mapping[str, float],
    settings: Settings,
) -> list[Neighbor]:
    center_symbol = structure.species[center_index]
    center_radius = radii.get(center_symbol)
    if center_radius is None:
        raise DescriptorError(f"missing radius for center element {center_symbol}")
    max_radius = max(radii[symbol] for symbol in structure.species if symbol in radii)
    cutoff = min(
        settings.max_search_radius,
        settings.neighbor_search_ratio * (center_radius + max_radius) + 0.25,
    )
    shifts = _translation_grid(structure.lattice, cutoff)
    center_frac = structure.fractional[center_index]
    result: list[Neighbor] = []
    for atom_index, (symbol, frac) in enumerate(zip(structure.species, structure.fractional, strict=True)):
        donor_radius = radii.get(symbol)
        if donor_radius is None:
            continue
        delta_frac = frac[None, :] + shifts - center_frac[None, :]
        cart = delta_frac @ structure.lattice
        distances = np.linalg.norm(cart, axis=1)
        mask = (distances <= cutoff + 1.0e-10) & (distances > 1.0e-10)
        if atom_index == center_index:
            zero_mask = np.all(shifts == 0, axis=1)
            mask &= ~zero_mask
        for local_index in np.nonzero(mask)[0]:
            distance = float(distances[local_index])
            image = tuple(int(x) for x in shifts[local_index])
            result.append(
                Neighbor(
                    atom_index=atom_index,
                    image=image,
                    symbol=symbol,
                    vector=np.asarray(cart[local_index], dtype=float),
                    distance=distance,
                    normalized_distance=distance / (center_radius + donor_radius),
                )
            )
    result.sort(
        key=lambda n: (
            round(n.normalized_distance, 12),
            round(n.distance, 12),
            n.symbol,
            n.atom_index,
            n.image,
        )
    )
    return result


def select_six_shell(neighbors: Sequence[Neighbor], settings: Settings) -> tuple[list[Neighbor] | None, dict[str, Any]]:
    if len(neighbors) < 6:
        return None, {"reason": "fewer_than_six_neighbors", "neighbor_count": len(neighbors)}
    rho6 = float(neighbors[5].normalized_distance)
    diag: dict[str, Any] = {
        "rho_6": rho6,
        "neighbor_count_within_search": len(neighbors),
        "shell_ratio_max": settings.shell_ratio_max,
    }
    if rho6 > settings.shell_ratio_max:
        diag["reason"] = "sixth_neighbor_beyond_radius_protocol"
        return None, diag
    if len(neighbors) >= 7:
        rho7 = float(neighbors[6].normalized_distance)
        gap_abs = rho7 - rho6
        gap_ratio = rho7 / rho6 if rho6 > 0 else math.inf
        diag.update({"rho_7": rho7, "gap_abs": gap_abs, "gap_ratio": gap_ratio})
        if gap_abs < settings.shell_gap_abs_min or gap_ratio < settings.shell_gap_ratio_min:
            diag["reason"] = "ambiguous_six_neighbor_shell"
            return None, diag
    diag["reason"] = "selected"
    return list(neighbors[:6]), diag


def _pair_metric(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(x) for x in values)
    m1, m2, m3 = ordered
    mean = (m1 + m2 + m3) / 3.0
    if mean <= 0:
        raise DescriptorError("pair means must be positive")
    return {
        "pair_range": (m3 - m1) / mean,
        # Positive: one isolated short pair (compression-like topology).
        # Negative: one isolated long pair (elongation-like topology).
        "uniaxial_polarity": (2.0 * m2 - m1 - m3) / (2.0 * mean),
        "short_pair_log_compression": math.log(((m2 + m3) / 2.0) / m1),
        "long_pair_log_elongation": math.log(m3 / ((m1 + m2) / 2.0)),
    }


def _correlation(x: Sequence[float], y: Sequence[float], tol: float = 1.0e-14) -> float | None:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size < 2 or ya.size != xa.size:
        return None
    xc = xa - float(np.mean(xa))
    yc = ya - float(np.mean(ya))
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    if denom <= tol:
        return None
    return float(np.dot(xc, yc) / denom)


def _cosine_similarity(x: Sequence[float], y: Sequence[float], tol: float = 1.0e-14) -> float | None:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size == 0 or ya.size != xa.size:
        return None
    denom = float(np.linalg.norm(xa) * np.linalg.norm(ya))
    if denom <= tol:
        return None
    return float(np.dot(xa, ya) / denom)


