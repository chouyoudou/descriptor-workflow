"""Local bonded coordination-shell geometry."""

from __future__ import annotations

import numpy as np

from .model import Structure
from .periodic import _radii

def coordination_shell_metrics(
    structure: Structure,
    adjacency: list[list[tuple[int, tuple[int, int, int], np.ndarray, float]]],
) -> tuple[float, float]:
    _, covalent, _ = _radii(structure)
    radial_weighted = 0.0
    orient_weighted = 0.0
    total_weight = 0.0
    for center, neighbors in enumerate(adjacency):
        # Canonical bond construction gives unique neighbor images. Protect
        # against accidental exact duplicates from pathological cells.
        unique: dict[tuple[int, tuple[int, int, int]], tuple[np.ndarray, float]] = {}
        for neighbor, shift, vector, distance in neighbors:
            key = (neighbor, shift)
            if key not in unique or distance < unique[key][1]:
                unique[key] = (vector, distance)
        if len(unique) < 3:
            continue
        normalized = []
        directions = []
        for (neighbor, _shift), (vector, distance) in unique.items():
            radius_sum = float(covalent[center] + covalent[neighbor])
            normalized.append(distance / radius_sum)
            directions.append(vector / distance)
        values = np.asarray(normalized, dtype=float)
        mean = float(np.mean(values))
        radial = float(np.sqrt(np.mean((values - mean) ** 2)) / mean) if mean > 1e-15 else 0.0
        dirs = np.asarray(directions, dtype=float)
        tensor = dirs.T @ dirs / len(dirs)
        eig = np.linalg.eigvalsh(tensor)
        orient = float(np.clip(1.5 * np.sum((eig - 1.0 / 3.0) ** 2), 0.0, 1.0))
        weight = float(len(unique))
        radial_weighted += weight * radial
        orient_weighted += weight * orient
        total_weight += weight
    if not total_weight:
        return 0.0, 0.0
    return radial_weighted / total_weight, orient_weighted / total_weight

