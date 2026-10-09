"""Exact periodic shell and fabric operations for PIAN."""
from __future__ import annotations

import math
import numpy as np

from pian_model import (
    DescriptorUnavailable,
    DISTANCE_TOLERANCE_ANGSTROM,
    MAX_FIRST_SHELL_EDGES,
    MIN_SEPARATION_ANGSTROM,
    SHELL_FACTOR,
    StructureData,
)


def _translation_bounds(delta: np.ndarray, radius: float, inverse_lattice: np.ndarray) -> list[range]:
    column_norms = np.linalg.norm(inverse_lattice, axis=0)
    bounds: list[range] = []
    for component, reciprocal_norm in zip(delta, column_norms):
        reach = radius * float(reciprocal_norm) + 2.0e-12
        bounds.append(range(math.ceil(-float(component)-reach), math.floor(-float(component)+reach)+1))
    return bounds


def _nearest_image_vector(delta_fractional: np.ndarray, lattice: np.ndarray, inverse_lattice: np.ndarray, *, exclude_zero_translation: bool) -> tuple[np.ndarray, tuple[int, int, int], float]:
    rounded = -np.rint(delta_fractional).astype(int)
    candidates: list[np.ndarray] = [rounded]
    if exclude_zero_translation and np.all(rounded == 0):
        for axis in range(3):
            for sign in (-1, 1):
                item = np.zeros(3, dtype=int); item[axis] = sign; candidates.append(item)
    best_vector: np.ndarray | None = None
    best_translation: tuple[int, int, int] | None = None
    best_distance = math.inf
    for translation in candidates:
        if exclude_zero_translation and np.all(translation == 0):
            continue
        vector = (delta_fractional + translation) @ lattice
        distance = float(np.linalg.norm(vector))
        if distance < best_distance:
            best_vector, best_translation, best_distance = vector, tuple(int(x) for x in translation), distance
    if best_vector is None or not math.isfinite(best_distance):
        raise DescriptorUnavailable("algorithm", "nearest_image_initialization", "could not initialize nearest image")
    ranges = _translation_bounds(delta_fractional, best_distance, inverse_lattice)
    for a in ranges[0]:
        for b in ranges[1]:
            for c in ranges[2]:
                if exclude_zero_translation and a == b == c == 0:
                    continue
                vector = (delta_fractional + np.asarray([a,b,c], dtype=np.float64)) @ lattice
                distance = float(np.linalg.norm(vector))
                if distance + 1.0e-13 < best_distance:
                    best_vector, best_translation, best_distance = vector, (a,b,c), distance
    assert best_translation is not None
    return best_vector, best_translation, best_distance


def _adaptive_shell(structure: StructureData) -> dict[str, np.ndarray]:
    lattice = structure.lattice
    try:
        inverse = np.linalg.inv(lattice)
    except np.linalg.LinAlgError as exc:
        raise DescriptorUnavailable("input", "singular_lattice", "lattice cannot be inverted") from exc
    fractions = structure.fractional_coordinates
    site_count = len(structure.species)
    nearest = np.full(site_count, math.inf, dtype=np.float64)
    for i in range(site_count):
        for j in range(site_count):
            _, _, distance = _nearest_image_vector(fractions[j]-fractions[i], lattice, inverse, exclude_zero_translation=(i==j))
            if distance < MIN_SEPARATION_ANGSTROM:
                raise DescriptorUnavailable("input", "overlapping_periodic_sites", f"site {i} and site/image {j} are separated by {distance:.6g} angstrom")
            nearest[i] = min(nearest[i], distance)
    if not np.isfinite(nearest).all() or np.any(nearest <= 0):
        raise DescriptorUnavailable("algorithm", "nearest_neighbor_failure", "nearest distances are invalid")
    centers: list[int] = []; points: list[int] = []; translations: list[tuple[int,int,int]] = []; vectors: list[np.ndarray] = []; distances: list[float] = []
    for i in range(site_count):
        radius = SHELL_FACTOR * float(nearest[i]) + 10.0 * DISTANCE_TOLERANCE_ANGSTROM
        for j in range(site_count):
            delta = fractions[j]-fractions[i]
            ranges = _translation_bounds(delta, radius, inverse)
            for a in ranges[0]:
                for b in ranges[1]:
                    for c in ranges[2]:
                        if i == j and a == b == c == 0:
                            continue
                        vector = (delta + np.asarray([a,b,c], dtype=np.float64)) @ lattice
                        distance = float(np.linalg.norm(vector))
                        if distance < MIN_SEPARATION_ANGSTROM:
                            raise DescriptorUnavailable("input", "overlapping_periodic_sites", f"site {i} and site/image {j} are separated by {distance:.6g} angstrom")
                        if distance <= radius:
                            centers.append(i); points.append(j); translations.append((a,b,c)); vectors.append(vector); distances.append(distance)
                            if len(centers) > MAX_FIRST_SHELL_EDGES:
                                raise DescriptorUnavailable("resource", "first_shell_edge_limit", f"first shell exceeds {MAX_FIRST_SHELL_EDGES} directed edges")
    center_array = np.asarray(centers, dtype=np.int64)
    if center_array.size == 0:
        raise DescriptorUnavailable("algorithm", "empty_first_shell", "no first-shell edges were found")
    coordination = np.bincount(center_array, minlength=site_count).astype(np.int64)
    if np.any(coordination < 1):
        raise DescriptorUnavailable("algorithm", "empty_first_shell", f"empty shell for sites {np.flatnonzero(coordination<1).tolist()}")
    return {"center":center_array, "point":np.asarray(points,dtype=np.int64), "translations":np.asarray(translations,dtype=np.int64), "vectors":np.asarray(vectors,dtype=np.float64), "distances":np.asarray(distances,dtype=np.float64), "coordination":coordination, "nearest":nearest}


def _stats(values: np.ndarray) -> tuple[float,float,float,float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise DescriptorUnavailable("numerical", "invalid_statistics_input", "statistics require finite non-empty values")
    return float(np.mean(values)), float(np.std(values)), float(np.min(values)), float(np.max(values))


def _fabric(vectors: np.ndarray, weights: np.ndarray) -> dict[str,float]:
    vectors = np.asarray(vectors,dtype=np.float64); weights=np.asarray(weights,dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or weights.shape != (len(vectors),):
        raise DescriptorUnavailable("algorithm", "invalid_fabric_input", "fabric arrays have inconsistent shapes")
    if not np.isfinite(vectors).all() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise DescriptorUnavailable("numerical", "invalid_fabric_input", "fabric arrays must be finite and nonnegative")
    total=float(np.sum(weights))
    if total <= 1.0e-15:
        raise DescriptorUnavailable("definition_domain", "zero_chi_contrast", "all first-shell chi contrasts are zero")
    unit=vectors/np.linalg.norm(vectors,axis=1)[:,None]
    tensor=np.einsum("n,ni,nj->ij",weights,unit,unit,optimize=True)/total
    eigenvalues=np.linalg.eigvalsh(0.5*(tensor+tensor.T)); eigenvalues[np.abs(eigenvalues)<1e-14]=0.0
    eigenvalues=np.clip(eigenvalues,0.0,None); eigenvalues/=float(np.sum(eigenvalues))
    l1,l2,l3=(float(x) for x in eigenvalues)
    return {"q2":math.sqrt(max(0.0,1.5*sum((x-1/3)**2 for x in eigenvalues))), "linearity":l3-l2, "planarity":2*(l2-l1), "isotropy":3*l1}
