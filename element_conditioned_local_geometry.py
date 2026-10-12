"""Numerically stable public facade for element-conditioned local geometry.

The fixed science implementation lives in ``element_conditioned_local_geometry_base``.
This facade installs two representation-stability safeguards before re-exporting the
public API: an absolute distance tie tolerance for the periodic K-neighbour shell,
and a variance floor for otherwise undefined ratios of vanishing site variance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import element_conditioned_local_geometry_base as _base

TIE_ABSOLUTE_TOLERANCE_ANGSTROM = 1.0e-5
ELEMENT_CONDITIONED_VARIANCE_FLOOR = 1.0e-24


def _tie_margin(kth_distances: np.ndarray) -> np.ndarray:
    return np.maximum(
        TIE_ABSOLUTE_TOLERANCE_ANGSTROM,
        np.abs(kth_distances) * _base.TIE_RELATIVE_TOLERANCE,
    )


def _tie_complete_knn_shell(structure: Any) -> dict[str, Any]:
    site_count = len(structure)
    density_scale = float((structure.volume / site_count) ** (1.0 / 3.0))
    cutoff = min(
        max(_base.INITIAL_SEARCH_CUTOFF_ANGSTROM, 2.1 * density_scale),
        _base.MAX_SEARCH_CUTOFF_ANGSTROM,
    )

    for _ in range(14):
        center, point, offsets, distances = _base._raw_periodic_pairs(structure, cutoff)
        zero_offset = np.all(np.abs(offsets) <= 1.0e-12, axis=1)
        exact_self = (center == point) & zero_offset & (distances <= 1.0e-7)
        overlap = (~exact_self) & (distances < _base.MIN_SEPARATION_ANGSTROM)
        if np.any(overlap):
            first = int(np.flatnonzero(overlap)[0])
            raise _base.DescriptorUnavailable(
                "input",
                "overlapping_periodic_sites",
                (
                    "distinct sites/images are closer than "
                    f"{_base.MIN_SEPARATION_ANGSTROM:.2f} angstrom "
                    f"(sites {int(center[first])} and {int(point[first])}, "
                    f"distance {distances[first]:.6g})"
                ),
            )

        nonself = (~exact_self) & (distances > _base.DISTANCE_TOLERANCE_ANGSTROM)
        counts = np.bincount(center[nonself], minlength=site_count).astype(np.int64)
        if np.any(counts < _base.K_NEIGHBORS):
            next_cutoff = cutoff * 1.65
            if next_cutoff > _base.MAX_SEARCH_CUTOFF_ANGSTROM + 1.0e-12:
                missing = np.flatnonzero(counts < _base.K_NEIGHBORS).tolist()
                raise _base.DescriptorUnavailable(
                    "resource",
                    "knn_beyond_search_limit",
                    (
                        f"sites {missing[:12]} have fewer than {_base.K_NEIGHBORS} "
                        f"periodic neighbors within {_base.MAX_SEARCH_CUTOFF_ANGSTROM:g} angstrom"
                    ),
                )
            cutoff = next_cutoff
            continue

        kth = np.empty(site_count, dtype=np.float64)
        for site_index in range(site_count):
            site_distances = distances[nonself & (center == site_index)]
            kth[site_index] = float(
                np.partition(site_distances, _base.K_NEIGHBORS - 1)[
                    _base.K_NEIGHBORS - 1
                ]
            )

        thresholds = kth + _tie_margin(kth)
        required = float(np.max(thresholds)) + 4.0 * _base.DISTANCE_TOLERANCE_ANGSTROM
        if required >= cutoff:
            next_cutoff = max(cutoff * 1.25, required * 1.001)
            if next_cutoff > _base.MAX_SEARCH_CUTOFF_ANGSTROM + 1.0e-12:
                raise _base.DescriptorUnavailable(
                    "resource",
                    "tie_complete_shell_beyond_search_limit",
                    (
                        f"tie-complete {_base.K_NEIGHBORS}-neighbor shell needs cutoff "
                        f"{required:.6g} angstrom, above "
                        f"{_base.MAX_SEARCH_CUTOFF_ANGSTROM:g}"
                    ),
                )
            cutoff = next_cutoff
            continue

        shell = nonself & (distances <= thresholds[center])
        shell_center = center[shell]
        shell_point = point[shell]
        shell_offsets = offsets[shell]
        shell_distances = distances[shell]
        coordination = np.bincount(shell_center, minlength=site_count).astype(np.int64)
        if np.any(coordination < _base.K_NEIGHBORS):
            missing = np.flatnonzero(coordination < _base.K_NEIGHBORS).tolist()
            raise _base.DescriptorUnavailable(
                "algorithm",
                "incomplete_knn_shell",
                f"tie-complete shell remained incomplete at sites {missing[:12]}",
            )
        maximum_coordination = int(np.max(coordination))
        if maximum_coordination > _base.MAX_TIE_COMPLETE_COORDINATION:
            raise _base.DescriptorUnavailable(
                "resource",
                "tie_complete_coordination_limit",
                (
                    f"tie-complete coordination {maximum_coordination} exceeds "
                    f"limit {_base.MAX_TIE_COMPLETE_COORDINATION}"
                ),
            )

        cart = np.asarray(structure.cart_coords, dtype=np.float64)
        lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)
        vectors = cart[shell_point] + shell_offsets @ lattice - cart[shell_center]
        reconstructed = np.linalg.norm(vectors, axis=1)
        residual = float(np.max(np.abs(reconstructed - shell_distances)))
        if residual > 2.0e-6:
            raise _base.DescriptorUnavailable(
                "algorithm",
                "neighbor_vector_residual",
                f"periodic vectors disagree with distances by {residual:.3e} angstrom",
            )
        return {
            "center": shell_center,
            "point": shell_point,
            "distances": shell_distances,
            "coordination": coordination,
            "kth_distances": kth,
            "search_cutoff_angstrom": cutoff,
        }

    raise _base.DescriptorUnavailable(
        "algorithm",
        "knn_search_iteration_limit",
        "periodic k-neighbor search did not converge",
    )


def _between_fraction(
    values: np.ndarray,
    species_indices: np.ndarray,
    species_fractions: np.ndarray,
) -> tuple[float, np.ndarray, float, float]:
    global_mean = float(np.mean(values))
    species_means = np.empty(len(species_fractions), dtype=np.float64)
    for species_index in range(len(species_fractions)):
        selected = values[species_indices == species_index]
        if len(selected) < 1:
            raise _base.DescriptorUnavailable(
                "algorithm", "empty_species_group", "element-conditioned group was empty"
            )
        species_means[species_index] = float(np.mean(selected))
    total_variance = float(np.mean((values - global_mean) ** 2))
    between_variance = float(
        np.sum(species_fractions * (species_means - global_mean) ** 2)
    )
    if total_variance <= ELEMENT_CONDITIONED_VARIANCE_FLOOR:
        fraction = 0.0
    else:
        fraction = min(1.0, max(0.0, between_variance / total_variance))
    return fraction, species_means, global_mean, total_variance


_base_compute_descriptor = _base.compute_descriptor
_base._tie_complete_knn_shell = _tie_complete_knn_shell
_base._between_fraction = _between_fraction
_base.TIE_ABSOLUTE_TOLERANCE_ANGSTROM = TIE_ABSOLUTE_TOLERANCE_ANGSTROM
_base.ELEMENT_CONDITIONED_VARIANCE_FLOOR = ELEMENT_CONDITIONED_VARIANCE_FLOOR


def compute_descriptor(structure: Any) -> dict[str, Any]:
    result = _base_compute_descriptor(structure)
    result["parameters"]["tie_absolute_tolerance_angstrom"] = (
        TIE_ABSOLUTE_TOLERANCE_ANGSTROM
    )
    result["parameters"]["element_conditioned_variance_floor"] = (
        ELEMENT_CONDITIONED_VARIANCE_FLOOR
    )
    return result


_base.compute_descriptor = compute_descriptor

SCHEMA = _base.SCHEMA
FEATURE_NAMES = _base.FEATURE_NAMES
REFERENCE_METADATA = _base.REFERENCE_METADATA
DescriptorUnavailable = _base.DescriptorUnavailable
structure_from_record = _base.structure_from_record
structure_from_poscar = _base.structure_from_poscar
descriptor_from_record = _base.descriptor_from_record
main = _base.main


if __name__ == "__main__":
    raise SystemExit(main())
