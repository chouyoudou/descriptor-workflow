"""Element-conditioned local geometry descriptors for periodic structures.

The fixed descriptor uses only an explicit fully occupied periodic structure and
neutral-element atomic numbers.  It decomposes local radial geometry and
central-to-neighbour elemental contrast into within-element and between-element
parts, with a documented composition-surprisal weighting that is sensitive to
explicit minority species without declaring any species to be a solute.

It does not infer grains, deformation history, dislocations, strain, oxidation
state, energetics, kinetics, or response properties.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.optimization.neighbors import find_points_in_spheres


SCHEMA = "element-conditioned-local-geometry/1"
K_NEIGHBORS = 12
DISTANCE_TOLERANCE_ANGSTROM = 1.0e-8
TIE_RELATIVE_TOLERANCE = 2.0e-8
MIN_SEPARATION_ANGSTROM = 0.10
INITIAL_SEARCH_CUTOFF_ANGSTROM = 3.0
MAX_SEARCH_CUTOFF_ANGSTROM = 48.0
MAX_ATOMS = 4096
MAX_PERIODIC_PAIRS = 2_000_000
MAX_TIE_COMPLETE_COORDINATION = 512
ATOMIC_NUMBER_CONTRAST_DENOMINATOR = 117.0

FEATURE_NAMES = (
    "composition_element_count",
    "composition_entropy_norm",
    "composition_atomic_number_std_norm",
    "composition_pairwise_atomic_number_contrast",
    "knn_shell_size_mean",
    "knn_tie_excess_fraction",
    "knn_distance_mean_angstrom",
    "site_log_shell_scale_std",
    "site_radial_log_rms_mean",
    "site_radial_log_rms_std",
    "site_atomic_number_contrast_mean",
    "site_atomic_number_contrast_std",
    "site_contrast_radial_coupling_mean",
    "site_contrast_radial_coupling_std",
    "shell_scale_element_between_fraction",
    "radial_disorder_element_between_fraction",
    "atomic_number_contrast_element_between_fraction",
    "contrast_geometry_element_between_fraction",
    "rarity_weighted_shell_scale_shift",
    "rarity_weighted_radial_disorder_shift",
    "rarity_weighted_atomic_number_contrast_shift",
    "rarity_weighted_contrast_geometry_shift",
    "rarity_weighted_element_environment_contrast",
    "z_weighted_shell_scale_separation",
    "z_weighted_radial_disorder_separation",
    "z_weighted_atomic_number_contrast_separation",
    "z_weighted_contrast_geometry_separation",
    "z_weighted_element_environment_separation",
    "within_element_environment_rms",
    "total_environment_heterogeneity_rms",
)

REFERENCE_METADATA = {
    "elemental_reference": "neutral-element atomic number",
    "normalization": "absolute atomic-number difference divided by 117",
    "provider": "pymatgen Element.Z",
    "pymatgen_version_pin_used_for_formal_validation": "2026.5.4",
    "pymatgen_license": "MIT",
    "oxidation_state_policy": "non-neutral Species and dummy species are rejected",
    "scientific_boundary": (
        "atomic number is an independent elemental label; it is not charge, valence, "
        "bond order, electronic structure, or a fitted response"
    ),
}


class DescriptorUnavailable(RuntimeError):
    """A typed, scientifically meaningful non-result."""

    def __init__(self, category: str, code: str, message: str):
        super().__init__(message)
        self.category = category
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "code": self.code,
            "message": self.message,
        }


def _finite_float(value: Any, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise DescriptorUnavailable(
            "numerical", "non_finite_result", f"{name} was not finite"
        )
    if abs(out) < 1.0e-15:
        return 0.0
    return out


def _normalise_structure(structure: Structure) -> Structure:
    if not isinstance(structure, Structure):
        raise DescriptorUnavailable(
            "input", "not_a_structure", "input is not a pymatgen Structure"
        )
    if len(structure) < 1:
        raise DescriptorUnavailable("input", "empty_structure", "structure has no sites")
    if len(structure) > MAX_ATOMS:
        raise DescriptorUnavailable(
            "resource",
            "atom_limit",
            f"structure has {len(structure)} sites; limit is {MAX_ATOMS}",
        )
    if not structure.is_ordered:
        raise DescriptorUnavailable(
            "definition_domain",
            "partial_occupancy",
            "the fixed descriptor requires explicit fully occupied sites",
        )

    lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise DescriptorUnavailable("input", "invalid_lattice", "lattice is not finite 3x3")
    determinant = float(np.linalg.det(lattice))
    if not math.isfinite(determinant) or abs(determinant) <= 1.0e-10:
        raise DescriptorUnavailable(
            "input", "singular_lattice", "lattice has zero or non-finite volume"
        )

    fractions = np.mod(np.asarray(structure.frac_coords, dtype=np.float64), 1.0)
    if fractions.shape != (len(structure), 3) or not np.isfinite(fractions).all():
        raise DescriptorUnavailable(
            "input", "invalid_coordinates", "fractional coordinates are not finite Nx3"
        )

    species = []
    for index, site in enumerate(structure):
        specie = site.specie
        if type(specie).__name__ != "Element":
            raise DescriptorUnavailable(
                "definition_domain",
                "non_neutral_species_label",
                (
                    f"site {index} uses {type(specie).__name__}; the fixed descriptor "
                    "accepts neutral Element labels only"
                ),
            )
        atomic_number = int(specie.Z)
        if atomic_number < 1 or atomic_number > 118:
            raise DescriptorUnavailable(
                "definition_domain",
                "invalid_atomic_number",
                f"site {index} has atomic number {atomic_number}",
            )
        species.append(specie)

    try:
        return Structure(Lattice(lattice), species, fractions, coords_are_cartesian=False)
    except Exception as exc:
        raise DescriptorUnavailable(
            "input", "invalid_structure", f"could not normalize structure: {exc}"
        ) from exc


def structure_from_record(record: Mapping[str, Any]) -> Structure:
    """Build an ordered periodic structure from a frozen JSONL record."""
    try:
        lattice = record["lattice"]
        species = record["species"]
        fractions = record["fractional_coordinates"]
    except (KeyError, TypeError) as exc:
        raise DescriptorUnavailable(
            "input",
            "missing_structure_field",
            "record must contain lattice, species, and fractional_coordinates",
        ) from exc
    try:
        structure = Structure(
            Lattice(np.asarray(lattice, dtype=np.float64)),
            list(species),
            np.asarray(fractions, dtype=np.float64),
            coords_are_cartesian=False,
        )
    except Exception as exc:
        raise DescriptorUnavailable(
            "input", "invalid_structure_record", f"could not build structure: {exc}"
        ) from exc
    return _normalise_structure(structure)


def structure_from_poscar(path: str | Path) -> Structure:
    try:
        structure = Poscar.from_file(str(path), check_for_potcar=False).structure
    except Exception as exc:
        raise DescriptorUnavailable(
            "input", "invalid_poscar", f"could not read POSCAR: {exc}"
        ) from exc
    return _normalise_structure(structure)


def _raw_periodic_pairs(
    structure: Structure, cutoff_angstrom: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.ascontiguousarray(structure.cart_coords, dtype=np.float64)
    lattice = np.ascontiguousarray(structure.lattice.matrix, dtype=np.float64)
    pbc = np.ascontiguousarray([1, 1, 1], dtype=np.int64)
    try:
        center, point, offsets, distances = find_points_in_spheres(
            all_coords=coordinates,
            center_coords=coordinates,
            r=float(cutoff_angstrom),
            pbc=pbc,
            lattice=lattice,
            tol=DISTANCE_TOLERANCE_ANGSTROM,
        )
    except MemoryError as exc:
        raise DescriptorUnavailable(
            "resource",
            "periodic_neighbor_memory",
            "periodic neighbor search exhausted memory",
        ) from exc
    except Exception as exc:
        raise DescriptorUnavailable(
            "algorithm",
            "periodic_neighbor_search",
            f"pymatgen periodic neighbor search failed: {exc}",
        ) from exc

    center = np.asarray(center, dtype=np.int64)
    point = np.asarray(point, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    if len(center) > MAX_PERIODIC_PAIRS:
        raise DescriptorUnavailable(
            "resource",
            "periodic_pair_limit",
            f"periodic search produced {len(center)} pairs; limit is {MAX_PERIODIC_PAIRS}",
        )
    if not (
        center.ndim == point.ndim == distances.ndim == 1
        and offsets.shape == (len(center), 3)
        and len(center) == len(point) == len(distances)
        and np.isfinite(offsets).all()
        and np.isfinite(distances).all()
    ):
        raise DescriptorUnavailable(
            "algorithm", "malformed_neighbor_list", "periodic neighbor arrays are malformed"
        )
    return center, point, offsets, distances


def _tie_complete_knn_shell(structure: Structure) -> dict[str, Any]:
    site_count = len(structure)
    density_scale = float((structure.volume / site_count) ** (1.0 / 3.0))
    cutoff = max(INITIAL_SEARCH_CUTOFF_ANGSTROM, 2.1 * density_scale)
    cutoff = min(cutoff, MAX_SEARCH_CUTOFF_ANGSTROM)

    for _ in range(14):
        center, point, offsets, distances = _raw_periodic_pairs(structure, cutoff)
        zero_offset = np.all(np.abs(offsets) <= 1.0e-12, axis=1)
        exact_self = (center == point) & zero_offset & (distances <= 1.0e-7)
        overlap = (~exact_self) & (distances < MIN_SEPARATION_ANGSTROM)
        if np.any(overlap):
            first = int(np.flatnonzero(overlap)[0])
            raise DescriptorUnavailable(
                "input",
                "overlapping_periodic_sites",
                (
                    "distinct sites/images are closer than "
                    f"{MIN_SEPARATION_ANGSTROM:.2f} angstrom "
                    f"(sites {int(center[first])} and {int(point[first])}, "
                    f"distance {distances[first]:.6g})"
                ),
            )

        nonself = (~exact_self) & (distances > DISTANCE_TOLERANCE_ANGSTROM)
        counts = np.bincount(center[nonself], minlength=site_count).astype(np.int64)
        if np.any(counts < K_NEIGHBORS):
            next_cutoff = cutoff * 1.65
            if next_cutoff > MAX_SEARCH_CUTOFF_ANGSTROM + 1.0e-12:
                missing = np.flatnonzero(counts < K_NEIGHBORS).tolist()
                raise DescriptorUnavailable(
                    "resource",
                    "knn_beyond_search_limit",
                    (
                        f"sites {missing[:12]} have fewer than {K_NEIGHBORS} periodic "
                        f"neighbors within {MAX_SEARCH_CUTOFF_ANGSTROM:g} angstrom"
                    ),
                )
            cutoff = next_cutoff
            continue

        kth = np.empty(site_count, dtype=np.float64)
        for site_index in range(site_count):
            site_distances = distances[nonself & (center == site_index)]
            kth[site_index] = float(
                np.partition(site_distances, K_NEIGHBORS - 1)[K_NEIGHBORS - 1]
            )

        required = float(np.max(kth)) * (1.0 + TIE_RELATIVE_TOLERANCE) + (
            20.0 * DISTANCE_TOLERANCE_ANGSTROM
        )
        if required >= cutoff:
            next_cutoff = max(cutoff * 1.25, required * 1.001)
            if next_cutoff > MAX_SEARCH_CUTOFF_ANGSTROM + 1.0e-12:
                raise DescriptorUnavailable(
                    "resource",
                    "tie_complete_shell_beyond_search_limit",
                    (
                        f"tie-complete {K_NEIGHBORS}-neighbor shell needs cutoff "
                        f"{required:.6g} angstrom, above {MAX_SEARCH_CUTOFF_ANGSTROM:g}"
                    ),
                )
            cutoff = next_cutoff
            continue

        shell = nonself & (
            distances
            <= kth[center] * (1.0 + TIE_RELATIVE_TOLERANCE)
            + 20.0 * DISTANCE_TOLERANCE_ANGSTROM
        )
        shell_center = center[shell]
        shell_point = point[shell]
        shell_offsets = offsets[shell]
        shell_distances = distances[shell]
        coordination = np.bincount(shell_center, minlength=site_count).astype(np.int64)
        if np.any(coordination < K_NEIGHBORS):
            missing = np.flatnonzero(coordination < K_NEIGHBORS).tolist()
            raise DescriptorUnavailable(
                "algorithm",
                "incomplete_knn_shell",
                f"tie-complete shell remained incomplete at sites {missing[:12]}",
            )
        maximum_coordination = int(np.max(coordination))
        if maximum_coordination > MAX_TIE_COMPLETE_COORDINATION:
            raise DescriptorUnavailable(
                "resource",
                "tie_complete_coordination_limit",
                (
                    f"tie-complete coordination {maximum_coordination} exceeds "
                    f"limit {MAX_TIE_COMPLETE_COORDINATION}"
                ),
            )

        cart = np.asarray(structure.cart_coords, dtype=np.float64)
        lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)
        vectors = cart[shell_point] + shell_offsets @ lattice - cart[shell_center]
        reconstructed = np.linalg.norm(vectors, axis=1)
        residual = float(np.max(np.abs(reconstructed - shell_distances)))
        if residual > 2.0e-6:
            raise DescriptorUnavailable(
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

    raise DescriptorUnavailable(
        "algorithm", "knn_search_iteration_limit", "periodic k-neighbor search did not converge"
    )


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    if len(positive) == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise DescriptorUnavailable(
            "algorithm", "invalid_site_values", "site-value vector is empty or non-finite"
        )
    return float(np.mean(values)), float(np.std(values, ddof=0))


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
            raise DescriptorUnavailable(
                "algorithm", "empty_species_group", "element-conditioned group was empty"
            )
        species_means[species_index] = float(np.mean(selected))
    total_variance = float(np.mean((values - global_mean) ** 2))
    between_variance = float(
        np.sum(species_fractions * (species_means - global_mean) ** 2)
    )
    if total_variance <= 1.0e-30:
        fraction = 0.0
    else:
        fraction = min(1.0, max(0.0, between_variance / total_variance))
    return fraction, species_means, global_mean, total_variance


def _rarity_weights(species_fractions_by_site: np.ndarray) -> tuple[np.ndarray, bool]:
    rarity = -np.log(species_fractions_by_site)
    total = float(np.sum(rarity))
    if total <= 1.0e-15:
        return np.full(len(rarity), 1.0 / len(rarity), dtype=np.float64), False
    return rarity / total, True


def compute_descriptor(structure: Structure) -> dict[str, Any]:
    """Compute the fixed 30-scalar descriptor."""
    structure = _normalise_structure(structure)
    shell = _tie_complete_knn_shell(structure)
    center = shell["center"]
    point = shell["point"]
    distances = shell["distances"]
    coordination = shell["coordination"]
    site_count = len(structure)

    symbols = [site.specie.symbol for site in structure]
    atomic_numbers = np.asarray([int(site.specie.Z) for site in structure], dtype=np.float64)
    counts = Counter(symbols)
    element_symbols = sorted(counts, key=lambda symbol: int(next(
        site.specie.Z for site in structure if site.specie.symbol == symbol
    )))
    element_count = len(element_symbols)
    symbol_to_index = {symbol: index for index, symbol in enumerate(element_symbols)}
    species_indices = np.asarray([symbol_to_index[symbol] for symbol in symbols], dtype=np.int64)
    species_fractions = np.asarray(
        [counts[symbol] / site_count for symbol in element_symbols], dtype=np.float64
    )
    species_atomic_numbers = np.asarray(
        [atomic_numbers[symbols.index(symbol)] for symbol in element_symbols],
        dtype=np.float64,
    )

    composition_entropy = _entropy(species_fractions)
    composition_entropy_norm = (
        composition_entropy / math.log(element_count) if element_count > 1 else 0.0
    )
    composition_z_mean = float(np.sum(species_fractions * species_atomic_numbers))
    composition_z_std_norm = math.sqrt(
        float(np.sum(species_fractions * (species_atomic_numbers - composition_z_mean) ** 2))
    ) / ATOMIC_NUMBER_CONTRAST_DENOMINATOR
    pairwise_z_contrast = float(
        np.sum(
            species_fractions[:, None]
            * species_fractions[None, :]
            * np.abs(species_atomic_numbers[:, None] - species_atomic_numbers[None, :])
        )
        / ATOMIC_NUMBER_CONTRAST_DENOMINATOR
    )

    site_log_geometric_scale = np.empty(site_count, dtype=np.float64)
    site_radial_log_rms = np.empty(site_count, dtype=np.float64)
    site_z_contrast = np.empty(site_count, dtype=np.float64)
    site_contrast_radial_coupling = np.empty(site_count, dtype=np.float64)
    site_mean_distance = np.empty(site_count, dtype=np.float64)

    for site_index in range(site_count):
        selected = center == site_index
        local_distances = distances[selected]
        local_points = point[selected]
        if len(local_distances) < K_NEIGHBORS:
            raise DescriptorUnavailable(
                "algorithm", "empty_site_shell", f"site {site_index} has incomplete shell"
            )
        log_distances = np.log(local_distances)
        log_scale = float(np.mean(log_distances))
        radial_residual = log_distances - log_scale
        contrast = (
            np.abs(atomic_numbers[local_points] - atomic_numbers[site_index])
            / ATOMIC_NUMBER_CONTRAST_DENOMINATOR
        )
        site_log_geometric_scale[site_index] = log_scale
        site_radial_log_rms[site_index] = math.sqrt(
            float(np.mean(radial_residual**2))
        )
        site_z_contrast[site_index] = float(np.mean(contrast))
        site_contrast_radial_coupling[site_index] = float(
            np.mean(contrast * np.abs(radial_residual))
        )
        site_mean_distance[site_index] = float(np.mean(local_distances))

    global_log_scale = float(np.mean(site_log_geometric_scale))
    site_log_scale = site_log_geometric_scale - global_log_scale

    log_scale_mean, log_scale_std = _mean_std(site_log_scale)
    radial_mean, radial_std = _mean_std(site_radial_log_rms)
    z_contrast_mean, z_contrast_std = _mean_std(site_z_contrast)
    coupling_mean, coupling_std = _mean_std(site_contrast_radial_coupling)
    if abs(log_scale_mean) > 2.0e-12:
        raise DescriptorUnavailable(
            "algorithm",
            "log_scale_centering_residual",
            f"centered site log scale has mean {log_scale_mean:.3e}",
        )

    scale_fraction, scale_species_means, scale_global, _ = _between_fraction(
        site_log_scale, species_indices, species_fractions
    )
    radial_fraction, radial_species_means, radial_global, _ = _between_fraction(
        site_radial_log_rms, species_indices, species_fractions
    )
    contrast_fraction, contrast_species_means, contrast_global, _ = _between_fraction(
        site_z_contrast, species_indices, species_fractions
    )
    coupling_fraction, coupling_species_means, coupling_global, _ = _between_fraction(
        site_contrast_radial_coupling, species_indices, species_fractions
    )

    species_feature_means = np.column_stack(
        [
            scale_species_means,
            radial_species_means,
            contrast_species_means,
            coupling_species_means,
        ]
    )
    global_feature_mean = np.asarray(
        [scale_global, radial_global, contrast_global, coupling_global], dtype=np.float64
    )
    species_shifts = species_feature_means - global_feature_mean[None, :]
    site_shifts = species_shifts[species_indices]

    fractions_by_site = species_fractions[species_indices]
    rarity_weights, rarity_active = _rarity_weights(fractions_by_site)
    rarity_shift = np.sum(rarity_weights[:, None] * np.abs(site_shifts), axis=0)
    rarity_environment = float(
        np.sum(rarity_weights * np.linalg.norm(site_shifts, axis=1))
    )

    z_pair_weight = (
        species_fractions[:, None]
        * species_fractions[None, :]
        * np.abs(species_atomic_numbers[:, None] - species_atomic_numbers[None, :])
        / ATOMIC_NUMBER_CONTRAST_DENOMINATOR
    )
    pair_feature_difference = np.abs(
        species_feature_means[:, None, :] - species_feature_means[None, :, :]
    )
    z_weighted_feature_separation = np.sum(
        z_pair_weight[:, :, None] * pair_feature_difference, axis=(0, 1)
    )
    z_weighted_environment_separation = float(
        np.sum(
            z_pair_weight
            * np.linalg.norm(
                species_feature_means[:, None, :] - species_feature_means[None, :, :],
                axis=2,
            )
        )
    )

    site_feature_matrix = np.column_stack(
        [
            site_log_scale,
            site_radial_log_rms,
            site_z_contrast,
            site_contrast_radial_coupling,
        ]
    )
    species_mean_by_site = species_feature_means[species_indices]
    within_environment_rms = math.sqrt(
        float(np.mean(np.sum((site_feature_matrix - species_mean_by_site) ** 2, axis=1)))
    )
    total_environment_rms = math.sqrt(
        float(np.mean(np.sum((site_feature_matrix - global_feature_mean[None, :]) ** 2, axis=1)))
    )

    feature_values = (
        float(element_count),
        composition_entropy_norm,
        composition_z_std_norm,
        pairwise_z_contrast,
        float(np.mean(coordination)),
        float(np.mean((coordination - K_NEIGHBORS) / coordination)),
        float(np.mean(site_mean_distance)),
        log_scale_std,
        radial_mean,
        radial_std,
        z_contrast_mean,
        z_contrast_std,
        coupling_mean,
        coupling_std,
        scale_fraction,
        radial_fraction,
        contrast_fraction,
        coupling_fraction,
        float(rarity_shift[0]),
        float(rarity_shift[1]),
        float(rarity_shift[2]),
        float(rarity_shift[3]),
        rarity_environment,
        float(z_weighted_feature_separation[0]),
        float(z_weighted_feature_separation[1]),
        float(z_weighted_feature_separation[2]),
        float(z_weighted_feature_separation[3]),
        z_weighted_environment_separation,
        within_environment_rms,
        total_environment_rms,
    )
    features = {
        name: _finite_float(value, name=name)
        for name, value in zip(FEATURE_NAMES, feature_values, strict=True)
    }

    return {
        "schema": SCHEMA,
        "features": features,
        "parameters": {
            "k_neighbors": K_NEIGHBORS,
            "tie_relative_tolerance": TIE_RELATIVE_TOLERANCE,
            "atomic_number_contrast_denominator": ATOMIC_NUMBER_CONTRAST_DENOMINATOR,
            "rarity_weight": "site surprisal -ln(global atomic fraction of central element)",
        },
        "reference": dict(REFERENCE_METADATA),
        "diagnostics": {
            "site_count": site_count,
            "element_count": element_count,
            "directed_shell_edge_count": int(len(distances)),
            "shell_size_min": int(np.min(coordination)),
            "shell_size_max": int(np.max(coordination)),
            "search_cutoff_angstrom": _finite_float(
                shell["search_cutoff_angstrom"], name="search_cutoff_angstrom"
            ),
            "rarity_weighting_active": rarity_active,
            "scope_warning": (
                "minority-sensitive weighting is not a solute classifier; stoichiometric "
                "minority sublattices are treated by the same formula"
            ),
        },
    }


def descriptor_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return compute_descriptor(structure_from_record(record))


def _json_line(data: Any) -> str:
    return json.dumps(data, sort_keys=False, separators=(",", ":"), allow_nan=False)


def _process_jsonl(input_path: Path, output_path: Path | None) -> int:
    rows: list[str] = []
    unavailable = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for physical_line, text in enumerate(handle, start=1):
            if not text.strip():
                continue
            record_id = f"line-{physical_line}"
            try:
                record = json.loads(text)
                if not isinstance(record, dict):
                    raise DescriptorUnavailable(
                        "input", "json_record_not_object", "JSONL record is not an object"
                    )
                record_id = str(record.get("id", record_id))
                row = {
                    "id": record_id,
                    "status": "ok",
                    "descriptor": descriptor_from_record(record),
                }
            except DescriptorUnavailable as exc:
                unavailable += 1
                row = {"id": record_id, "status": "unavailable", "reason": exc.as_dict()}
            except Exception as exc:
                unavailable += 1
                row = {
                    "id": record_id,
                    "status": "unavailable",
                    "reason": {
                        "category": "input",
                        "code": "invalid_json_record",
                        "message": f"{type(exc).__name__}: {exc}",
                    },
                }
            rows.append(_json_line(row))
    rendered = "\n".join(rows) + ("\n" if rows else "")
    if output_path is None:
        print(rendered, end="")
    else:
        output_path.write_text(rendered, encoding="utf-8")
    return 1 if unavailable else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Element-conditioned local geometry descriptor"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--format", choices=("auto", "poscar", "jsonl"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    input_format = args.format
    if input_format == "auto":
        input_format = "jsonl" if args.input.suffix.lower() == ".jsonl" else "poscar"
    if input_format == "jsonl":
        return _process_jsonl(args.input, args.output)

    try:
        payload = {
            "status": "ok",
            "descriptor": compute_descriptor(structure_from_poscar(args.input)),
        }
        exit_code = 0
    except DescriptorUnavailable as exc:
        payload = {"status": "unavailable", "reason": exc.as_dict()}
        exit_code = 1
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
