"""Intrinsic first-shell geometry and directionality descriptors.

The fixed descriptor is target-blind and uses only a fully occupied periodic
structure plus atomic numbers that are independently tabulated by pymatgen.  It
does not infer texture, grains, domains, loading history, experimental strain,
phase fractions, oxidation states, or response properties.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.optimization.neighbors import find_points_in_spheres


SCHEMA = "intrinsic-first-shell-directionality/1"
PAIRED_SCHEMA = "explicit-paired-structure-change/1"
SHELL_FACTOR = 1.25
DISTANCE_TOLERANCE_ANGSTROM = 1.0e-8
MIN_SEPARATION_ANGSTROM = 0.10
INITIAL_SEARCH_CUTOFF_ANGSTROM = 3.0
MAX_SEARCH_CUTOFF_ANGSTROM = 48.0
MAX_ATOMS = 4096
MAX_PERIODIC_PAIRS = 2_000_000
MAX_FIRST_SHELL_COORDINATION = 256

FEATURE_NAMES = (
    "first_shell_coordination_mean",
    "first_shell_coordination_std",
    "first_shell_distance_mean_angstrom",
    "first_shell_distance_cv",
    "baur_first_shell_distortion_mean",
    "baur_first_shell_distortion_rms",
    "baur_first_shell_distortion_max",
    "first_shell_offcentering_mean",
    "first_shell_offcentering_rms",
    "first_shell_offcentering_max",
    "first_shell_offcentering_alignment",
    "bond_fabric_q2",
    "bond_fabric_linearity",
    "bond_fabric_planarity",
    "bond_fabric_isotropy",
    "site_fabric_q2_mean",
    "site_fabric_q2_std",
    "site_fabric_q2_bond_weighted_mean",
    "site_to_global_q2_cancellation",
    "heterochemical_first_shell_fraction",
    "atomic_number_contrast_mean",
    "atomic_number_contrast_fabric_q2",
    "atomic_number_contrast_q2_shift",
)


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
            "numerical",
            "non_finite_result",
            f"{name} was not finite",
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
    species = [site.specie for site in structure]
    return Structure(Lattice(lattice), species, fractions, coords_are_cartesian=False)


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
        return _normalise_structure(
            Structure(
                Lattice(np.asarray(lattice, dtype=np.float64)),
                list(species),
                np.asarray(fractions, dtype=np.float64),
                coords_are_cartesian=False,
            )
        )
    except DescriptorUnavailable:
        raise
    except Exception as exc:  # pymatgen supplies detailed validation internally
        raise DescriptorUnavailable(
            "input", "invalid_structure_record", f"could not build structure: {exc}"
        ) from exc


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
            f"periodic neighbor search produced {len(center)} pairs; limit is {MAX_PERIODIC_PAIRS}",
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


def _adaptive_first_shell(structure: Structure) -> dict[str, Any]:
    site_count = len(structure)
    cutoff = INITIAL_SEARCH_CUTOFF_ANGSTROM

    for _ in range(12):
        center, point, offsets, distances = _raw_periodic_pairs(structure, cutoff)
        zero_offset = np.all(np.abs(offsets) <= 1.0e-12, axis=1)
        exact_self = (center == point) & zero_offset & (distances <= 1.0e-7)
        overlap = (~exact_self) & (distances < MIN_SEPARATION_ANGSTROM)
        if np.any(overlap):
            i = int(np.flatnonzero(overlap)[0])
            raise DescriptorUnavailable(
                "input",
                "overlapping_periodic_sites",
                (
                    "distinct periodic sites/images are closer than "
                    f"{MIN_SEPARATION_ANGSTROM:.2f} angstrom "
                    f"(sites {int(center[i])} and {int(point[i])}, distance {distances[i]:.6g})"
                ),
            )

        nonself = (~exact_self) & (distances > DISTANCE_TOLERANCE_ANGSTROM)
        nearest = np.full(site_count, np.inf, dtype=np.float64)
        if np.any(nonself):
            np.minimum.at(nearest, center[nonself], distances[nonself])

        if not np.isfinite(nearest).all():
            next_cutoff = cutoff * 2.0
            if next_cutoff > MAX_SEARCH_CUTOFF_ANGSTROM + 1.0e-12:
                missing = np.flatnonzero(~np.isfinite(nearest)).tolist()
                raise DescriptorUnavailable(
                    "resource",
                    "nearest_neighbor_beyond_search_limit",
                    (
                        "no periodic neighbor was found for sites "
                        f"{missing[:12]} within {MAX_SEARCH_CUTOFF_ANGSTROM:g} angstrom"
                    ),
                )
            cutoff = next_cutoff
            continue

        required = float(np.max(SHELL_FACTOR * nearest)) + 10.0 * DISTANCE_TOLERANCE_ANGSTROM
        if required > cutoff:
            next_cutoff = max(cutoff * 1.5, required * (1.0 + 1.0e-10))
            if next_cutoff > MAX_SEARCH_CUTOFF_ANGSTROM + 1.0e-12:
                raise DescriptorUnavailable(
                    "resource",
                    "first_shell_beyond_search_limit",
                    (
                        f"adaptive first shell needs cutoff {required:.6g} angstrom, "
                        f"above limit {MAX_SEARCH_CUTOFF_ANGSTROM:g}"
                    ),
                )
            cutoff = next_cutoff
            continue

        shell = nonself & (
            distances <= SHELL_FACTOR * nearest[center] + 10.0 * DISTANCE_TOLERANCE_ANGSTROM
        )
        shell_center = center[shell]
        shell_point = point[shell]
        shell_offsets = offsets[shell]
        shell_distances = distances[shell]
        coordination = np.bincount(shell_center, minlength=site_count).astype(np.int64)
        if np.any(coordination < 1):
            missing = np.flatnonzero(coordination < 1).tolist()
            raise DescriptorUnavailable(
                "algorithm",
                "empty_first_shell",
                f"adaptive first shell was empty for sites {missing[:12]}",
            )
        maximum_coordination = int(np.max(coordination))
        if maximum_coordination > MAX_FIRST_SHELL_COORDINATION:
            raise DescriptorUnavailable(
                "resource",
                "first_shell_coordination_limit",
                (
                    f"first-shell coordination {maximum_coordination} exceeds "
                    f"limit {MAX_FIRST_SHELL_COORDINATION}"
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
                f"periodic neighbor vectors disagree with distances by {residual:.3e} angstrom",
            )
        return {
            "center": shell_center,
            "point": shell_point,
            "offsets": shell_offsets,
            "distances": shell_distances,
            "vectors": vectors,
            "coordination": coordination,
            "nearest_distances": nearest,
            "search_cutoff_angstrom": cutoff,
        }

    raise DescriptorUnavailable(
        "algorithm", "adaptive_search_iteration_limit", "adaptive first-shell search did not converge"
    )


def _fabric(
    vectors: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, Any]:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or len(vectors) < 1:
        raise DescriptorUnavailable(
            "algorithm", "empty_fabric", "fabric tensor requires at least one 3D vector"
        )
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= DISTANCE_TOLERANCE_ANGSTROM):
        raise DescriptorUnavailable(
            "input", "zero_length_shell_vector", "first shell contains a zero-length vector"
        )
    unit = vectors / norms[:, None]
    if weights is None:
        weight = np.ones(len(vectors), dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64)
        if weight.shape != (len(vectors),) or not np.isfinite(weight).all() or np.any(weight < 0):
            raise DescriptorUnavailable(
                "input", "invalid_fabric_weights", "fabric weights must be finite and nonnegative"
            )
    total = float(np.sum(weight))
    if total <= 1.0e-15:
        raise DescriptorUnavailable(
            "definition_domain", "zero_fabric_weight", "fabric weights sum to zero"
        )
    tensor = np.einsum("n,ni,nj->ij", weight, unit, unit, optimize=True) / total
    tensor = 0.5 * (tensor + tensor.T)
    eigenvalues = np.linalg.eigvalsh(tensor)
    eigenvalues[np.abs(eigenvalues) < 1.0e-14] = 0.0
    if np.min(eigenvalues) < -1.0e-10 or abs(float(np.sum(eigenvalues)) - 1.0) > 1.0e-9:
        raise DescriptorUnavailable(
            "numerical", "invalid_fabric_spectrum", "fabric tensor spectrum is not positive trace-one"
        )
    eigenvalues = np.clip(eigenvalues, 0.0, 1.0)
    eigenvalues = eigenvalues / np.sum(eigenvalues)
    l1, l2, l3 = (float(x) for x in eigenvalues)
    q2 = math.sqrt(max(0.0, 1.5 * sum((x - 1.0 / 3.0) ** 2 for x in eigenvalues)))
    linearity = l3 - l2
    planarity = 2.0 * (l2 - l1)
    isotropy = 3.0 * l1
    return {
        "tensor": tensor,
        "eigenvalues": eigenvalues,
        "q2": _finite_float(q2, name="fabric q2"),
        "linearity": _finite_float(linearity, name="fabric linearity"),
        "planarity": _finite_float(planarity, name="fabric planarity"),
        "isotropy": _finite_float(isotropy, name="fabric isotropy"),
    }


def _site_element_data(structure: Structure) -> tuple[list[str], np.ndarray, bool]:
    symbols: list[str] = []
    atomic_numbers: list[float] = []
    complete = True
    for site in structure:
        specie = site.specie
        symbol = getattr(specie, "symbol", None)
        symbols.append(str(symbol if symbol is not None else specie))
        number = getattr(specie, "Z", None)
        try:
            value = float(number)
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value) or value <= 0:
            complete = False
            value = math.nan
        atomic_numbers.append(value)
    return symbols, np.asarray(atomic_numbers, dtype=np.float64), complete


def compute_descriptor(structure: Structure) -> dict[str, Any]:
    """Compute the fixed single-structure descriptor contract."""
    structure = _normalise_structure(structure)
    shell = _adaptive_first_shell(structure)
    center = shell["center"]
    point = shell["point"]
    vectors = shell["vectors"]
    distances = shell["distances"]
    coordination = shell["coordination"]
    site_count = len(structure)

    global_fabric = _fabric(vectors)
    site_baur = np.empty(site_count, dtype=np.float64)
    site_offcentering = np.empty(site_count, dtype=np.float64)
    site_off_vectors = np.empty((site_count, 3), dtype=np.float64)
    site_q2 = np.empty(site_count, dtype=np.float64)

    for site_index in range(site_count):
        selected = center == site_index
        site_distances = distances[selected]
        site_vectors = vectors[selected]
        mean_distance = float(np.mean(site_distances))
        site_baur[site_index] = float(
            np.mean(np.abs(site_distances - mean_distance) / mean_distance)
        )
        mean_vector = np.mean(site_vectors, axis=0)
        scaled_off_vector = mean_vector / mean_distance
        site_off_vectors[site_index] = scaled_off_vector
        site_offcentering[site_index] = float(np.linalg.norm(scaled_off_vector))
        site_q2[site_index] = float(_fabric(site_vectors)["q2"])

    all_distance_mean = float(np.mean(distances))
    all_distance_std = float(np.std(distances, ddof=0))
    q2_bond_weighted_mean = float(np.average(site_q2, weights=coordination))
    cancellation = q2_bond_weighted_mean - float(global_fabric["q2"])
    if cancellation < -1.0e-10:
        raise DescriptorUnavailable(
            "numerical",
            "negative_q2_cancellation",
            f"site/global q2 cancellation was unexpectedly negative ({cancellation:.3e})",
        )
    cancellation = max(0.0, cancellation)

    features: dict[str, float | None] = {
        "first_shell_coordination_mean": _finite_float(
            np.mean(coordination), name="coordination mean"
        ),
        "first_shell_coordination_std": _finite_float(
            np.std(coordination, ddof=0), name="coordination std"
        ),
        "first_shell_distance_mean_angstrom": _finite_float(
            all_distance_mean, name="shell distance mean"
        ),
        "first_shell_distance_cv": _finite_float(
            all_distance_std / all_distance_mean, name="shell distance cv"
        ),
        "baur_first_shell_distortion_mean": _finite_float(
            np.mean(site_baur), name="Baur distortion mean"
        ),
        "baur_first_shell_distortion_rms": _finite_float(
            math.sqrt(float(np.mean(site_baur**2))), name="Baur distortion rms"
        ),
        "baur_first_shell_distortion_max": _finite_float(
            np.max(site_baur), name="Baur distortion max"
        ),
        "first_shell_offcentering_mean": _finite_float(
            np.mean(site_offcentering), name="offcentering mean"
        ),
        "first_shell_offcentering_rms": _finite_float(
            math.sqrt(float(np.mean(site_offcentering**2))), name="offcentering rms"
        ),
        "first_shell_offcentering_max": _finite_float(
            np.max(site_offcentering), name="offcentering max"
        ),
        "first_shell_offcentering_alignment": None,
        "bond_fabric_q2": global_fabric["q2"],
        "bond_fabric_linearity": global_fabric["linearity"],
        "bond_fabric_planarity": global_fabric["planarity"],
        "bond_fabric_isotropy": global_fabric["isotropy"],
        "site_fabric_q2_mean": _finite_float(np.mean(site_q2), name="site q2 mean"),
        "site_fabric_q2_std": _finite_float(np.std(site_q2, ddof=0), name="site q2 std"),
        "site_fabric_q2_bond_weighted_mean": _finite_float(
            q2_bond_weighted_mean, name="bond-weighted site q2 mean"
        ),
        "site_to_global_q2_cancellation": _finite_float(
            cancellation, name="site-to-global q2 cancellation"
        ),
        "heterochemical_first_shell_fraction": None,
        "atomic_number_contrast_mean": None,
        "atomic_number_contrast_fabric_q2": None,
        "atomic_number_contrast_q2_shift": None,
    }
    unavailable: dict[str, dict[str, str]] = {}

    off_norms = np.linalg.norm(site_off_vectors, axis=1)
    total_off_norm = float(np.sum(off_norms))
    if total_off_norm <= 1.0e-14:
        unavailable["first_shell_offcentering_alignment"] = {
            "category": "definition_domain",
            "code": "zero_offcentering_norm",
            "message": "all local first-shell off-centering vectors have zero norm",
        }
    else:
        features["first_shell_offcentering_alignment"] = _finite_float(
            np.linalg.norm(np.sum(site_off_vectors, axis=0)) / total_off_norm,
            name="offcentering alignment",
        )

    symbols, atomic_numbers, complete_atomic_numbers = _site_element_data(structure)
    heterochemical = np.fromiter(
        (symbols[int(i)] != symbols[int(j)] for i, j in zip(center, point, strict=True)),
        dtype=np.float64,
        count=len(center),
    )
    features["heterochemical_first_shell_fraction"] = _finite_float(
        np.mean(heterochemical), name="heterochemical shell fraction"
    )

    if not complete_atomic_numbers:
        for feature_name in (
            "atomic_number_contrast_mean",
            "atomic_number_contrast_fabric_q2",
            "atomic_number_contrast_q2_shift",
        ):
            unavailable[feature_name] = {
                "category": "reference",
                "code": "atomic_number_unavailable",
                "message": "at least one explicit species has no positive tabulated atomic number",
            }
    else:
        z_center = atomic_numbers[center]
        z_point = atomic_numbers[point]
        contrast = np.abs(z_center - z_point) / (z_center + z_point)
        features["atomic_number_contrast_mean"] = _finite_float(
            np.mean(contrast), name="atomic-number contrast mean"
        )
        contrast_sum = float(np.sum(contrast))
        if contrast_sum <= 1.0e-15:
            for feature_name in (
                "atomic_number_contrast_fabric_q2",
                "atomic_number_contrast_q2_shift",
            ):
                unavailable[feature_name] = {
                    "category": "definition_domain",
                    "code": "zero_atomic_number_contrast",
                    "message": "all selected first-shell pairs have identical atomic numbers",
                }
        else:
            contrast_fabric = _fabric(vectors, contrast)
            features["atomic_number_contrast_fabric_q2"] = contrast_fabric["q2"]
            features["atomic_number_contrast_q2_shift"] = _finite_float(
                float(contrast_fabric["q2"]) - float(global_fabric["q2"]),
                name="atomic-number contrast q2 shift",
            )

    if tuple(features) != FEATURE_NAMES:
        raise DescriptorUnavailable(
            "algorithm", "feature_contract_order", "internal feature order differs from contract"
        )
    for name, value in features.items():
        if value is None:
            if name not in unavailable:
                raise DescriptorUnavailable(
                    "algorithm",
                    "untyped_null",
                    f"feature {name} is null without a typed reason",
                )
        elif not math.isfinite(float(value)):
            raise DescriptorUnavailable(
                "numerical", "non_finite_feature", f"feature {name} is not finite"
            )

    return {
        "schema": SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "features": features,
        "unavailable": unavailable,
        "metadata": {
            "atom_count": site_count,
            "directed_first_shell_pair_count": int(len(center)),
            "shell_factor_times_nearest_distance": SHELL_FACTOR,
            "search_cutoff_angstrom": _finite_float(
                shell["search_cutoff_angstrom"], name="search cutoff"
            ),
            "periodic_neighbor_backend": "pymatgen.optimization.neighbors.find_points_in_spheres",
            "fabric_eigenvalues_ascending": [
                _finite_float(value, name="fabric eigenvalue")
                for value in global_fabric["eigenvalues"]
            ],
            "scope": (
                "single explicit periodic structure; no texture, grains, domains, loading, "
                "phase fractions, experimental strain, frequency, or response inference"
            ),
        },
    }


def compare_explicit_structures(
    reference: Structure,
    target: Structure,
    *,
    assume_lattice_correspondence: bool = False,
) -> dict[str, Any]:
    """Conditional paired-state extension with caller-asserted lattice correspondence."""
    if not assume_lattice_correspondence:
        raise DescriptorUnavailable(
            "required_external_input",
            "lattice_correspondence_not_asserted",
            "paired strain requires assume_lattice_correspondence=True from the caller",
        )
    reference = _normalise_structure(reference)
    target = _normalise_structure(target)
    if reference.composition.element_composition != target.composition.element_composition:
        raise DescriptorUnavailable(
            "input",
            "composition_mismatch",
            "paired structures must have the same explicit elemental composition",
        )

    reference_lattice = np.asarray(reference.lattice.matrix, dtype=np.float64)
    target_lattice = np.asarray(target.lattice.matrix, dtype=np.float64)
    try:
        deformation = np.linalg.solve(reference_lattice, target_lattice)
    except np.linalg.LinAlgError as exc:
        raise DescriptorUnavailable(
            "numerical", "lattice_solve_failed", "could not solve paired lattice deformation"
        ) from exc
    determinant = float(np.linalg.det(deformation))
    if not math.isfinite(determinant) or determinant <= 0:
        raise DescriptorUnavailable(
            "definition_domain",
            "non_positive_deformation_determinant",
            "paired lattice deformation must preserve orientation and positive volume",
        )
    right_metric = deformation @ deformation.T
    eigenvalues = np.linalg.eigvalsh(0.5 * (right_metric + right_metric.T))
    if np.any(eigenvalues <= 0) or not np.isfinite(eigenvalues).all():
        raise DescriptorUnavailable(
            "numerical", "invalid_stretch_spectrum", "paired stretch spectrum is not positive"
        )
    principal_log_strains = 0.5 * np.log(eigenvalues)
    mean_log_strain = float(np.mean(principal_log_strains))
    deviator = principal_log_strains - mean_log_strain

    reference_descriptor = compute_descriptor(reference)
    target_descriptor = compute_descriptor(target)
    deltas: dict[str, float | None] = {}
    delta_unavailable: dict[str, dict[str, str]] = {}
    for name in FEATURE_NAMES:
        left = reference_descriptor["features"][name]
        right = target_descriptor["features"][name]
        if left is None or right is None:
            deltas[name] = None
            missing_in = []
            if left is None:
                missing_in.append("reference")
            if right is None:
                missing_in.append("target")
            delta_unavailable[name] = {
                "category": "definition_domain",
                "code": "paired_feature_unavailable",
                "message": "feature is unavailable in " + " and ".join(missing_in),
            }
        else:
            deltas[name] = _finite_float(float(right) - float(left), name=f"delta {name}")

    return {
        "schema": PAIRED_SCHEMA,
        "assumption": "caller asserted direct lattice correspondence; no atom mapping was inferred",
        "lattice_change": {
            "log_volume_strain": _finite_float(
                math.log(determinant), name="log volume strain"
            ),
            "hencky_deviatoric_norm": _finite_float(
                np.linalg.norm(deviator), name="Hencky deviatoric norm"
            ),
            "max_abs_principal_log_strain": _finite_float(
                np.max(np.abs(principal_log_strains)),
                name="max principal log strain",
            ),
            "principal_log_strains_ascending": [
                _finite_float(value, name="principal log strain")
                for value in principal_log_strains
            ],
        },
        "feature_deltas": deltas,
        "delta_unavailable": delta_unavailable,
        "reference": reference_descriptor,
        "target": target_descriptor,
        "scope": (
            "explicit paired structures only; this is not a texture, grain, domain-switching, "
            "phase-fraction, or diffraction-refinement model"
        ),
    }


def _json_dump(data: Any, destination: Path | None) -> None:
    text = json.dumps(data, indent=2, sort_keys=False, allow_nan=False) + "\n"
    if destination is None:
        print(text, end="")
    else:
        destination.write_text(text, encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute intrinsic first-shell geometry and directionality descriptors."
    )
    parser.add_argument("reference", type=Path, help="ordinary POSCAR/CONTCAR path")
    parser.add_argument("--target", type=Path, help="optional explicit paired POSCAR/CONTCAR")
    parser.add_argument(
        "--assume-lattice-correspondence",
        action="store_true",
        help="assert direct reference/target lattice correspondence for paired output",
    )
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        reference = structure_from_poscar(args.reference)
        if args.target is None:
            result = compute_descriptor(reference)
        else:
            target = structure_from_poscar(args.target)
            result = compare_explicit_structures(
                reference,
                target,
                assume_lattice_correspondence=args.assume_lattice_correspondence,
            )
        _json_dump(result, args.output)
        return 0
    except DescriptorUnavailable as exc:
        error = {"status": "unavailable", "reason": exc.as_dict()}
        _json_dump(error, args.output)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
