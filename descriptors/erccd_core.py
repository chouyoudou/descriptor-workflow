"""ERCCD scalar and directional tensor implementation."""

from __future__ import annotations

import math
from typing import Any

from .erccd_elements import ATOMIC_NUMBERS, DescriptorError, covalent_radius_angstrom
from .erccd_geometry import fractional_to_cartesian, inverse, norm, symmetric_eigendecomposition
from .erccd_structure import DescriptorConfig, Structure, parse_poscar


def analyze_structure(
    structure: Structure,
    config: DescriptorConfig | None = None,
) -> dict[str, Any]:
    """Compute the fixed ERCCD scalar and directional tensor extension."""
    cfg = (config or DescriptorConfig()).validate()
    atom_count = structure.atom_count
    species = structure.species
    frac_coords = tuple(tuple(value % 1.0 for value in row) for row in structure.frac_coords)
    lattice = structure.lattice
    lattice_inverse = inverse(lattice)
    dual_norms = [
        math.sqrt(
            lattice_inverse[0][column] ** 2
            + lattice_inverse[1][column] ** 2
            + lattice_inverse[2][column] ** 2
        )
        for column in range(3)
    ]
    numbers = [ATOMIC_NUMBERS[symbol] for symbol in species]
    radii = [covalent_radius_angstrom(symbol) for symbol in species]
    tensor = [[0.0, 0.0, 0.0] for _ in range(3)]
    contrast_sum = 0.0
    z_mismatch_sum = 0.0
    radius_mismatch_sum = 0.0
    contact_count = 0
    near_cutoff_count = 0
    pair_type_counts: dict[str, int] = {}

    for i in range(atom_count):
        for j in range(i + 1, atom_count):
            if species[i] == species[j]:
                continue
            radius_sum = radii[i] + radii[j]
            cutoff = cfg.contact_factor * radius_sum
            delta_frac = tuple(frac_coords[j][axis] - frac_coords[i][axis] for axis in range(3))
            ranges: list[range] = []
            for axis in range(3):
                bound = cutoff * dual_norms[axis] + 1.0e-12
                lower = math.ceil(-delta_frac[axis] - bound)
                upper = math.floor(-delta_frac[axis] + bound)
                ranges.append(range(lower, upper + 1))
            delta_z = abs(numbers[i] - numbers[j]) / (numbers[i] + numbers[j])
            delta_radius = abs(radii[i] - radii[j]) / radius_sum
            contrast = math.sqrt(0.5 * (delta_z * delta_z + delta_radius * delta_radius))
            for tx in ranges[0]:
                for ty in ranges[1]:
                    for tz in ranges[2]:
                        image_frac = (
                            delta_frac[0] + tx,
                            delta_frac[1] + ty,
                            delta_frac[2] + tz,
                        )
                        vector = fractional_to_cartesian(image_frac, lattice)
                        distance = norm(vector)
                        if distance <= 1.0e-12 or distance > cutoff + cfg.cutoff_tolerance_angstrom:
                            continue
                        unit = tuple(component / distance for component in vector)
                        contrast_sum += contrast
                        z_mismatch_sum += delta_z
                        radius_mismatch_sum += delta_radius
                        contact_count += 1
                        if abs(distance / cutoff - 1.0) <= cfg.near_cutoff_fraction:
                            near_cutoff_count += 1
                        pair = "-".join(sorted((species[i], species[j])))
                        pair_type_counts[pair] = pair_type_counts.get(pair, 0) + 1
                        for row in range(3):
                            for column in range(3):
                                tensor[row][column] += contrast * unit[row] * unit[column]

    scalar = contrast_sum / atom_count
    tensor = [[value / atom_count for value in row] for row in tensor]
    eigenvalues, eigenvectors = symmetric_eigendecomposition(tensor)
    trace = sum(tensor[index][index] for index in range(3))
    trace_error = abs(trace - scalar)
    if trace_error > 2.0e-10 * max(1.0, abs(scalar)):
        raise DescriptorError("internal tensor trace identity failed")
    if scalar > 0.0:
        mean = scalar / 3.0
        anisotropy = math.sqrt(1.5 * sum((value - mean) ** 2 for value in eigenvalues)) / scalar
        fractions = [value / scalar for value in eigenvalues]
        entropy = -sum(value * math.log(value) for value in fractions if value > 0.0)
        effective_rank = math.exp(entropy)
        principal_axis: list[float] | None = list(eigenvectors[0])
    else:
        anisotropy = 0.0
        fractions = [0.0, 0.0, 0.0]
        effective_rank = 0.0
        principal_axis = None
    warnings: list[str] = []
    if contact_count == 0:
        warnings.append("no unlike-element contacts under the fixed radius-scaled cutoff")
    if near_cutoff_count:
        warnings.append(
            f"{near_cutoff_count} contacts lie within {cfg.near_cutoff_fraction:.3f} of the cutoff; "
            "small coordinate changes may alter contact membership"
        )

    return {
        "schema": "element-reference-contact-contrast/1",
        "structure": {
            "identifier": structure.identifier,
            "atom_count": atom_count,
            "volume_angstrom3": structure.volume_angstrom3,
            "elements": sorted(set(species), key=lambda symbol: ATOMIC_NUMBERS[symbol]),
        },
        "fixed_scalar": {
            "name": "element_reference_contact_contrast_density",
            "alias": "erccd",
            "value": scalar,
            "units": "dimensionless_per_atom",
            "direction": (
                "larger means more radius-scaled near contacts between elements with larger "
                "neutral-element nuclear-charge/covalent-size contrast"
            ),
        },
        "components": {
            "heterocontact_count": contact_count,
            "heterocontact_count_per_atom": contact_count / atom_count,
            "mean_contrast_per_heterocontact": contrast_sum / contact_count if contact_count else 0.0,
            "nuclear_charge_mismatch_density": z_mismatch_sum / atom_count,
            "covalent_radius_mismatch_density": radius_mismatch_sum / atom_count,
            "near_cutoff_contact_count": near_cutoff_count,
            "pair_type_counts": dict(sorted(pair_type_counts.items())),
        },
        "directional_extension": {
            "contrast_tensor_cartesian": tensor,
            "trace": trace,
            "eigenvalues_descending": eigenvalues,
            "eigenvalue_fractions": fractions,
            "anisotropy_0_to_1": min(1.0, max(0.0, anisotropy)),
            "effective_rank_0_to_3": effective_rank,
            "principal_axis_cartesian": principal_axis,
            "trace_identity_abs_error": trace_error,
        },
        "protocol": {
            "contact_factor": cfg.contact_factor,
            "contact_rule": "d_ij <= contact_factor * (r_cov_i + r_cov_j)",
            "element_reference_channels": ["atomic_number", "neutral_element_covalent_radius_angstrom"],
            "radius_source": "RDKit PeriodicTable GetRcovalent numeric snapshot, queried 2026-10-02",
            "radius_source_license": "BSD-3-Clause",
            "aliases": {"D": "H", "T": "H", "POTCAR_suffix_example": "Fe_pv -> Fe"},
        },
        "scope": (
            "Static periodic near-contact topology plus independent neutral-element references only; "
            "not an oxidation-state, bond-order, interface-energy, optical, thin-film, resonator, "
            "illumination, device, or processing descriptor. A unit cell represents an interface "
            "only when that interface is explicitly present in the coordinates."
        ),
        "warnings": warnings,
    }


def analyze_poscar(text: str, config: DescriptorConfig | None = None) -> dict[str, Any]:
    return analyze_structure(parse_poscar(text), config)


__all__ = ["analyze_poscar", "analyze_structure"]
