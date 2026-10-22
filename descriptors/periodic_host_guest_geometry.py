#!/usr/bin/env python3
"""Periodic host--guest geometry (PHGG) descriptor.

PHGG separates radius-gated periodic coordination components from finite
components, estimates host-only open space on a deterministic fractional grid,
and summarizes explicit host--guest nearest-surface geometry.  It is a static
structure descriptor.  It does not infer charge, oxidation state, bond energy,
adsorption, transport, detonation, sensitivity, synthesis, or stability.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .phgg_graph import (
    Component,
    build_periodic_bond_graph,
    build_surface_radii,
    contact_directionality_anisotropy,
    find_components,
    host_grid_clearance,
    minimum_image_vector,
    partition_host_guest,
    unwrapped_component_cartesian,
)
from .phgg_radii import RADIUS_PROVENANCE
from .phgg_types import (
    DescriptorError,
    FeatureResult,
    Settings,
    StructureData,
    Vector3,
    norm,
    parse_poscar_text,
    row_times_matrix,
    wrap_fractional,
)

FEATURE_ID = "periodic-host-guest-geometry/1"
DESCRIPTOR_VERSION = "1.0.0"

FEATURE_NAMES: tuple[str, ...] = (
    "component_count_per_cell",
    "host_network_dimensionality",
    "host_selection_periodic_ratio",
    "host_atom_fraction_ratio",
    "finite_guest_component_count_per_cell",
    "finite_guest_atom_fraction_ratio",
    "host_guest_present_ratio",
    "host_largest_grid_included_diameter_A",
    "host_grid_void_fraction_ratio",
    "host_grid_mean_positive_clearance_A",
    "host_guest_min_distance_A",
    "host_guest_min_surface_gap_A",
    "host_guest_mean_nearest_surface_gap_A",
    "host_guest_contact_atom_fraction_ratio",
    "guest_centered_cavity_diameter_A",
    "guest_surface_envelope_diameter_A",
    "guest_cavity_fit_ratio",
    "guest_contact_directionality_anisotropy_ratio",
)

FEATURE_UNITS: dict[str, str] = {
    "component_count_per_cell": "count/cell",
    "host_network_dimensionality": "rank (0-3)",
    "host_selection_periodic_ratio": "ratio",
    "host_atom_fraction_ratio": "ratio",
    "finite_guest_component_count_per_cell": "count/cell",
    "finite_guest_atom_fraction_ratio": "ratio",
    "host_guest_present_ratio": "ratio",
    "host_largest_grid_included_diameter_A": "angstrom",
    "host_grid_void_fraction_ratio": "ratio",
    "host_grid_mean_positive_clearance_A": "angstrom",
    "host_guest_min_distance_A": "angstrom",
    "host_guest_min_surface_gap_A": "angstrom",
    "host_guest_mean_nearest_surface_gap_A": "angstrom",
    "host_guest_contact_atom_fraction_ratio": "ratio",
    "guest_centered_cavity_diameter_A": "angstrom",
    "guest_surface_envelope_diameter_A": "angstrom",
    "guest_cavity_fit_ratio": "ratio",
    "guest_contact_directionality_anisotropy_ratio": "ratio",
}

STATIC_SCOPE = {
    "allowed": [
        "periodic coordination-network connectivity",
        "host-only geometric open space",
        "explicit finite-component host-guest contact geometry",
    ],
    "not_claimed": [
        "Hirshfeld surface fractions",
        "bond order or coordination energy",
        "charge or oxidation state",
        "adsorption, diffusion, transport, energetic, detonation, or sensitivity performance",
        "synthesis, stability, or optimization",
    ],
}


def _component_metadata(components: Sequence[Component]) -> list[dict[str, Any]]:
    return [
        {
            "atom_indices": list(component.atom_indices),
            "atom_count": len(component.atom_indices),
            "translation_rank": component.translation_rank,
            "cycle_translations": [list(vector) for vector in component.cycle_translations],
        }
        for component in components
    ]


def _guest_component_geometry(
    structure: StructureData,
    guest: Component,
    host_atoms: Sequence[int],
    surface_radii: Sequence[float],
) -> tuple[float, float, float, bool]:
    coords = unwrapped_component_cartesian(structure, guest)
    center: Vector3 = (
        sum(coord[0] for coord in coords) / len(coords),
        sum(coord[1] for coord in coords) / len(coords),
        sum(coord[2] for coord in coords) / len(coords),
    )
    center_frac = wrap_fractional(row_times_matrix(center, structure.inverse_lattice))
    envelope_radius = 0.0
    for atom, coord in zip(guest.atom_indices, coords):
        displacement = (coord[0] - center[0], coord[1] - center[1], coord[2] - center[2])
        envelope_radius = max(envelope_radius, norm(displacement) + surface_radii[atom])
    cavity_radius = math.inf
    for host in host_atoms:
        _, distance, _ = minimum_image_vector(
            center_frac,
            structure.fractional[host],
            structure.lattice,
            structure.inverse_lattice,
        )
        cavity_radius = min(cavity_radius, distance - surface_radii[host])
    cavity_radius = max(0.0, cavity_radius)
    valid_fit = cavity_radius > 1.0e-12
    fit = envelope_radius / cavity_radius if valid_fit else 0.0
    return 2.0 * cavity_radius, 2.0 * envelope_radius, fit, valid_fit


def _nearest_contact_record(
    structure: StructureData,
    guest_atom: int,
    host_atoms: Sequence[int],
    surface_radii: Sequence[float],
    contact_scale: float,
) -> tuple[float, float, bool, list[Vector3], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for host_atom in host_atoms:
        vector, distance, image = minimum_image_vector(
            structure.fractional[guest_atom],
            structure.fractional[host_atom],
            structure.lattice,
            structure.inverse_lattice,
        )
        radius_sum = surface_radii[guest_atom] + surface_radii[host_atom]
        candidates.append(
            {
                "host_atom_index": host_atom,
                "image": image,
                "vector": vector,
                "distance_A": distance,
                "surface_gap_A": distance - radius_sum,
                "surface_distance_ratio": distance / radius_sum,
            }
        )
    center_nearest = min(candidates, key=lambda item: (item["distance_A"], item["host_atom_index"], item["image"]))
    surface_nearest = min(
        candidates,
        key=lambda item: (item["surface_gap_A"], item["surface_distance_ratio"], item["host_atom_index"], item["image"]),
    )
    minimum_ratio = min(item["surface_distance_ratio"] for item in candidates)
    ratio_tolerance = max(1.0e-12, 1.0e-12 * abs(minimum_ratio))
    ratio_ties = [
        item for item in candidates if abs(item["surface_distance_ratio"] - minimum_ratio) <= ratio_tolerance
    ]
    ratio_ties.sort(key=lambda item: (item["host_atom_index"], item["image"]))
    is_contact = minimum_ratio <= contact_scale
    directions = [item["vector"] for item in ratio_ties] if is_contact else []
    record = {
        "guest_atom_index": guest_atom,
        "center_nearest_host_atom_index": center_nearest["host_atom_index"],
        "center_nearest_image": list(center_nearest["image"]),
        "center_nearest_distance_A": center_nearest["distance_A"],
        "surface_nearest_host_atom_index": surface_nearest["host_atom_index"],
        "surface_nearest_image": list(surface_nearest["image"]),
        "surface_nearest_gap_A": surface_nearest["surface_gap_A"],
        "minimum_surface_distance_ratio": minimum_ratio,
        "contact_tied_host_atom_indices": [item["host_atom_index"] for item in ratio_ties],
        "within_contact_scale": is_contact,
    }
    return center_nearest["distance_A"], surface_nearest["surface_gap_A"], is_contact, directions, record


def analyze_structure(structure: StructureData, settings: Settings) -> FeatureResult:
    edges, covalent_fallback = build_periodic_bond_graph(structure, settings)
    components = find_components(structure, edges)
    host_atoms, guests, host_selection_mode = partition_host_guest(components)
    if not host_atoms:
        raise DescriptorError("host partition is empty")
    surface_radii, surface_fallback = build_surface_radii(structure)

    max_grid_diameter, grid_void_fraction, grid_mean_clearance, grid_dims, grid_anchor = host_grid_clearance(
        structure, host_atoms, settings, surface_radii
    )

    guest_atoms = tuple(atom for component in guests for atom in component.atom_indices)
    nearest_distances: list[float] = []
    nearest_gaps: list[float] = []
    contact_direction_groups: list[list[Vector3]] = []
    contact_count = 0
    contact_records: list[dict[str, Any]] = []
    for guest_atom in guest_atoms:
        distance, gap, is_contact, directions, record = _nearest_contact_record(
            structure, guest_atom, host_atoms, surface_radii, settings.contact_scale
        )
        nearest_distances.append(distance)
        nearest_gaps.append(gap)
        contact_count += int(is_contact)
        if directions:
            contact_direction_groups.append(directions)
        contact_records.append(record)

    cavity_diameters: list[float] = []
    envelope_diameters: list[float] = []
    fit_ratios: list[float] = []
    fit_weights: list[int] = []
    invalid_fit_components = 0
    for guest in guests:
        cavity, envelope, fit, valid_fit = _guest_component_geometry(
            structure, guest, host_atoms, surface_radii
        )
        cavity_diameters.append(cavity)
        envelope_diameters.append(envelope)
        fit_ratios.append(fit)
        fit_weights.append(len(guest.atom_indices))
        invalid_fit_components += int(not valid_fit)

    guest_present = bool(guest_atoms)
    total_guest_weight = sum(fit_weights)
    weighted_cavity = (
        sum(value * weight for value, weight in zip(cavity_diameters, fit_weights)) / total_guest_weight
        if total_guest_weight
        else 0.0
    )
    weighted_envelope = (
        sum(value * weight for value, weight in zip(envelope_diameters, fit_weights)) / total_guest_weight
        if total_guest_weight
        else 0.0
    )
    weighted_fit = (
        sum(value * weight for value, weight in zip(fit_ratios, fit_weights)) / total_guest_weight
        if total_guest_weight
        else 0.0
    )

    periodic_components = [component for component in components if component.translation_rank > 0]
    host_dimensionality = max((component.translation_rank for component in periodic_components), default=0)
    features = {
        "component_count_per_cell": float(len(components)),
        "host_network_dimensionality": float(host_dimensionality),
        "host_selection_periodic_ratio": 1.0 if periodic_components else 0.0,
        "host_atom_fraction_ratio": len(host_atoms) / structure.n_atoms,
        "finite_guest_component_count_per_cell": float(len(guests)),
        "finite_guest_atom_fraction_ratio": len(guest_atoms) / structure.n_atoms,
        "host_guest_present_ratio": 1.0 if guest_present else 0.0,
        "host_largest_grid_included_diameter_A": max_grid_diameter,
        "host_grid_void_fraction_ratio": grid_void_fraction,
        "host_grid_mean_positive_clearance_A": grid_mean_clearance,
        "host_guest_min_distance_A": min(nearest_distances) if guest_present else 0.0,
        "host_guest_min_surface_gap_A": min(nearest_gaps) if guest_present else 0.0,
        "host_guest_mean_nearest_surface_gap_A": (
            sum(nearest_gaps) / len(nearest_gaps) if guest_present else 0.0
        ),
        "host_guest_contact_atom_fraction_ratio": contact_count / len(guest_atoms) if guest_present else 0.0,
        "guest_centered_cavity_diameter_A": weighted_cavity,
        "guest_surface_envelope_diameter_A": weighted_envelope,
        "guest_cavity_fit_ratio": weighted_fit,
        "guest_contact_directionality_anisotropy_ratio": contact_directionality_anisotropy(
            contact_direction_groups
        ),
    }
    if tuple(features) != FEATURE_NAMES:
        raise AssertionError("internal fixed-schema order changed")
    for name, value in features.items():
        if not isinstance(value, float) or not math.isfinite(value):
            raise DescriptorError(f"non-finite feature {name}: {value!r}")

    metadata: dict[str, Any] = {
        "feature_id": FEATURE_ID,
        "descriptor_version": DESCRIPTOR_VERSION,
        "fixed_feature_names": list(FEATURE_NAMES),
        "units": FEATURE_UNITS,
        "n_atoms": structure.n_atoms,
        "cell_volume_A3": structure.volume_A3,
        "edge_count": len(edges),
        "components": _component_metadata(components),
        "host_atom_indices": list(host_atoms),
        "guest_component_atom_indices": [list(component.atom_indices) for component in guests],
        "host_selection_mode": host_selection_mode,
        "host_guest_valid": guest_present,
        "guest_fit_valid": guest_present and invalid_fit_components == 0,
        "guest_fit_invalid_component_count": invalid_fit_components,
        "nearest_host_contact_records": contact_records,
        "grid_dimensions": list(grid_dims),
        "grid_point_count": grid_dims[0] * grid_dims[1] * grid_dims[2],
        "grid_anchor_fractional": list(grid_anchor),
        "grid_axis_step_A": [norm(structure.lattice[axis]) / grid_dims[axis] for axis in range(3)],
        "parameters": {
            "bond_scale": settings.bond_scale,
            "bond_offset_A": settings.bond_offset_A,
            "contact_scale": settings.contact_scale,
            "grid_spacing_A": settings.grid_spacing_A,
            "max_grid_points": settings.max_grid_points,
            "probe_radius_A": settings.probe_radius_A,
            "max_atoms": settings.max_atoms,
            "max_edges": settings.max_edges,
        },
        "element_lookup": {
            **RADIUS_PROVENANCE,
            "covalent_fallback_elements": list(covalent_fallback),
            "surface_fallback_elements": list(surface_fallback),
        },
        "method_semantics": {
            "connectivity": (
                "radius-gated periodic multigraph; component dimension is the integer rank of "
                "independent quotient-graph cycle translations"
            ),
            "host_guest_partition": (
                "all periodic components are host; if none are periodic, all largest finite "
                "components are fallback host and only smaller components are guests"
            ),
            "pore_size": (
                "deterministic host-only fractional-grid clearance approximation; not an exact "
                "Voronoi largest-included-sphere or pore-limiting diameter"
            ),
            "contact": (
                "explicit nearest host/guest center and tabulated-surface geometry; contact fraction "
                "uses radius-normalized distance and is not a Hirshfeld-surface contact fraction"
            ),
        },
        "missing_value_policy": (
            "guest-only scalar channels are numeric zero when no finite guest is present; "
            "host_guest_present_ratio and metadata validity fields carry applicability"
        ),
        "scope": STATIC_SCOPE,
    }
    return FeatureResult(features=features, metadata=metadata)


def compute(poscar_data: str, params: Mapping[str, Any] | None = None) -> FeatureResult:
    """Compute the fixed PHGG scalar vector from a POSCAR string."""
    settings = Settings.from_params(params)
    structure = parse_poscar_text(poscar_data, max_atoms=settings.max_atoms)
    return analyze_structure(structure, settings)


def _parse_params(raw: str | None) -> Mapping[str, Any]:
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DescriptorError(f"invalid --params JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise DescriptorError("--params must decode to a JSON object")
    return value


def _main() -> int:
    parser = argparse.ArgumentParser(description="Compute periodic host-guest geometry scalars")
    parser.add_argument("--poscar", required=True, help="path to a VASP 5/6 POSCAR")
    parser.add_argument("--params", help="JSON object with explicit descriptor parameters")
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    args = parser.parse_args()
    try:
        result = compute(Path(args.poscar).read_text(encoding="utf-8"), _parse_params(args.params))
    except (OSError, DescriptorError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            result.to_dict(),
            sort_keys=True,
            allow_nan=False,
            indent=2 if args.pretty else None,
            separators=None if args.pretty else (",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
