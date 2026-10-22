#!/usr/bin/env python3
"""Per-site and structure-level site--void coupling calculations."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from .site_void_atoms import SCHEMA, STATIC_SCOPE, DescriptorError, _canonical_symbol
    from .site_void_math import (
        _dot, _norm, _outer, _symmetric_eigendecomposition,
        _unit, _vscale,
    )
    from .site_void_model import (
        DescriptorConfig, RadiusProjection, SiteDescriptor, Structure,
    )
    from .site_void_neighbors import (
        _clamp, _median, _radius_projection, _ray_sphere_clearance, _radius_table,
        enumerate_neighbor_images, select_coordination_shell,
    )
except ImportError:  # flat private task bundle
    from site_void_atoms import SCHEMA, STATIC_SCOPE, DescriptorError, _canonical_symbol
    from site_void_math import (
        _dot, _norm, _outer, _symmetric_eigendecomposition,
        _unit, _vscale,
    )
    from site_void_model import DescriptorConfig, RadiusProjection, SiteDescriptor, Structure
    from site_void_neighbors import (
        _clamp, _median, _radius_projection, _ray_sphere_clearance, _radius_table,
        enumerate_neighbor_images, select_coordination_shell,
    )

def analyze_site(
    structure: Structure,
    site_index: int,
    *,
    config: DescriptorConfig | None = None,
    radii_overrides: Mapping[str, float] | None = None,
    candidate_elements: Sequence[str] = (),
) -> SiteDescriptor:
    config = config or DescriptorConfig()
    config.validate()
    radii = _radius_table(radii_overrides)
    candidates = enumerate_neighbor_images(
        structure,
        site_index,
        config.search_radius_angstrom,
        radii,
    )
    shell, shell_selection = select_coordination_shell(candidates, config)
    distances = [row.distance for row in shell]
    mean_distance = sum(distances) / len(distances)

    # Equal-weight shell centroid: a deliberately geometric off-centering proxy.
    centroid = (
        sum(row.vector[0] for row in shell) / len(shell),
        sum(row.vector[1] for row in shell) / len(shell),
        sum(row.vector[2] for row in shell) / len(shell),
    )
    offcenter_vector = _vscale(centroid, -1.0)
    offcenter = _norm(offcenter_vector)
    offcenter_fraction = offcenter / max(mean_distance, 1.0e-12)

    # The normalized second moment of shell directions is a blocking tensor.
    tensor = [[0.0] * 3 for _ in range(3)]
    angular_moment = [0.0, 0.0, 0.0]
    for row in shell:
        direction = _unit(row.vector)
        outer = _outer(direction)
        for i in range(3):
            angular_moment[i] -= direction[i] / len(shell)
            for j in range(3):
                tensor[i][j] += outer[i][j] / len(shell)
    eigenvalues, eigenvectors = _symmetric_eigendecomposition(
        tuple(tuple(row) for row in tensor)  # type: ignore[arg-type]
    )
    axis = eigenvectors[0]
    lambda_min, lambda_mid, lambda_max = eigenvalues
    angular_void_anisotropy = _clamp(1.0 - 3.0 * lambda_min, 0.0, 1.0)
    void_axis_uniqueness = _clamp(
        (lambda_mid - lambda_min) / max(lambda_max, 1.0e-12),
        0.0,
        1.0,
    )

    forward = _ray_sphere_clearance(
        axis,
        candidates,
        radii,
        config.clearance_cap_angstrom,
        config.obstacle_radius_scale,
    )
    backward = _ray_sphere_clearance(
        _vscale(axis, -1.0),
        candidates,
        radii,
        config.clearance_cap_angstrom,
        config.obstacle_radius_scale,
    )
    # Orient the axis toward the larger free path.  Equal free paths are resolved
    # by the first angular moment only to make the reported vector deterministic.
    if backward > forward + config.minimum_clearance_contrast:
        axis = _vscale(axis, -1.0)
        forward, backward = backward, forward
    elif abs(forward - backward) <= config.minimum_clearance_contrast:
        moment = (angular_moment[0], angular_moment[1], angular_moment[2])
        if _dot(axis, moment) < 0:
            axis = _vscale(axis, -1.0)
    clearance_contrast = (forward - backward) / max(forward + backward, 1.0e-12)
    clearance_contrast = _clamp(clearance_contrast, 0.0, 1.0)

    if offcenter <= 1.0e-14:
        alignment = 0.0
    else:
        alignment = _clamp(_dot(_unit(offcenter_vector), axis), -1.0, 1.0)
    coupling_support = void_axis_uniqueness * clearance_contrast
    coupling_signed = offcenter_fraction * coupling_support * alignment
    coupling_positive = max(0.0, coupling_signed)

    center_symbol = structure.species[site_index]
    center_radius = radii[center_symbol]
    median_neighbor_radius = _median([radii[row.symbol] for row in shell])
    center_to_neighbor_radius_ratio = center_radius / max(median_neighbor_radius, 1.0e-12)
    locally_large_radius_site = center_to_neighbor_radius_ratio >= 1.0 - 1.0e-12
    available_radii = [row.distance - radii[row.symbol] for row in shell]
    # Median is robust to a single unusually short or long contact while still
    # retaining an explicit minimum-margin diagnostic.
    effective_cavity_radius = max(_median(available_radii), 1.0e-12)
    margins = [value - center_radius for value in available_radii]
    mean_margin = sum(margins) / len(margins)
    minimum_margin = min(margins)
    radius_mismatch = (center_radius - effective_cavity_radius) / effective_cavity_radius
    size_void_coupling = abs(radius_mismatch) * coupling_support

    projections: list[RadiusProjection] = []
    seen_candidates: set[str] = set()
    for raw_candidate in candidate_elements:
        candidate = _canonical_symbol(raw_candidate)
        if candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        projections.append(
            _radius_projection(
                candidate,
                radii[candidate],
                shell,
                radii,
                effective_cavity_radius,
                void_axis_uniqueness,
                clearance_contrast,
            )
        )

    return SiteDescriptor(
        site_index=site_index,
        symbol=center_symbol,
        fractional_coordinate=structure.frac_coords[site_index],
        neighbor_count=len(shell),
        neighbor_indices=tuple(row.atom_index for row in shell),
        neighbor_symbols=tuple(row.symbol for row in shell),
        neighbor_translations=tuple(row.translation for row in shell),
        neighbor_distances_angstrom=tuple(row.distance for row in shell),
        normalized_neighbor_distances=tuple(row.normalized_distance for row in shell),
        mean_neighbor_distance_angstrom=mean_distance,
        shell_cutoff_normalized_distance=max(
            row.normalized_distance for row in shell
        ),
        shell_selection=shell_selection,
        offcenter_vector_angstrom=offcenter_vector,
        offcenter_angstrom=offcenter,
        offcenter_fraction=offcenter_fraction,
        blocking_tensor_eigenvalues=eigenvalues,
        void_axis=axis,
        angular_void_anisotropy=angular_void_anisotropy,
        void_axis_uniqueness=void_axis_uniqueness,
        forward_clearance_angstrom=forward,
        backward_clearance_angstrom=backward,
        clearance_contrast=clearance_contrast,
        offcenter_void_alignment=alignment,
        site_void_coupling_signed=coupling_signed,
        site_void_coupling_positive=coupling_positive,
        coupling_support=coupling_support,
        center_covalent_radius_angstrom=center_radius,
        median_neighbor_covalent_radius_angstrom=median_neighbor_radius,
        center_to_neighbor_radius_ratio=center_to_neighbor_radius_ratio,
        locally_large_radius_site=locally_large_radius_site,
        effective_cavity_radius_angstrom=effective_cavity_radius,
        mean_radial_margin_angstrom=mean_margin,
        minimum_radial_margin_angstrom=minimum_margin,
        radius_mismatch_fraction=radius_mismatch,
        size_void_coupling=size_void_coupling,
        candidate_radius_projections=tuple(projections),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate_sites(sites: Sequence[SiteDescriptor]) -> dict[str, float | int]:
    if not sites:
        raise DescriptorError("no sites were selected")
    signed = [row.site_void_coupling_signed for row in sites]
    positive = [row.site_void_coupling_positive for row in sites]
    locally_large = [row for row in sites if row.locally_large_radius_site]
    if not locally_large:  # mathematically unlikely, retained as a hard diagnostic
        raise DescriptorError("no locally large-radius site found for role aggregate")
    return {
        "site_count": len(sites),
        "offcenter_fraction_mean": _mean([row.offcenter_fraction for row in sites]),
        "offcenter_fraction_max": max(row.offcenter_fraction for row in sites),
        "angular_void_anisotropy_mean": _mean(
            [row.angular_void_anisotropy for row in sites]
        ),
        "void_axis_uniqueness_mean": _mean(
            [row.void_axis_uniqueness for row in sites]
        ),
        "clearance_contrast_mean": _mean([row.clearance_contrast for row in sites]),
        "site_void_coupling_signed_mean": _mean(signed),
        "site_void_coupling_abs_mean": _mean([abs(value) for value in signed]),
        "site_void_coupling_positive_mean": _mean(positive),
        "site_void_coupling_positive_max": max(positive),
        "offcenter_aligned_fraction": _mean(
            [1.0 if row.offcenter_void_alignment > 0 else 0.0 for row in sites]
        ),
        "radius_mismatch_abs_mean": _mean(
            [abs(row.radius_mismatch_fraction) for row in sites]
        ),
        "size_void_coupling_mean": _mean([row.size_void_coupling for row in sites]),
        "size_void_coupling_max": max(row.size_void_coupling for row in sites),
        "locally_large_radius_site_fraction": len(locally_large) / len(sites),
        "locally_large_radius_ratio_mean": _mean(
            [row.center_to_neighbor_radius_ratio for row in locally_large]
        ),
        "locally_large_site_void_coupling_positive_mean": _mean(
            [row.site_void_coupling_positive for row in locally_large]
        ),
        "locally_large_site_void_coupling_positive_max": max(
            row.site_void_coupling_positive for row in locally_large
        ),
        "locally_large_size_void_coupling_mean": _mean(
            [row.size_void_coupling for row in locally_large]
        ),
        "locally_large_size_void_coupling_max": max(
            row.size_void_coupling for row in locally_large
        ),
    }


def _select_site_indices(
    structure: Structure,
    center_elements: Sequence[str] = (),
    center_indices: Sequence[int] = (),
) -> list[int]:
    explicit_indices = list(dict.fromkeys(int(index) for index in center_indices))
    for index in explicit_indices:
        if not 0 <= index < structure.atom_count:
            raise DescriptorError(f"center index {index} is out of range")
    element_set = {_canonical_symbol(symbol) for symbol in center_elements}
    element_indices = [
        index for index, symbol in enumerate(structure.species) if symbol in element_set
    ]
    if explicit_indices and element_set:
        selected = [index for index in explicit_indices if index in set(element_indices)]
    elif explicit_indices:
        selected = explicit_indices
    elif element_set:
        selected = element_indices
    else:
        selected = list(range(structure.atom_count))
    if not selected:
        raise DescriptorError("site filters selected no atoms")
    return selected


def analyze_structure(
    structure: Structure,
    *,
    center_elements: Sequence[str] = (),
    center_indices: Sequence[int] = (),
    candidate_elements: Sequence[str] = (),
    config: DescriptorConfig | None = None,
    radii_overrides: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    config = config or DescriptorConfig()
    config.validate()
    indices = _select_site_indices(structure, center_elements, center_indices)
    sites = [
        analyze_site(
            structure,
            index,
            config=config,
            radii_overrides=radii_overrides,
            candidate_elements=candidate_elements,
        )
        for index in indices
    ]
    return {
        "schema": SCHEMA,
        "interpretation_scope": STATIC_SCOPE,
        "protocol": {
            **dataclasses.asdict(config),
            "radius_lookup": "embedded RDKit GetRcovalent-compatible table; explicit overrides allowed",
            "offcenter_reference": "equal-weight centroid of adaptive local shell",
            "void_axis": "least-eigenvalue axis of normalized shell direction second moment",
            "axis_sign": "larger capped ray-sphere free path",
            "structure_aggregation": "equal weight per selected explicit POSCAR site",
            "local_size_role": (
                "locally large-radius sites have center covalent radius at least the "
                "median covalent radius of their selected shell; this is not an oxidation-state label"
            ),
        },
        "structure": {
            "comment": structure.comment,
            "atom_count": structure.atom_count,
            "volume_angstrom3": structure.volume_angstrom3,
            "selected_site_indices": indices,
        },
        "scalars": _aggregate_sites(sites),
        "sites": [row.to_dict() for row in sites],
    }
