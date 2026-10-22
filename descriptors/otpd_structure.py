"""Site and structure aggregation for the OTPD descriptor."""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:
    from .otpd_types import (
        ALL_SCALARS, ELEMENT_PROTOCOL, SCHEMA, STATIC_SCOPE, DescriptorError,
        Neighbor, Settings, SiteResult, StructureData, canonical_symbol, radius_table,
    )
    from .otpd_geometry import _translation_grid, _unit, periodic_neighbors, select_six_shell
    from .otpd_vectors import analyze_six_vectors
except ImportError:
    from otpd_types import (
        ALL_SCALARS, ELEMENT_PROTOCOL, SCHEMA, STATIC_SCOPE, DescriptorError,
        Neighbor, Settings, SiteResult, StructureData, canonical_symbol, radius_table,
    )
    from otpd_geometry import _translation_grid, _unit, periodic_neighbors, select_six_shell
    from otpd_vectors import analyze_six_vectors

def _absolute_fractional(structure: StructureData, neighbor: Neighbor) -> np.ndarray:
    return structure.fractional[neighbor.atom_index] + np.asarray(neighbor.image, dtype=float)


def _local_donor_plane_normal(
    structure: StructureData,
    center_index: int,
    donor: Neighbor,
    radii: Mapping[str, float],
    settings: Settings,
) -> np.ndarray | None:
    donor_abs = _absolute_fractional(structure, donor)
    donor_radius = radii[donor.symbol]
    cutoff = min(settings.max_search_radius, settings.backbone_ratio_max * (donor_radius + max(radii.values())) + 0.25)
    shifts = _translation_grid(structure.lattice, cutoff)
    candidates: list[tuple[float, float, tuple[int, int, int, int], np.ndarray]] = []
    center_abs = structure.fractional[center_index]
    for atom_index, (symbol, frac) in enumerate(zip(structure.species, structure.fractional, strict=True)):
        radius = radii.get(symbol)
        if radius is None:
            continue
        absolute = frac[None, :] + shifts
        delta = absolute - donor_abs[None, :]
        cart = delta @ structure.lattice
        distances = np.linalg.norm(cart, axis=1)
        ratios = distances / (donor_radius + radius)
        mask = (distances > 1.0e-10) & (ratios <= settings.backbone_ratio_max + 1.0e-12)
        for local_index in np.nonzero(mask)[0]:
            candidate_abs = absolute[local_index]
            # Exclude the coordinating center image used for this environment.
            if atom_index == center_index and np.linalg.norm((candidate_abs - center_abs) @ structure.lattice) <= 1.0e-8:
                continue
            image = tuple(int(x) for x in shifts[local_index])
            candidates.append(
                (
                    float(ratios[local_index]),
                    float(distances[local_index]),
                    (atom_index, *image),
                    np.asarray(cart[local_index], dtype=float),
                )
            )
    candidates.sort(key=lambda row: (round(row[0], 12), round(row[1], 12), row[2]))
    pool = candidates[:6]
    best: tuple[tuple[float, tuple[int, int, int, int], tuple[int, int, int, int]], np.ndarray] | None = None
    for a, b in itertools.combinations(pool, 2):
        cross = np.cross(a[3], b[3])
        denom = float(np.linalg.norm(a[3]) * np.linalg.norm(b[3]))
        if denom <= settings.numerical_tolerance:
            continue
        quality = float(np.linalg.norm(cross) / denom)
        if quality < settings.plane_cross_min:
            continue
        normal = _unit(cross)
        objective = (a[0] + b[0], a[2], b[2])
        if best is None or objective < best[0]:
            best = (objective, normal)
    return None if best is None else best[1]


def analyze_site(
    structure: StructureData,
    center_index: int,
    *,
    radii: Mapping[str, float] | None = None,
    settings: Settings | None = None,
    include_plane_extension: bool = True,
) -> tuple[SiteResult | None, dict[str, Any]]:
    settings = settings or Settings()
    settings.validate()
    radius_map = radius_table(radii)
    if not (0 <= center_index < structure.atom_count):
        raise DescriptorError("center index out of range")
    try:
        neighbors = periodic_neighbors(structure, center_index, radius_map, settings)
    except DescriptorError as exc:
        return None, {"reason": "radius_or_neighbor_error", "detail": str(exc)}
    shell, shell_diag = select_six_shell(neighbors, settings)
    if shell is None:
        return None, shell_diag
    plane_normals: list[np.ndarray | None] | None = None
    if include_plane_extension:
        plane_normals = [
            _local_donor_plane_normal(structure, center_index, donor, radius_map, settings) for donor in shell
        ]
    result = analyze_six_vectors(
        structure.species[center_index],
        [n.symbol for n in shell],
        [n.vector for n in shell],
        radii=radius_map,
        settings=settings,
        center_index=center_index,
        neighbor_keys=[n.key for n in shell],
        donor_plane_normals=plane_normals,
    )
    if result is None:
        shell_diag = dict(shell_diag)
        shell_diag["reason"] = "six_shell_not_octahedral_like"
        return None, shell_diag
    result.diagnostics["shell"] = shell_diag
    return result, shell_diag


def _aggregate(values: Sequence[float | None], *, absolute: bool = False) -> float | None:
    finite = [abs(float(v)) if absolute else float(v) for v in values if v is not None and math.isfinite(float(v))]
    return None if not finite else float(sum(finite) / len(finite))


def analyze_structure(
    structure: StructureData,
    *,
    center_elements: set[str] | None = None,
    radii: Mapping[str, float] | None = None,
    settings: Settings | None = None,
    include_sites: bool = False,
    include_plane_extension: bool = True,
) -> dict[str, Any]:
    settings = settings or Settings()
    settings.validate()
    radius_map = radius_table(radii)
    selected_elements = None if center_elements is None else {canonical_symbol(x) for x in center_elements}
    sites: list[SiteResult] = []
    rejection_counts: dict[str, int] = {}
    attempted = 0
    for center_index, symbol in enumerate(structure.species):
        if selected_elements is not None and symbol not in selected_elements:
            continue
        attempted += 1
        site, diag = analyze_site(
            structure,
            center_index,
            radii=radius_map,
            settings=settings,
            include_plane_extension=include_plane_extension,
        )
        if site is None:
            reason = str(diag.get("reason", "unknown"))
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        else:
            sites.append(site)

    scalars = {name: None for name in ALL_SCALARS}
    if sites:
        scalars.update(
            {
                "otpd_pair_range_mean": _aggregate([s.values["pair_range"] for s in sites]),
                "otpd_pair_range_max": max(float(s.values["pair_range"]) for s in sites),
                "otpd_uniaxial_polarity_mean": _aggregate([s.values["uniaxial_polarity"] for s in sites]),
                "otpd_uniaxial_polarity_abs_mean": _aggregate(
                    [s.values["uniaxial_polarity"] for s in sites], absolute=True
                ),
                "otpd_short_pair_log_compression_mean": _aggregate(
                    [s.values["short_pair_log_compression"] for s in sites]
                ),
                "otpd_long_pair_log_elongation_mean": _aggregate(
                    [s.values["long_pair_log_elongation"] for s in sites]
                ),
                "otpd_pair_asymmetry_rms_mean": _aggregate([s.values["pair_asymmetry_rms"] for s in sites]),
                "otpd_trans_bend_rms_mean_deg": _aggregate([s.values["trans_bend_rms_deg"] for s in sites]),
                "otpd_cis_bend_rms_mean_deg": _aggregate([s.values["cis_bend_rms_deg"] for s in sites]),
                "otpd_baur_distortion_mean": _aggregate([s.values["baur_distortion"] for s in sites]),
                "otpd_radius_pair_range_mean": _aggregate([s.values["radius_pair_range"] for s in sites]),
                "otpd_radius_uniaxial_polarity_mean": _aggregate(
                    [s.values["radius_uniaxial_polarity"] for s in sites]
                ),
                "otpd_strain_bend_correlation_mean": _aggregate(
                    [s.values["strain_bend_correlation"] for s in sites]
                ),
                "otpd_donor_size_bond_correlation_mean": _aggregate(
                    [s.values["donor_size_bond_correlation"] for s in sites]
                ),
                "otpd_donor_plane_twist_mean_deg": _aggregate(
                    [s.values["donor_plane_twist_mean_deg"] for s in sites]
                ),
            }
        )

    extension_support = {
        "strain_bend_correlation_sites": sum(s.values["strain_bend_correlation"] is not None for s in sites),
        "donor_size_bond_correlation_sites": sum(
            s.values["donor_size_bond_correlation"] is not None for s in sites
        ),
        "donor_plane_twist_sites": sum(s.values["donor_plane_twist_mean_deg"] is not None for s in sites),
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": STATIC_SCOPE,
        "id": structure.identifier,
        "status": "ok" if sites else "domain_no_octahedral_like_site",
        "atom_count": structure.atom_count,
        "attempted_center_count": attempted,
        "qualified_site_count": len(sites),
        "qualified_site_fraction": (len(sites) / attempted) if attempted else None,
        "center_elements": None if selected_elements is None else sorted(selected_elements),
        "scalars": scalars,
        "extension_support": extension_support,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "settings": dataclasses.asdict(settings),
        "element_protocol": ELEMENT_PROTOCOL,
    }
    if include_sites:
        result["sites"] = [site.to_dict() for site in sites]
    return result


