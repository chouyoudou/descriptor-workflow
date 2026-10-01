"""Radius-contact and explicit bond-valence site/structure analysis."""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
from collections.abc import Sequence
from typing import Any

try:
    from .bv_directionality_model import (
        COVALENT_RADII_ANGSTROM, EXPLICIT_PROTOCOL_SCHEMA, INTERPRETATION_SCOPE,
        SCHEMA, ContactConfig, DescriptorError, ExplicitBondValenceProtocol,
        PairParameter, Structure, Vec3, _cart_to_frac, _frac_to_cart, _mean,
        _norm, _percentile, _vadd, _vscale, _wrap_frac,
    )
    from .bv_directionality_io import _neighbor_images, _point_clearance, read_poscar
except ImportError:  # direct script sibling import
    from bv_directionality_model import (
        COVALENT_RADII_ANGSTROM, EXPLICIT_PROTOCOL_SCHEMA, INTERPRETATION_SCOPE,
        SCHEMA, ContactConfig, DescriptorError, ExplicitBondValenceProtocol,
        PairParameter, Structure, Vec3, _cart_to_frac, _frac_to_cart, _mean,
        _norm, _percentile, _vadd, _vscale, _wrap_frac,
    )
    from bv_directionality_io import _neighbor_images, _point_clearance, read_poscar


def _contact_weight(normalized_distance: float, softness: float) -> float:
    return math.exp(-max(normalized_distance - 1.0, 0.0) / softness)


def analyze_radius_contact_site(
    structure: Structure,
    site_index: int,
    config: ContactConfig | None = None,
) -> dict[str, Any]:
    config = config or ContactConfig()
    config.validate()
    if site_index < 0 or site_index >= len(structure.species):
        raise DescriptorError("site index out of range")

    center_symbol = structure.species[site_index]
    center_radius = COVALENT_RADII_ANGSTROM[center_symbol]
    neighbors = _neighbor_images(structure, site_index, config.fallback_search_radius_angstrom)
    selected = [item for item in neighbors if item.normalized_distance <= config.contact_cutoff_ratio]
    fallback_used = len(selected) < config.minimum_neighbors
    if fallback_used:
        selected = neighbors[: config.minimum_neighbors]
    selected = selected[: config.maximum_neighbors]

    if not selected:
        return {
            "site_index": site_index,
            "element": center_symbol,
            "covalent_radius_angstrom": center_radius,
            "neighbor_count": 0,
            "fallback_used": True,
            "contact_weight_sum": 0.0,
            "effective_coordination": 0.0,
            "contact_vector_imbalance": 0.0,
            "contact_vector_magnitude": 0.0,
            "open_axis_cartesian": None,
            "mean_normalized_contact_distance": 0.0,
            "radial_dispersion": 0.0,
            "cavity_radius_angstrom": 0.0,
            "radius_fit_mismatch": 1.0,
            "virtual_point_fractional": None,
            "virtual_point_clearance_angstrom": 0.0,
            "virtual_point_nearest_site_index": None,
            "opposed_void_support": 0.0,
            "neighbors": [],
        }

    weights = [
        _contact_weight(item.normalized_distance, config.contact_softness_ratio)
        for item in selected
    ]
    weight_sum = sum(weights)
    weighted_vector: Vec3 = (0.0, 0.0, 0.0)
    for item, weight in zip(selected, weights, strict=True):
        unit = _vscale(item.vector_cart, 1.0 / item.distance_angstrom)
        weighted_vector = _vadd(weighted_vector, _vscale(unit, weight))
    vector_magnitude = _norm(weighted_vector)
    imbalance = vector_magnitude / weight_sum if weight_sum > 0 else 0.0
    effective_coordination = (
        weight_sum * weight_sum / sum(weight * weight for weight in weights)
        if weight_sum > 0
        else 0.0
    )
    mean_q = sum(
        weight * item.normalized_distance
        for item, weight in zip(selected, weights, strict=True)
    ) / weight_sum
    radial_dispersion = math.sqrt(
        sum(
            weight * (item.normalized_distance - mean_q) ** 2
            for item, weight in zip(selected, weights, strict=True)
        )
        / weight_sum
    )
    cavity_radius = sum(
        weight * max(item.distance_angstrom - item.radius_angstrom, 0.0)
        for item, weight in zip(selected, weights, strict=True)
    ) / weight_sum
    radius_fit_mismatch = abs(center_radius - cavity_radius) / max(
        center_radius + cavity_radius, 1e-12
    )

    center_cart = _frac_to_cart(structure.frac_coords[site_index], structure.lattice)
    open_axis: Vec3 | None = None
    virtual_frac: Vec3 | None = None
    clearance = 0.0
    nearest_index: int | None = None
    opposed_support = 0.0
    if vector_magnitude > config.vector_epsilon:
        open_axis = _vscale(weighted_vector, -1.0 / vector_magnitude)
        virtual_cart = _vadd(
            center_cart, _vscale(open_axis, config.virtual_offset_angstrom)
        )
        virtual_frac = _wrap_frac(_cart_to_frac(virtual_cart, structure.lattice))
        clearance, nearest_index = _point_clearance(
            structure,
            virtual_cart,
            excluded_site_index=site_index,
            search_radius_angstrom=config.fallback_search_radius_angstrom,
        )
        clearance_fraction = min(
            max(clearance, 0.0) / config.virtual_clearance_cap_angstrom, 1.0
        )
        opposed_support = imbalance * clearance_fraction

    neighbor_rows = []
    for item, weight in zip(selected, weights, strict=True):
        neighbor_rows.append(
            {
                "site_index": item.site_index,
                "element": structure.species[item.site_index],
                "translation": list(item.translation),
                "distance_angstrom": item.distance_angstrom,
                "normalized_distance": item.normalized_distance,
                "contact_weight": weight,
            }
        )

    return {
        "site_index": site_index,
        "element": center_symbol,
        "covalent_radius_angstrom": center_radius,
        "neighbor_count": len(selected),
        "fallback_used": fallback_used,
        "contact_weight_sum": weight_sum,
        "effective_coordination": effective_coordination,
        "contact_vector_imbalance": imbalance,
        "contact_vector_magnitude": vector_magnitude,
        "open_axis_cartesian": list(open_axis) if open_axis is not None else None,
        "mean_normalized_contact_distance": mean_q,
        "radial_dispersion": radial_dispersion,
        "cavity_radius_angstrom": cavity_radius,
        "radius_fit_mismatch": radius_fit_mismatch,
        "virtual_point_fractional": list(virtual_frac) if virtual_frac is not None else None,
        "virtual_point_clearance_angstrom": clearance,
        "virtual_point_nearest_site_index": nearest_index,
        "opposed_void_support": opposed_support,
        "neighbors": neighbor_rows,
    }


def load_explicit_protocol(
    source: str | pathlib.Path | dict[str, Any],
    structure: Structure,
) -> ExplicitBondValenceProtocol:
    if isinstance(source, dict):
        data = source
    else:
        data = json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
    if data.get("schema") != EXPLICIT_PROTOCOL_SCHEMA:
        raise DescriptorError(
            f"explicit protocol schema must be {EXPLICIT_PROTOCOL_SCHEMA!r}"
        )
    parameters = tuple(
        PairParameter(
            cation=str(row["cation"]),
            cation_oxidation=int(row["cation_oxidation"]),
            anion=str(row["anion"]),
            anion_oxidation=int(row["anion_oxidation"]),
            r0_angstrom=float(row["r0_angstrom"]),
            b_angstrom=float(row["b_angstrom"]),
            source=str(row["source"]),
        )
        for row in data.get("pair_parameters", [])
    )
    protocol = ExplicitBondValenceProtocol(
        site_oxidation_states=tuple(
            int(value) for value in data["site_oxidation_states"]
        ),
        pair_parameters=parameters,
        candidate_site_indices=tuple(
            int(value) for value in data.get("candidate_site_indices", [])
        ),
        minimum_bond_valence=float(data.get("minimum_bond_valence", 1e-4)),
        maximum_distance_angstrom=float(
            data.get("maximum_distance_angstrom", 8.0)
        ),
        vector_cutoff=float(data.get("vector_cutoff", 0.5)),
        dummy_offset_angstrom=float(data.get("dummy_offset_angstrom", 1.0)),
        note=str(data.get("note", "")),
    )
    protocol.validate(structure)
    return protocol


def _pair_key(
    symbol_a: str,
    oxidation_a: int,
    symbol_b: str,
    oxidation_b: int,
) -> tuple[str, int, str, int] | None:
    if oxidation_a * oxidation_b >= 0:
        return None
    if oxidation_a > 0:
        return (symbol_a, oxidation_a, symbol_b, oxidation_b)
    return (symbol_b, oxidation_b, symbol_a, oxidation_a)


def analyze_explicit_bond_valence_site(
    structure: Structure,
    site_index: int,
    protocol: ExplicitBondValenceProtocol,
) -> dict[str, Any]:
    protocol.validate(structure)
    parameter_map = protocol.parameter_map
    center_symbol = structure.species[site_index]
    center_oxidation = protocol.site_oxidation_states[site_index]
    neighbors = _neighbor_images(
        structure, site_index, protocol.maximum_distance_angstrom
    )
    contributions: list[dict[str, Any]] = []
    scalar_sum = 0.0
    vector_sum: Vec3 = (0.0, 0.0, 0.0)
    for neighbor in neighbors:
        neighbor_symbol = structure.species[neighbor.site_index]
        neighbor_oxidation = protocol.site_oxidation_states[neighbor.site_index]
        key = _pair_key(
            center_symbol,
            center_oxidation,
            neighbor_symbol,
            neighbor_oxidation,
        )
        if key is None:
            continue
        parameter = parameter_map.get(key)
        if parameter is None:
            continue
        bond_valence = math.exp(
            (parameter.r0_angstrom - neighbor.distance_angstrom)
            / parameter.b_angstrom
        )
        if bond_valence < protocol.minimum_bond_valence:
            continue
        unit = _vscale(
            neighbor.vector_cart, 1.0 / neighbor.distance_angstrom
        )
        scalar_sum += bond_valence
        vector_sum = _vadd(vector_sum, _vscale(unit, bond_valence))
        contributions.append(
            {
                "site_index": neighbor.site_index,
                "element": neighbor_symbol,
                "oxidation_state": neighbor_oxidation,
                "translation": list(neighbor.translation),
                "distance_angstrom": neighbor.distance_angstrom,
                "bond_valence": bond_valence,
                "parameter_source": parameter.source,
                "r0_angstrom": parameter.r0_angstrom,
                "b_angstrom": parameter.b_angstrom,
            }
        )

    vector_magnitude = _norm(vector_sum)
    directionality = vector_magnitude / scalar_sum if scalar_sum > 0 else 0.0
    mismatch = abs(scalar_sum - abs(center_oxidation)) if contributions else None
    vector_list = list(vector_sum) if contributions else None

    candidate = site_index in set(protocol.candidate_site_indices)
    dummy_created = bool(
        candidate and contributions and vector_magnitude > protocol.vector_cutoff
    )
    dummy_frac: Vec3 | None = None
    dummy_clearance: float | None = None
    dummy_nearest: int | None = None
    if dummy_created:
        open_axis = _vscale(vector_sum, -1.0 / vector_magnitude)
        center_cart = _frac_to_cart(
            structure.frac_coords[site_index], structure.lattice
        )
        dummy_cart = _vadd(
            center_cart, _vscale(open_axis, protocol.dummy_offset_angstrom)
        )
        dummy_frac = _wrap_frac(_cart_to_frac(dummy_cart, structure.lattice))
        dummy_clearance, dummy_nearest = _point_clearance(
            structure,
            dummy_cart,
            excluded_site_index=site_index,
            search_radius_angstrom=protocol.maximum_distance_angstrom,
        )

    return {
        "site_index": site_index,
        "element": center_symbol,
        "oxidation_state": center_oxidation,
        "parameterized_contribution_count": len(contributions),
        "bond_valence_sum": scalar_sum if contributions else None,
        "bond_valence_sum_mismatch": mismatch,
        "bond_valence_vector_sum": vector_list,
        "bond_valence_vector_magnitude": vector_magnitude if contributions else None,
        "bond_valence_vector_imbalance": directionality if contributions else None,
        "candidate_site_explicit": candidate,
        "vector_cutoff": protocol.vector_cutoff,
        "dummy_site_created": dummy_created,
        "dummy_site_fractional": list(dummy_frac) if dummy_frac is not None else None,
        "dummy_site_clearance_angstrom": dummy_clearance,
        "dummy_site_nearest_site_index": dummy_nearest,
        "contributions": contributions,
    }


def _explicit_parameter_coverage(
    structure: Structure,
    protocol: ExplicitBondValenceProtocol,
) -> dict[str, Any]:
    present_pairs: set[tuple[str, int, str, int]] = set()
    states = protocol.site_oxidation_states
    for i, symbol_i in enumerate(structure.species):
        for j in range(i + 1, len(structure.species)):
            key = _pair_key(
                symbol_i, states[i], structure.species[j], states[j]
            )
            if key is not None:
                present_pairs.add(key)
    parameter_keys = set(protocol.parameter_map)
    covered = present_pairs & parameter_keys
    missing = sorted(present_pairs - parameter_keys)
    return {
        "distinct_opposite_charge_pairs": len(present_pairs),
        "covered_pair_count": len(covered),
        "coverage_fraction": (
            len(covered) / len(present_pairs) if present_pairs else 1.0
        ),
        "missing_pairs": [list(key) for key in missing],
        "parameter_sources": sorted(
            {parameter.source for parameter in protocol.pair_parameters}
        ),
    }


def analyze_structure(
    structure: Structure,
    *,
    contact_config: ContactConfig | None = None,
    explicit_protocol: ExplicitBondValenceProtocol | None = None,
    site_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    config = contact_config or ContactConfig()
    config.validate()
    indices = (
        tuple(range(len(structure.species)))
        if site_indices is None
        else tuple(site_indices)
    )
    if not indices:
        raise DescriptorError("site_indices cannot be empty")
    if any(index < 0 or index >= len(structure.species) for index in indices):
        raise DescriptorError("site index out of range")

    radius_sites = [
        analyze_radius_contact_site(structure, index, config)
        for index in indices
    ]
    imbalance = [float(row["contact_vector_imbalance"]) for row in radius_sites]
    effective_cn = [float(row["effective_coordination"]) for row in radius_sites]
    mismatch = [float(row["radius_fit_mismatch"]) for row in radius_sites]
    clearance = [
        float(row["virtual_point_clearance_angstrom"]) for row in radius_sites
    ]
    opposed = [float(row["opposed_void_support"]) for row in radius_sites]
    scalars = {
        "analyzed_site_count": len(indices),
        "contact_vector_imbalance_mean": _mean(imbalance),
        "contact_vector_imbalance_p90": _percentile(imbalance, 0.90),
        "contact_vector_imbalance_max": max(imbalance),
        "effective_coordination_mean": _mean(effective_cn),
        "radius_fit_mismatch_mean": _mean(mismatch),
        "virtual_point_clearance_mean_angstrom": _mean(clearance),
        "opposed_void_support_mean": _mean(opposed),
        "opposed_void_support_max": max(opposed),
        "fallback_site_fraction": (
            sum(bool(row["fallback_used"]) for row in radius_sites)
            / len(radius_sites)
        ),
        "zero_vector_site_fraction": (
            sum(row["open_axis_cartesian"] is None for row in radius_sites)
            / len(radius_sites)
        ),
    }

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "interpretation_scope": INTERPRETATION_SCOPE,
        "structure": {
            "comment": structure.comment,
            "atom_count": len(structure.species),
            "species": list(structure.species),
            "lattice_angstrom": [list(vector) for vector in structure.lattice],
        },
        "radius_contact": {
            "definition": (
                "Original radius-normalized contact first moment; not a "
                "bond-valence quantity. "
                "w_ij=exp(-max(d_ij/(r_i+r_j)-1,0)/tau), "
                "imbalance=|sum(w_ij*u_ij)|/sum(w_ij)."
            ),
            "radius_source": (
                "Cordero-style single-bond covalent radii; fixed embedded "
                "lookup. See examples/bv_directionality/README.md."
            ),
            "protocol": dataclasses.asdict(config),
            "scalars": scalars,
            "sites": radius_sites,
        },
        "explicit_bond_valence": None,
    }

    if explicit_protocol is not None:
        explicit_protocol.validate(structure)
        bvs_sites = [
            analyze_explicit_bond_valence_site(
                structure, index, explicit_protocol
            )
            for index in indices
        ]
        available = [
            row for row in bvs_sites if row["bond_valence_sum"] is not None
        ]
        bvs_scalars = {
            "available_site_count": len(available),
            "available_site_fraction": len(available) / len(bvs_sites),
            "bond_valence_sum_mismatch_mean": _mean(
                [
                    float(row["bond_valence_sum_mismatch"])
                    for row in available
                ]
            ),
            "bond_valence_vector_magnitude_mean": _mean(
                [
                    float(row["bond_valence_vector_magnitude"])
                    for row in available
                ]
            ),
            "bond_valence_vector_imbalance_mean": _mean(
                [
                    float(row["bond_valence_vector_imbalance"])
                    for row in available
                ]
            ),
            "dummy_site_count": sum(
                bool(row["dummy_site_created"]) for row in bvs_sites
            ),
        }
        result["explicit_bond_valence"] = {
            "definition": (
                "Paper-aligned site BVS/BVSM/vector equations with caller-"
                "supplied oxidation states and pair parameters. Exponential "
                "tails below minimum_bond_valence are truncated. "
                "bond_valence_vector_imbalance=|Psi|/BVS is a newly "
                "introduced normalized extension."
            ),
            "not_implemented": (
                "No voxel BVSM map, Morse/Coulomb BVSE field, percolation "
                "analysis, barrier, conductivity, or automatic oxidation-"
                "state assignment."
            ),
            "protocol": {
                "schema": EXPLICIT_PROTOCOL_SCHEMA,
                "site_oxidation_states": list(
                    explicit_protocol.site_oxidation_states
                ),
                "candidate_site_indices": list(
                    explicit_protocol.candidate_site_indices
                ),
                "minimum_bond_valence": (
                    explicit_protocol.minimum_bond_valence
                ),
                "maximum_distance_angstrom": (
                    explicit_protocol.maximum_distance_angstrom
                ),
                "vector_cutoff": explicit_protocol.vector_cutoff,
                "dummy_offset_angstrom": (
                    explicit_protocol.dummy_offset_angstrom
                ),
                "note": explicit_protocol.note,
            },
            "coverage": _explicit_parameter_coverage(
                structure, explicit_protocol
            ),
            "scalars": bvs_scalars,
            "sites": bvs_sites,
        }
    return result


def analyze_structure_file(
    structure_file: str | pathlib.Path,
    *,
    explicit_protocol_file: str | pathlib.Path | None = None,
    contact_config: ContactConfig | None = None,
    site_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    structure = read_poscar(structure_file)
    protocol = (
        load_explicit_protocol(explicit_protocol_file, structure)
        if explicit_protocol_file is not None
        else None
    )
    return analyze_structure(
        structure,
        contact_config=contact_config,
        explicit_protocol=protocol,
        site_indices=site_indices,
    )
