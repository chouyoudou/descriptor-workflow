"""Fixed scalar aggregation for PRLT."""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from .prlt_graph import PeriodicComponent, PeriodicEdge
    from .prlt_rings import RingRecord, ring_edge_incidence
    from .prlt_types import COLUMNS, ElementLookup, StructureData
except ImportError:  # Flat source layout in the private Actions bundle.
    from prlt_graph import PeriodicComponent, PeriodicEdge
    from prlt_rings import RingRecord, ring_edge_incidence
    from prlt_types import COLUMNS, ElementLookup, StructureData


def categorical_assortativity(
    structure: StructureData,
    edges: Sequence[PeriodicEdge],
) -> float | None:
    """Newman's categorical assortativity for an undirected edge set."""

    if not edges:
        return None
    labels = sorted({structure.species[edge.u] for edge in edges} | {structure.species[edge.v] for edge in edges})
    index = {label: position for position, label in enumerate(labels)}
    mixing = np.zeros((len(labels), len(labels)), dtype=float)
    for edge in edges:
        first = index[structure.species[edge.u]]
        second = index[structure.species[edge.v]]
        if first == second:
            mixing[first, first] += 1.0
        else:
            mixing[first, second] += 0.5
            mixing[second, first] += 0.5
    mixing /= float(len(edges))
    marginals = np.sum(mixing, axis=1)
    baseline = float(np.dot(marginals, marginals))
    denominator = 1.0 - baseline
    if denominator <= 1.0e-14:
        return None
    return float((np.trace(mixing) - baseline) / denominator)


def _set_value(
    scalars: dict[str, float | int | None],
    reasons: dict[str, str],
    name: str,
    value: float | int | None,
    reason: str | None = None,
) -> None:
    if value is None:
        scalars[name] = None
        reasons[name] = reason or "unavailable"
    else:
        numeric = float(value) if isinstance(value, np.floating) else value
        if isinstance(numeric, float) and not math.isfinite(numeric):
            scalars[name] = None
            reasons[name] = reason or "non_finite"
        else:
            scalars[name] = numeric


def _rank2_bond_geometry(
    edges: Sequence[PeriodicEdge],
    components: Sequence[PeriodicComponent],
) -> tuple[list[float], int]:
    edge_by_id = {edge.edge_id: edge for edge in edges}
    values: list[float] = []
    missing_normals = 0
    for component in components:
        if component.rank != 2:
            continue
        if component.normal is None:
            missing_normals += 1
            continue
        for edge_id in component.edge_ids:
            vector = edge_by_id[edge_id].vector
            norm = float(np.linalg.norm(vector))
            if norm > 1.0e-14:
                values.append(abs(float(np.dot(vector / norm, component.normal))))
    return values, missing_normals


def aggregate_scalars(
    structure: StructureData,
    lookup: ElementLookup,
    edges: list[PeriodicEdge],
    components: list[PeriodicComponent],
    rings: list[RingRecord],
    ring_search_incomplete: bool,
) -> tuple[dict[str, float | int | None], dict[str, str], str, dict[str, Any]]:
    scalars: dict[str, float | int | None] = {name: None for name in COLUMNS}
    reasons: dict[str, str] = {}
    atom_count = structure.atom_count
    bonded_atoms = {edge.u for edge in edges} | {edge.v for edge in edges}
    max_rank = max((component.rank for component in components), default=0)
    rank2_atoms = {
        atom
        for component in components
        if component.rank == 2
        for atom in component.atoms
    }
    _set_value(scalars, reasons, "prlt_bonded_atom_fraction", len(bonded_atoms) / atom_count)
    _set_value(scalars, reasons, "prlt_max_translation_rank", int(max_rank))
    _set_value(scalars, reasons, "prlt_rank2_atom_fraction", len(rank2_atoms) / atom_count)

    all_assortativity = categorical_assortativity(structure, edges)
    _set_value(
        scalars,
        reasons,
        "prlt_element_assortativity_all_bonds",
        all_assortativity,
        "assortativity_degenerate" if edges else "no_bonds",
    )

    rank2_bond_values, missing_normals = _rank2_bond_geometry(edges, components)
    if rank2_bond_values:
        array = np.asarray(rank2_bond_values, dtype=float)
        _set_value(
            scalars,
            reasons,
            "prlt_rank2_bond_out_of_plane_rms",
            float(np.sqrt(np.mean(array * array))),
        )
        _set_value(
            scalars,
            reasons,
            "prlt_rank2_bond_out_of_plane_p90",
            float(np.quantile(array, 0.90)),
        )
    else:
        reason = "rank2_component_normal_unavailable" if missing_normals else "no_rank2_components"
        _set_value(scalars, reasons, "prlt_rank2_bond_out_of_plane_rms", None, reason)
        _set_value(scalars, reasons, "prlt_rank2_bond_out_of_plane_p90", None, reason)

    ring_columns = {
        "prlt_ring_atom_fraction",
        "prlt_ring_count_per_atom",
        "prlt_shortest_ring_size",
        "prlt_ring_size_mean",
        "prlt_ring_size_std",
        "prlt_ring_even_fraction",
        "prlt_ring_edge_coverage",
        "prlt_shared_ring_edge_fraction",
        "prlt_ring_planarity_rms_norm_mean",
        "prlt_ring_planarity_rms_norm_max",
        "prlt_rank2_ring_fraction",
        "prlt_rank2_ring_normal_alignment_mean",
        "prlt_rank2_ring_normal_alignment_min",
        "prlt_element_assortativity_ring_bonds",
        "prlt_ring_heteroedge_fraction",
        "prlt_ring_species_diversity_mean",
        "prlt_ring_atomic_number_contrast_mean",
        "prlt_ring_radius_contrast_mean",
    }
    if ring_search_incomplete:
        for name in ring_columns:
            _set_value(scalars, reasons, name, None, "ring_search_resource_limit")
        status = "resource_limited_partial"
        return scalars, reasons, status, {
            "bonded_atom_count": len(bonded_atoms),
            "rank2_atom_count": len(rank2_atoms),
            "rank2_bond_sample_count": len(rank2_bond_values),
            "ring_metrics_used": False,
        }

    if not rings:
        _set_value(scalars, reasons, "prlt_ring_atom_fraction", 0.0)
        _set_value(scalars, reasons, "prlt_ring_count_per_atom", 0.0)
        _set_value(scalars, reasons, "prlt_ring_edge_coverage", 0.0)
        _set_value(scalars, reasons, "prlt_shared_ring_edge_fraction", 0.0)
        for name in ring_columns - {
            "prlt_ring_atom_fraction",
            "prlt_ring_count_per_atom",
            "prlt_ring_edge_coverage",
            "prlt_shared_ring_edge_fraction",
        }:
            _set_value(scalars, reasons, name, None, "no_admitted_rings")
        status = "domain_no_bonds" if not edges else "domain_no_shortest_path_rings"
        return scalars, reasons, status, {
            "bonded_atom_count": len(bonded_atoms),
            "rank2_atom_count": len(rank2_atoms),
            "rank2_bond_sample_count": len(rank2_bond_values),
            "ring_metrics_used": True,
            "ring_count": 0,
        }

    sizes = np.asarray([ring.size for ring in rings], dtype=float)
    ring_atoms = {state[0] for ring in rings for state in ring.states}
    incidence = ring_edge_incidence(rings)
    ring_edge_ids = sorted(incidence)
    edge_by_id = {edge.edge_id: edge for edge in edges}
    ring_edges = [edge_by_id[edge_id] for edge_id in ring_edge_ids]
    _set_value(scalars, reasons, "prlt_ring_atom_fraction", len(ring_atoms) / atom_count)
    _set_value(scalars, reasons, "prlt_ring_count_per_atom", len(rings) / atom_count)
    _set_value(scalars, reasons, "prlt_shortest_ring_size", int(np.min(sizes)))
    _set_value(scalars, reasons, "prlt_ring_size_mean", float(np.mean(sizes)))
    _set_value(scalars, reasons, "prlt_ring_size_std", float(np.std(sizes)))
    _set_value(scalars, reasons, "prlt_ring_even_fraction", float(np.mean((sizes.astype(int) % 2) == 0)))
    _set_value(scalars, reasons, "prlt_ring_edge_coverage", len(ring_edge_ids) / len(edges) if edges else 0.0)
    _set_value(
        scalars,
        reasons,
        "prlt_shared_ring_edge_fraction",
        sum(value >= 2 for value in incidence.values()) / len(incidence),
    )
    planarities = np.asarray([ring.planarity_rms_norm for ring in rings], dtype=float)
    _set_value(scalars, reasons, "prlt_ring_planarity_rms_norm_mean", float(np.mean(planarities)))
    _set_value(scalars, reasons, "prlt_ring_planarity_rms_norm_max", float(np.max(planarities)))

    component_by_id = {component.component_id: component for component in components}
    rank2_rings = [ring for ring in rings if component_by_id[ring.component_id].rank == 2]
    _set_value(scalars, reasons, "prlt_rank2_ring_fraction", len(rank2_rings) / len(rings))
    alignments: list[float] = []
    for ring in rank2_rings:
        normal = component_by_id[ring.component_id].normal
        if normal is not None:
            alignments.append(abs(float(np.dot(normal, ring.normal))))
    if alignments:
        _set_value(
            scalars,
            reasons,
            "prlt_rank2_ring_normal_alignment_mean",
            float(np.mean(alignments)),
        )
        _set_value(
            scalars,
            reasons,
            "prlt_rank2_ring_normal_alignment_min",
            float(np.min(alignments)),
        )
    else:
        reason = "rank2_component_normal_unavailable" if rank2_rings else "no_rank2_rings"
        _set_value(scalars, reasons, "prlt_rank2_ring_normal_alignment_mean", None, reason)
        _set_value(scalars, reasons, "prlt_rank2_ring_normal_alignment_min", None, reason)

    ring_assortativity = categorical_assortativity(structure, ring_edges)
    _set_value(
        scalars,
        reasons,
        "prlt_element_assortativity_ring_bonds",
        ring_assortativity,
        "assortativity_degenerate",
    )
    hetero = [structure.species[edge.u] != structure.species[edge.v] for edge in ring_edges]
    _set_value(scalars, reasons, "prlt_ring_heteroedge_fraction", float(np.mean(hetero)))

    diversities: list[float] = []
    for ring in rings:
        counts = Counter(structure.species[state[0]] for state in ring.states)
        total = float(ring.size)
        diversities.append(1.0 - sum((count / total) ** 2 for count in counts.values()))
    _set_value(scalars, reasons, "prlt_ring_species_diversity_mean", float(np.mean(diversities)))

    atomic_contrasts: list[float] = []
    radius_contrasts: list[float] = []
    for edge in ring_edges:
        first_symbol = structure.species[edge.u]
        second_symbol = structure.species[edge.v]
        first_z = lookup.atomic_number(first_symbol)
        second_z = lookup.atomic_number(second_symbol)
        first_radius = lookup.covalent_radius(first_symbol)
        second_radius = lookup.covalent_radius(second_symbol)
        atomic_contrasts.append(abs(first_z - second_z) / (first_z + second_z))
        radius_contrasts.append(abs(first_radius - second_radius) / (first_radius + second_radius))
    _set_value(
        scalars,
        reasons,
        "prlt_ring_atomic_number_contrast_mean",
        float(np.mean(atomic_contrasts)),
    )
    _set_value(
        scalars,
        reasons,
        "prlt_ring_radius_contrast_mean",
        float(np.mean(radius_contrasts)),
    )

    status = "ok" if rank2_rings else "domain_no_rank2_ring_layer"
    diagnostics = {
        "bonded_atom_count": len(bonded_atoms),
        "rank2_atom_count": len(rank2_atoms),
        "rank2_bond_sample_count": len(rank2_bond_values),
        "ring_metrics_used": True,
        "ring_count": len(rings),
        "ring_atom_count": len(ring_atoms),
        "ring_edge_count": len(ring_edge_ids),
        "rank2_ring_count": len(rank2_rings),
    }
    return scalars, reasons, status, diagnostics
