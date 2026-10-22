"""Internal coordination, bridge-path, and aggregate descriptors."""
from __future__ import annotations

from collections import Counter
import math
from typing import Mapping, Sequence

from _layer_bridge_graph import (
    _adjacency,
    _component_summary,
    build_periodic_bonds,
    component_dimensionalities,
)
from _layer_bridge_model import (
    DEFAULT_BOND_SCALE,
    DEFAULT_MAX_BRIDGE_ATOMS,
    DESCRIPTOR_NAMES,
    SCHEMA,
    Edge,
    ElementData,
    Structure,
    _ELEMENT_SYMBOLS,
    _add_int,
    _angle_deg,
    _coefficient_of_variation,
    _direction_anisotropy,
    _frac_vector_to_cart,
    _mean,
    _norm,
    _population_std,
    _sub_int,
    default_element_table,
)


def _site_cart(
    structure: Structure,
    site: int,
    offset: Sequence[int] = (0, 0, 0),
) -> tuple[float, float, float]:
    fractional = tuple(
        structure.frac_coords[site][axis] + offset[axis] for axis in range(3)
    )
    return _frac_vector_to_cart(fractional, structure.lattice)


def _coordination_summary(
    structure: Structure,
    edges: Sequence[Edge],
    table: Mapping[str, ElementData],
) -> tuple[dict[str, float], dict[str, object]]:
    graph = _adjacency(len(structure.species), edges)
    metal_sites = [
        index
        for index, symbol in enumerate(structure.species)
        if table[symbol].is_metal
    ]
    coordination_numbers: list[float] = []
    site_bond_cv: list[float] = []
    site_anisotropy: list[float] = []
    site_hinge_angles: list[float] = []
    site_hinge_cos_half: list[float] = []
    radius_mismatches: list[float] = []
    atomic_number_contrasts: list[float] = []

    for site in metal_sites:
        symbol = structure.species[site]
        ligand_neighbors = [
            item
            for item in graph[site]
            if not table[structure.species[item[0]]].is_metal
        ]
        coordination_numbers.append(float(len(ligand_neighbors)))
        distances = [item[2] for item in ligand_neighbors]
        site_bond_cv.append(_coefficient_of_variation(distances))
        vectors = [item[3] for item in ligand_neighbors]
        site_anisotropy.append(_direction_anisotropy(vectors))
        angles = []
        for left in range(len(vectors)):
            for right in range(left + 1, len(vectors)):
                angles.append(_angle_deg(vectors[left], vectors[right]))
        if angles:
            site_hinge_angles.append(_mean(angles))
            site_hinge_cos_half.append(
                _mean(
                    [
                        math.cos(math.radians(angle) / 2.0)
                        for angle in angles
                    ]
                )
            )
        for neighbor, _shift, _distance, _vector in ligand_neighbors:
            ligand_symbol = structure.species[neighbor]
            metal_radius = table[symbol].covalent_radius
            ligand_radius = table[ligand_symbol].covalent_radius
            radius_mismatches.append(
                abs(metal_radius - ligand_radius)
                / (metal_radius + ligand_radius)
            )
            atomic_number_contrasts.append(
                abs(
                    table[symbol].atomic_number
                    - table[ligand_symbol].atomic_number
                )
            )

    descriptors = {
        "metal_fraction": len(metal_sites) / len(structure.species),
        "metal_coordination_mean": _mean(coordination_numbers),
        "metal_coordination_std": _population_std(coordination_numbers),
        "metal_ligand_bond_cv_mean": _mean(site_bond_cv),
        "metal_coordination_anisotropy_mean": _mean(site_anisotropy),
        "metal_ligand_hinge_angle_mean_deg": _mean(site_hinge_angles),
        "metal_ligand_hinge_cos_half_mean": _mean(site_hinge_cos_half),
        "metal_ligand_radius_mismatch_mean": _mean(radius_mismatches),
        "metal_ligand_atomic_number_contrast_mean": _mean(
            atomic_number_contrasts
        ),
    }
    diagnostics = {
        "metal_site_count": len(metal_sites),
        "metal_ligand_bond_count": len(radius_mismatches),
    }
    return descriptors, diagnostics


def _normalize_path(
    path: Sequence[tuple[int, tuple[int, int, int]]],
):
    origin = path[0][1]
    return tuple((site, *_sub_int(offset, origin)) for site, offset in path)


def _canonical_path_key(
    path: Sequence[tuple[int, tuple[int, int, int]]],
):
    forward = _normalize_path(path)
    reverse = _normalize_path(list(reversed(path)))
    return min(forward, reverse)


def _bridge_summary(
    structure: Structure,
    edges: Sequence[Edge],
    table: Mapping[str, ElementData],
    *,
    max_bridge_atoms: int,
    max_bridge_paths: int,
) -> tuple[dict[str, float], dict[str, object]]:
    if type(max_bridge_atoms) is not int or not 1 <= max_bridge_atoms <= 8:
        raise ValueError("max_bridge_atoms must be an integer from 1 through 8")
    if type(max_bridge_paths) is not int or max_bridge_paths < 1:
        raise ValueError("max_bridge_paths must be a positive integer")
    graph = _adjacency(len(structure.species), edges)
    metal_sites = [
        index
        for index, symbol in enumerate(structure.species)
        if table[symbol].is_metal
    ]
    records: dict[tuple, dict[str, object]] = {}
    truncated = False

    for start in metal_sites:
        start_state = (start, (0, 0, 0))

        def visit(
            state: tuple[int, tuple[int, int, int]],
            path: list[tuple[int, tuple[int, int, int]]],
            edge_lengths: list[float],
            visited: set[tuple[int, tuple[int, int, int]]],
            internal_atoms: int,
        ) -> None:
            nonlocal truncated
            if truncated:
                return
            site, offset = state
            for neighbor, shift, distance, _vector in graph[site]:
                next_state = (neighbor, _add_int(offset, shift))
                if next_state in visited:
                    continue
                next_is_metal = table[structure.species[neighbor]].is_metal
                if next_is_metal:
                    if internal_atoms < 1:
                        continue
                    candidate_path = path + [next_state]
                    key = _canonical_path_key(candidate_path)
                    if key not in records:
                        coords = [
                            _site_cart(structure, item_site, item_offset)
                            for item_site, item_offset in candidate_path
                        ]
                        path_length = sum(edge_lengths) + distance
                        chord = _norm(
                            tuple(
                                coords[-1][axis] - coords[0][axis]
                                for axis in range(3)
                            )
                        )
                        angle = 0.0
                        if internal_atoms == 1:
                            left = tuple(
                                coords[0][axis] - coords[1][axis]
                                for axis in range(3)
                            )
                            right = tuple(
                                coords[-1][axis] - coords[1][axis]
                                for axis in range(3)
                            )
                            angle = _angle_deg(left, right)
                        records[key] = {
                            "internal_atoms": internal_atoms,
                            "path_length": path_length,
                            "chord": chord,
                            "straightness": (
                                chord / path_length if path_length > 0 else 0.0
                            ),
                            "single_atom_angle": angle,
                        }
                        if len(records) >= max_bridge_paths:
                            truncated = True
                            return
                    continue
                if internal_atoms >= max_bridge_atoms:
                    continue
                visited.add(next_state)
                visit(
                    next_state,
                    path + [next_state],
                    edge_lengths + [distance],
                    visited,
                    internal_atoms + 1,
                )
                visited.remove(next_state)

        visit(start_state, [start_state], [], {start_state}, 0)
        if truncated:
            break

    rows = list(records.values())
    internal = [float(row["internal_atoms"]) for row in rows]
    straightness = [float(row["straightness"]) for row in rows]
    single_angles = [
        float(row["single_atom_angle"])
        for row in rows
        if int(row["internal_atoms"]) == 1
    ]
    descriptors = {
        # Each undirected periodic path contributes one incidence at each endpoint.
        "bridge_path_incidence_per_metal": (
            2.0 * len(rows) / len(metal_sites) if metal_sites else 0.0
        ),
        "bridge_internal_atoms_mean": _mean(internal),
        "bridge_straightness_mean": _mean(straightness),
        "single_atom_bridge_angle_mean_deg": _mean(single_angles),
    }
    diagnostics = {
        "bridge_path_count": len(rows),
        "single_atom_bridge_count": len(single_angles),
        "bridge_path_truncated": truncated,
        "max_bridge_atoms": max_bridge_atoms,
        "max_bridge_paths": max_bridge_paths,
    }
    return descriptors, diagnostics


def analyze_structure(
    structure: Structure,
    *,
    element_table: Mapping[str, ElementData] | None = None,
    bond_scale: float = DEFAULT_BOND_SCALE,
    minimum_distance: float = 0.35,
    max_bridge_atoms: int = DEFAULT_MAX_BRIDGE_ATOMS,
    max_bridge_paths: int = 10000,
) -> dict[str, object]:
    """Compute the fixed descriptor vector and transparent diagnostics."""
    table = dict(element_table or default_element_table())
    missing = sorted({symbol for symbol in structure.species if symbol not in table})
    if missing:
        raise ValueError(
            "missing element data for "
            + ", ".join(missing)
            + "; provide a JSON override with covalent_radius and is_metal"
        )
    edges = build_periodic_bonds(
        structure,
        element_table=table,
        bond_scale=bond_scale,
        minimum_distance=minimum_distance,
    )
    components = component_dimensionalities(structure, edges)
    component_values, component_diagnostics = _component_summary(
        structure, components, table
    )
    coordination_values, coordination_diagnostics = _coordination_summary(
        structure, edges, table
    )
    bridge_values, bridge_diagnostics = _bridge_summary(
        structure,
        edges,
        table,
        max_bridge_atoms=max_bridge_atoms,
        max_bridge_paths=max_bridge_paths,
    )
    descriptors: dict[str, float] = {}
    descriptors.update(component_values)
    descriptors["bond_direction_anisotropy"] = _direction_anisotropy(
        [edge.vector for edge in edges]
    )
    descriptors.update(coordination_values)
    descriptors.update(bridge_values)
    if tuple(descriptors) != DESCRIPTOR_NAMES:
        raise RuntimeError(
            "internal descriptor order does not match DESCRIPTOR_NAMES"
        )
    for name, value in descriptors.items():
        if not math.isfinite(value):
            raise RuntimeError(f"descriptor {name} is not finite")
    formula_counts = Counter(structure.species)
    formula = "".join(
        symbol
        + (str(formula_counts[symbol]) if formula_counts[symbol] != 1 else "")
        for symbol in _ELEMENT_SYMBOLS
        if symbol in formula_counts
    )
    return {
        "schema": SCHEMA,
        "structure": {
            "title": structure.title,
            "atom_count": len(structure.species),
            "formula": formula,
        },
        "parameters": {
            "bond_rule": (
                "distance <= bond_scale * "
                "(covalent_radius_i + covalent_radius_j)"
            ),
            "bond_scale": float(bond_scale),
            "minimum_distance_angstrom": float(minimum_distance),
            "max_bridge_atoms": max_bridge_atoms,
            "element_table": (
                "built-in Cordero-style radii plus explicit overrides"
            ),
        },
        "descriptors": descriptors,
        "diagnostics": {
            "bond_count": len(edges),
            "component_dimensionalities": components,
            **component_diagnostics,
            **coordination_diagnostics,
            **bridge_diagnostics,
            "scope_warning": (
                "static structure proxy only; do not interpret as pressure, "
                "magnetic, elastic, energetic, kinetic, or synthesis-history "
                "information"
            ),
        },
    }


def analyze_cutoff_ensemble(
    structure: Structure,
    *,
    bond_scales: Sequence[float] = (1.10, DEFAULT_BOND_SCALE, 1.26),
    element_table: Mapping[str, ElementData] | None = None,
    minimum_distance: float = 0.35,
    max_bridge_atoms: int = DEFAULT_MAX_BRIDGE_ATOMS,
    max_bridge_paths: int = 10000,
) -> dict[str, object]:
    """Quantify sensitivity to the explicitly heuristic bond cutoff."""
    if not bond_scales:
        raise ValueError("bond_scales must not be empty")
    runs = [
        analyze_structure(
            structure,
            element_table=element_table,
            bond_scale=float(scale),
            minimum_distance=minimum_distance,
            max_bridge_atoms=max_bridge_atoms,
            max_bridge_paths=max_bridge_paths,
        )
        for scale in bond_scales
    ]
    ranges = {}
    for name in DESCRIPTOR_NAMES:
        values = [float(run["descriptors"][name]) for run in runs]
        ranges[name] = {
            "min": min(values),
            "max": max(values),
            "span": max(values) - min(values),
        }
    return {
        "schema": "layer-bridge-cutoff-ensemble/1",
        "bond_scales": [float(scale) for scale in bond_scales],
        "runs": runs,
        "descriptor_ranges": ranges,
    }
