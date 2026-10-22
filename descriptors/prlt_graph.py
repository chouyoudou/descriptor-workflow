"""Periodic radius graph and translation-rank analysis for PRLT."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree

try:
    from .prlt_types import ElementLookup, ResourceLimitError, Settings, StructureData
except ImportError:  # Flat source layout in the private Actions bundle.
    from prlt_types import ElementLookup, ResourceLimitError, Settings, StructureData


@dataclass(frozen=True)
class PeriodicEdge:
    edge_id: int
    u: int
    v: int
    shift: tuple[int, int, int]
    vector: np.ndarray
    distance: float
    normalized_distance: float


@dataclass(frozen=True)
class Arc:
    target: int
    shift: tuple[int, int, int]
    edge_id: int


@dataclass(frozen=True)
class PeriodicComponent:
    component_id: int
    atoms: tuple[int, ...]
    edge_ids: tuple[int, ...]
    rank: int
    cycle_translations: tuple[tuple[int, int, int], ...]
    normal: np.ndarray | None


def _translation_grid(lattice: np.ndarray, cutoff: float, settings: Settings) -> np.ndarray:
    inverse = np.linalg.inv(lattice)
    bounds = [int(math.ceil(cutoff * float(np.linalg.norm(inverse[:, axis])))) + 1 for axis in range(3)]
    count = int(np.prod([2 * bound + 1 for bound in bounds], dtype=np.int64))
    if count > settings.max_translation_images:
        raise ResourceLimitError(
            f"periodic image grid requires {count} translations; limit is {settings.max_translation_images}"
        )
    shifts = np.asarray(
        list(
            itertools.product(
                range(-bounds[0], bounds[0] + 1),
                range(-bounds[1], bounds[1] + 1),
                range(-bounds[2], bounds[2] + 1),
            )
        ),
        dtype=int,
    )
    return shifts


def _canonical_edge(
    i: int,
    j: int,
    shift: tuple[int, int, int],
) -> tuple[int, int, tuple[int, int, int], bool]:
    reverse_shift = tuple(-value for value in shift)
    forward = (i, j, shift)
    reverse = (j, i, reverse_shift)
    if reverse < forward:
        return j, i, reverse_shift, True
    return i, j, shift, False


def build_periodic_graph(
    structure: StructureData,
    lookup: ElementLookup,
    settings: Settings,
) -> tuple[list[PeriodicEdge], list[list[Arc]], dict[str, Any]]:
    """Build a deterministic undirected periodic contact graph from neutral radii."""

    atom_count = structure.atom_count
    if atom_count > settings.max_atoms:
        raise ResourceLimitError(f"atom count {atom_count} exceeds limit {settings.max_atoms}")
    radii = np.asarray([lookup.covalent_radius(symbol) for symbol in structure.species], dtype=float)
    max_pair_cutoff = min(settings.max_search_radius, settings.bond_ratio * 2.0 * float(np.max(radii)))
    shifts = _translation_grid(structure.lattice, max_pair_cutoff, settings)
    image_atom_count = int(atom_count * shifts.shape[0])
    if image_atom_count > settings.max_image_atoms:
        raise ResourceLimitError(
            f"periodic image atom count {image_atom_count} exceeds limit {settings.max_image_atoms}"
        )

    central_cart = structure.fractional @ structure.lattice
    # Shift-major ordering makes the image metadata cheap and deterministic.
    image_fractional = structure.fractional[None, :, :] + shifts[:, None, :]
    image_cart = image_fractional.reshape(-1, 3) @ structure.lattice
    image_atoms = np.tile(np.arange(atom_count, dtype=int), shifts.shape[0])
    image_shifts = np.repeat(shifts, atom_count, axis=0)
    tree = cKDTree(image_cart)
    neighborhoods = tree.query_ball_point(central_cart, max_pair_cutoff + settings.distance_tolerance)
    candidate_count = sum(len(items) for items in neighborhoods)
    if candidate_count > settings.max_neighbor_candidates:
        raise ResourceLimitError(
            f"neighbor candidate count {candidate_count} exceeds limit {settings.max_neighbor_candidates}"
        )

    keyed: dict[tuple[int, int, tuple[int, int, int]], tuple[np.ndarray, float, float]] = {}
    excluded_short_contacts = 0
    tested_within_global_cutoff = 0
    for i, image_indices in enumerate(neighborhoods):
        for image_index in image_indices:
            j = int(image_atoms[image_index])
            shift_array = image_shifts[image_index]
            shift = tuple(int(value) for value in shift_array)
            if i == j and shift == (0, 0, 0):
                continue
            vector = np.asarray(image_cart[image_index] - central_cart[i], dtype=float)
            distance = float(np.linalg.norm(vector))
            if distance <= settings.distance_tolerance:
                excluded_short_contacts += 1
                continue
            tested_within_global_cutoff += 1
            if distance < settings.min_distance:
                # Split/duplicate sites are not silently converted into a bond.
                excluded_short_contacts += 1
                continue
            denominator = float(radii[i] + radii[j])
            normalized = distance / denominator
            if normalized > settings.bond_ratio + settings.distance_tolerance:
                continue
            u, v, canonical_shift, reversed_orientation = _canonical_edge(i, j, shift)
            canonical_vector = -vector if reversed_orientation else vector
            key = (u, v, canonical_shift)
            previous = keyed.get(key)
            if previous is None or distance < previous[1] - settings.distance_tolerance:
                keyed[key] = (canonical_vector, distance, normalized)

    if len(keyed) > settings.max_edges:
        raise ResourceLimitError(f"bond edge count {len(keyed)} exceeds limit {settings.max_edges}")
    edges: list[PeriodicEdge] = []
    for edge_id, key in enumerate(sorted(keyed)):
        u, v, shift = key
        vector, distance, normalized = keyed[key]
        edges.append(
            PeriodicEdge(
                edge_id=edge_id,
                u=u,
                v=v,
                shift=shift,
                vector=np.asarray(vector, dtype=float),
                distance=float(distance),
                normalized_distance=float(normalized),
            )
        )
    adjacency: list[list[Arc]] = [[] for _ in range(atom_count)]
    for edge in edges:
        adjacency[edge.u].append(Arc(edge.v, edge.shift, edge.edge_id))
        adjacency[edge.v].append(
            Arc(edge.u, tuple(-value for value in edge.shift), edge.edge_id)
        )
    for arcs in adjacency:
        arcs.sort(key=lambda arc: (arc.target, arc.shift, arc.edge_id))
    diagnostics = {
        "translation_count": int(shifts.shape[0]),
        "image_atom_count": image_atom_count,
        "neighbor_candidate_count": candidate_count,
        "tested_within_global_cutoff": tested_within_global_cutoff,
        "excluded_short_contacts": excluded_short_contacts,
        "edge_count": len(edges),
        "max_pair_cutoff_angstrom": max_pair_cutoff,
        "maximum_degree": max((len(arcs) for arcs in adjacency), default=0),
        "mean_normalized_edge_distance": (
            float(np.mean([edge.normalized_distance for edge in edges])) if edges else None
        ),
    }
    return edges, adjacency, diagnostics


def _canonical_axis(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-14:
        raise ValueError("zero normal")
    result = np.asarray(vector, dtype=float) / norm
    for value in result:
        if abs(float(value)) > 1.0e-12:
            if value < 0.0:
                result = -result
            break
    return result


def analyze_components(
    structure: StructureData,
    edges: list[PeriodicEdge],
    adjacency: list[list[Arc]],
) -> tuple[list[PeriodicComponent], np.ndarray]:
    """Find quotient components and exact periodic translation ranks."""

    atom_count = structure.atom_count
    atom_component = np.full(atom_count, -1, dtype=int)
    components: list[PeriodicComponent] = []
    for root in range(atom_count):
        if atom_component[root] >= 0 or not adjacency[root]:
            continue
        component_id = len(components)
        queue = [root]
        atom_component[root] = component_id
        potentials: dict[int, np.ndarray] = {root: np.zeros(3, dtype=int)}
        atoms: list[int] = []
        residuals: list[tuple[int, int, int]] = []
        edge_ids: set[int] = set()
        cursor = 0
        while cursor < len(queue):
            atom = queue[cursor]
            cursor += 1
            atoms.append(atom)
            base = potentials[atom]
            for arc in adjacency[atom]:
                edge_ids.add(arc.edge_id)
                expected = base + np.asarray(arc.shift, dtype=int)
                if arc.target not in potentials:
                    potentials[arc.target] = expected
                    atom_component[arc.target] = component_id
                    queue.append(arc.target)
                else:
                    residual = expected - potentials[arc.target]
                    if np.any(residual):
                        residuals.append(tuple(int(value) for value in residual))
        unique_residuals = sorted(set(residuals))
        if unique_residuals:
            matrix = np.asarray(unique_residuals, dtype=float)
            rank = int(np.linalg.matrix_rank(matrix, tol=1.0e-9))
        else:
            rank = 0
        rank = min(rank, 3)
        normal: np.ndarray | None = None
        if rank == 2:
            cartesian_translations = np.asarray(unique_residuals, dtype=float) @ structure.lattice
            _, singular_values, right_vectors = np.linalg.svd(cartesian_translations, full_matrices=True)
            if singular_values.size >= 2 and singular_values[1] > 1.0e-10:
                normal = _canonical_axis(right_vectors[-1])
        components.append(
            PeriodicComponent(
                component_id=component_id,
                atoms=tuple(sorted(atoms)),
                edge_ids=tuple(sorted(edge_ids)),
                rank=rank,
                cycle_translations=tuple(unique_residuals),
                normal=normal,
            )
        )
    return components, atom_component


def arc_target_state(
    state: tuple[int, tuple[int, int, int]],
    arc: Arc,
) -> tuple[int, tuple[int, int, int]]:
    image = state[1]
    return (
        arc.target,
        tuple(image[axis] + arc.shift[axis] for axis in range(3)),
    )


def translated_arc_key(
    source: tuple[int, tuple[int, int, int]],
    target: tuple[int, tuple[int, int, int]],
    edge_id: int,
) -> tuple[int, tuple[int, int, int], int, tuple[int, int, int], int]:
    forward = (source[0], source[1], target[0], target[1], edge_id)
    reverse = (target[0], target[1], source[0], source[1], edge_id)
    return min(forward, reverse)
