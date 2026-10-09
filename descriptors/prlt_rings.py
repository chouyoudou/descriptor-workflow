"""Bounded periodic shortest-path ring enumeration and geometry for PRLT."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

try:
    from .prlt_graph import Arc, PeriodicComponent, PeriodicEdge, arc_target_state
    from .prlt_types import Settings, StructureData
except ImportError:  # Flat source layout in the private Actions bundle.
    from prlt_graph import Arc, PeriodicComponent, PeriodicEdge, arc_target_state
    from prlt_types import Settings, StructureData

State = tuple[int, tuple[int, int, int]]


@dataclass(frozen=True)
class RingRecord:
    states: tuple[State, ...]
    edge_ids: tuple[int, ...]
    component_id: int
    planarity_rms_norm: float
    normal: np.ndarray

    @property
    def size(self) -> int:
        return len(self.states)


def _canonical_ring_key(states: tuple[State, ...]) -> tuple[tuple[int, int, int, int], ...]:
    count = len(states)
    candidates: list[tuple[tuple[int, int, int, int], ...]] = []
    for orientation in (states, tuple(reversed(states))):
        for offset in range(count):
            rotated = orientation[offset:] + orientation[:offset]
            base_image = rotated[0][1]
            encoded = tuple(
                (
                    atom,
                    image[0] - base_image[0],
                    image[1] - base_image[1],
                    image[2] - base_image[2],
                )
                for atom, image in rotated
            )
            candidates.append(encoded)
    return min(candidates)


def _removed_root_occurrence(source: State, target: State, edge_id: int, root_edge: PeriodicEdge) -> bool:
    if edge_id != root_edge.edge_id:
        return False
    root_source: State = (root_edge.u, (0, 0, 0))
    root_target: State = (root_edge.v, root_edge.shift)
    return (source == root_source and target == root_target) or (
        source == root_target and target == root_source
    )


def _shortest_paths_without_root(
    adjacency: list[list[Arc]],
    root_edge: PeriodicEdge,
    settings: Settings,
) -> tuple[list[tuple[tuple[State, ...], tuple[int, ...]]], dict[str, Any]]:
    start: State = (root_edge.u, (0, 0, 0))
    target: State = (root_edge.v, root_edge.shift)
    max_path_length = settings.max_ring_size - 1
    distances: dict[State, int] = {start: 0}
    predecessors: dict[State, list[tuple[State, int]]] = {start: []}
    queue: deque[State] = deque([start])
    target_distance: int | None = None
    truncated = False
    while queue:
        state = queue.popleft()
        depth = distances[state]
        if target_distance is not None and depth >= target_distance:
            continue
        if depth >= max_path_length:
            continue
        for arc in adjacency[state[0]]:
            next_state = arc_target_state(state, arc)
            if _removed_root_occurrence(state, next_state, arc.edge_id, root_edge):
                continue
            next_depth = depth + 1
            prior_depth = distances.get(next_state)
            if prior_depth is None:
                if len(distances) >= settings.max_bfs_states_per_edge:
                    truncated = True
                    queue.clear()
                    break
                distances[next_state] = next_depth
                predecessors[next_state] = [(state, arc.edge_id)]
                queue.append(next_state)
                if next_state == target:
                    target_distance = next_depth
            elif prior_depth == next_depth:
                pred_list = predecessors[next_state]
                if len(pred_list) < settings.max_paths_per_edge * 4:
                    pred_list.append((state, arc.edge_id))
    if truncated or target not in distances:
        return [], {
            "truncated": truncated,
            "visited_states": len(distances),
            "target_distance": distances.get(target),
        }
    distance = distances[target]
    if distance + 1 < 3 or distance + 1 > settings.max_ring_size:
        return [], {
            "truncated": False,
            "visited_states": len(distances),
            "target_distance": distance,
        }

    paths: list[tuple[tuple[State, ...], tuple[int, ...]]] = []
    reverse_states: list[State] = [target]
    reverse_edges: list[int] = []

    def backtrack(current: State) -> None:
        if len(paths) >= settings.max_paths_per_edge:
            return
        if current == start:
            states = tuple(reversed(reverse_states))
            edge_ids = tuple(reversed(reverse_edges))
            paths.append((states, edge_ids))
            return
        for previous, edge_id in predecessors.get(current, ()):
            reverse_states.append(previous)
            reverse_edges.append(edge_id)
            backtrack(previous)
            reverse_edges.pop()
            reverse_states.pop()
            if len(paths) >= settings.max_paths_per_edge:
                break

    backtrack(target)
    return paths, {
        "truncated": False,
        "visited_states": len(distances),
        "target_distance": distance,
        "shortest_path_count_retained": len(paths),
    }


def _normalized_pair_key(source: State, target: State) -> tuple[int, int, int, int, int]:
    delta = tuple(target[1][axis] - source[1][axis] for axis in range(3))
    forward = (source[0], target[0], delta[0], delta[1], delta[2])
    reverse = (target[0], source[0], -delta[0], -delta[1], -delta[2])
    return min(forward, reverse)


def _has_path_at_most(
    adjacency: list[list[Arc]],
    source: State,
    target: State,
    max_depth: int,
    cache: dict[tuple[tuple[int, int, int, int, int], int], bool],
) -> bool:
    if max_depth < 0:
        return False
    key = (_normalized_pair_key(source, target), max_depth)
    cached = cache.get(key)
    if cached is not None:
        return cached
    if source == target:
        cache[key] = True
        return True
    distances: dict[State, int] = {source: 0}
    queue: deque[State] = deque([source])
    found = False
    while queue and not found:
        state = queue.popleft()
        depth = distances[state]
        if depth >= max_depth:
            continue
        for arc in adjacency[state[0]]:
            next_state = arc_target_state(state, arc)
            if next_state == target:
                found = True
                break
            if next_state not in distances:
                distances[next_state] = depth + 1
                queue.append(next_state)
    cache[key] = found
    return found


def _is_shortest_path_ring(
    states: tuple[State, ...],
    adjacency: list[list[Arc]],
    distance_cache: dict[tuple[tuple[int, int, int, int, int], int], bool],
) -> bool:
    count = len(states)
    for first in range(count):
        for second in range(first + 1, count):
            cyclic_distance = min(second - first, count - (second - first))
            if cyclic_distance <= 1:
                continue
            if _has_path_at_most(
                adjacency,
                states[first],
                states[second],
                cyclic_distance - 1,
                distance_cache,
            ):
                return False
    return True


def _ring_geometry(structure: StructureData, states: tuple[State, ...]) -> tuple[float, np.ndarray]:
    coordinates = np.asarray(
        [
            (structure.fractional[atom] + np.asarray(image, dtype=float)) @ structure.lattice
            for atom, image in states
        ],
        dtype=float,
    )
    centered = coordinates - np.mean(coordinates, axis=0)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=True)
    normal = np.asarray(right_vectors[-1], dtype=float)
    normal /= float(np.linalg.norm(normal))
    for value in normal:
        if abs(float(value)) > 1.0e-12:
            if value < 0.0:
                normal = -normal
            break
    residuals = centered @ normal
    rms = float(np.sqrt(np.mean(residuals * residuals)))
    lengths = []
    for index in range(len(states)):
        delta = coordinates[(index + 1) % len(states)] - coordinates[index]
        lengths.append(float(np.linalg.norm(delta)))
    mean_edge = float(np.mean(lengths))
    planarity = rms / mean_edge if mean_edge > 1.0e-14 else math.nan
    return planarity, normal


def enumerate_shortest_path_rings(
    structure: StructureData,
    edges: list[PeriodicEdge],
    adjacency: list[list[Arc]],
    components: list[PeriodicComponent],
    atom_component: np.ndarray,
    settings: Settings,
) -> tuple[list[RingRecord], dict[str, Any], bool]:
    """Enumerate bounded, zero-winding isometric rings.

    Returns ``(rings, diagnostics, incomplete)``. If incomplete is true, callers must
    not interpret the retained prefix as an unbiased ring distribution.
    """

    edge_by_id = {edge.edge_id: edge for edge in edges}
    seen_keys: set[tuple[tuple[int, int, int, int], ...]] = set()
    rings: list[RingRecord] = []
    distance_cache: dict[tuple[tuple[int, int, int, int, int], int], bool] = {}
    candidate_count = 0
    duplicate_count = 0
    rejected_isometric = 0
    searched_root_edges = 0
    skipped_components: list[dict[str, Any]] = []
    truncated_root_edges = 0
    incomplete = False

    for component in components:
        max_degree = max((len(adjacency[atom]) for atom in component.atoms), default=0)
        if len(component.edge_ids) > settings.max_component_edges_for_rings:
            skipped_components.append(
                {
                    "component_id": component.component_id,
                    "reason": "component_edge_limit",
                    "edge_count": len(component.edge_ids),
                }
            )
            incomplete = True
            continue
        if max_degree > settings.max_degree_for_ring_search:
            skipped_components.append(
                {
                    "component_id": component.component_id,
                    "reason": "degree_limit",
                    "maximum_degree": max_degree,
                }
            )
            incomplete = True
            continue
        for edge_id in component.edge_ids:
            searched_root_edges += 1
            root_edge = edge_by_id[edge_id]
            paths, path_diagnostics = _shortest_paths_without_root(adjacency, root_edge, settings)
            if path_diagnostics.get("truncated"):
                truncated_root_edges += 1
                incomplete = True
                continue
            for states, path_edge_ids in paths:
                candidate_count += 1
                if candidate_count > settings.max_ring_candidates:
                    incomplete = True
                    break
                if len(set(states)) != len(states):
                    continue
                key = _canonical_ring_key(states)
                if key in seen_keys:
                    duplicate_count += 1
                    continue
                if not _is_shortest_path_ring(states, adjacency, distance_cache):
                    rejected_isometric += 1
                    continue
                planarity, normal = _ring_geometry(structure, states)
                if not math.isfinite(planarity):
                    incomplete = True
                    continue
                seen_keys.add(key)
                component_id = int(atom_component[states[0][0]])
                ring_edges = tuple(path_edge_ids) + (root_edge.edge_id,)
                rings.append(
                    RingRecord(
                        states=states,
                        edge_ids=ring_edges,
                        component_id=component_id,
                        planarity_rms_norm=planarity,
                        normal=normal,
                    )
                )
                if len(rings) >= settings.max_rings:
                    incomplete = True
                    break
            if incomplete and (candidate_count > settings.max_ring_candidates or len(rings) >= settings.max_rings):
                break
        if incomplete and (candidate_count > settings.max_ring_candidates or len(rings) >= settings.max_rings):
            break

    diagnostics = {
        "searched_root_edges": searched_root_edges,
        "candidate_count": candidate_count,
        "accepted_ring_count": len(rings),
        "duplicate_candidate_count": duplicate_count,
        "rejected_non_isometric_count": rejected_isometric,
        "truncated_root_edge_count": truncated_root_edges,
        "skipped_components": skipped_components,
        "distance_cache_entries": len(distance_cache),
        "incomplete": incomplete,
    }
    return rings, diagnostics, incomplete


def ring_edge_incidence(rings: Iterable[RingRecord]) -> Counter[int]:
    incidence: Counter[int] = Counter()
    for ring in rings:
        incidence.update(ring.edge_ids)
    return incidence
