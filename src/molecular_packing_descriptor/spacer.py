"""Finite-component spacer and molecular-shape geometry."""

from __future__ import annotations

import numpy as np

from .model import Component, DescriptorConfig, Structure
from .periodic import _unwrapped_component_positions

def _shape_anisotropy(eigenvalues: np.ndarray) -> float:
    eigenvalues = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
    total = float(np.sum(eigenvalues))
    if total <= 1e-20:
        return 0.0
    pair_sum = float(
        eigenvalues[0] * eigenvalues[1]
        + eigenvalues[1] * eigenvalues[2]
        + eigenvalues[2] * eigenvalues[0]
    )
    value = 1.0 - 3.0 * pair_sum / (total * total)
    return float(np.clip(value, 0.0, 1.0))


def component_shape_anisotropy(structure: Structure, components: list[Component]) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for component in components:
        if component.periodic or len(component.nodes) < 3:
            continue
        unwrapped = _unwrapped_component_positions(structure, component)
        xyz = np.asarray([unwrapped[node] for node in component.nodes], dtype=float)
        centered = xyz - np.mean(xyz, axis=0)
        gyration = centered.T @ centered / len(component.nodes)
        eig = np.linalg.eigvalsh(gyration)
        value = _shape_anisotropy(eig)
        weight = float(len(component.nodes))
        weighted_sum += weight * value
        total_weight += weight
    return weighted_sum / total_weight if total_weight else 0.0


def _simple_component_graph(
    component: Component,
    adjacency: list[list[tuple[int, tuple[int, int, int], np.ndarray, float]]],
) -> dict[int, set[int]]:
    nodes = set(component.nodes)
    graph = {node: set() for node in component.nodes}
    for u in component.nodes:
        for v, _shift, _vector, _distance in adjacency[u]:
            if v in nodes and v != u:
                graph[u].add(v)
    return graph


def _bridge_edges(graph: dict[int, set[int]]) -> set[tuple[int, int]]:
    """Tarjan bridges of a simple undirected graph."""

    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int | None] = {}
    bridges: set[tuple[int, int]] = set()
    time = 0

    def dfs(u: int) -> None:
        nonlocal time
        discovery[u] = time
        low[u] = time
        time += 1
        for v in sorted(graph[u]):
            if v not in discovery:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > discovery[u]:
                    bridges.add(tuple(sorted((u, v))))
            elif parent.get(u) != v:
                low[u] = min(low[u], discovery[v])

    for node in sorted(graph):
        if node not in discovery:
            parent[node] = None
            dfs(node)
    return bridges


def _spacer_paths(
    component: Component,
    adjacency: list[list[tuple[int, tuple[int, int, int], np.ndarray, float]]],
    min_bonds: int,
) -> list[list[int]]:
    if component.periodic or len(component.nodes) < min_bonds + 1:
        return []
    graph = _simple_component_graph(component, adjacency)
    bridges = _bridge_edges(graph)
    all_edges = {tuple(sorted((u, v))) for u, nbrs in graph.items() for v in nbrs if u != v}
    cycle_edges = all_edges - bridges
    ring_nodes = {node for edge in cycle_edges for node in edge}
    anchors = {node for node, nbrs in graph.items() if len(nbrs) != 2 or node in ring_nodes}
    if not anchors:
        return []

    visited: set[tuple[int, int]] = set()
    paths: list[list[int]] = []
    for start in sorted(anchors):
        for first in sorted(graph[start]):
            edge = tuple(sorted((start, first)))
            if edge in visited:
                continue
            path = [start, first]
            visited.add(edge)
            previous, current = start, first
            seen = {start}
            while current not in anchors:
                if current in seen:
                    break
                seen.add(current)
                options = [node for node in graph[current] if node != previous]
                if len(options) != 1:
                    break
                nxt = options[0]
                next_edge = tuple(sorted((current, nxt)))
                if next_edge in visited:
                    break
                visited.add(next_edge)
                path.append(nxt)
                previous, current = current, nxt
            if path[-1] in anchors and path[-1] != path[0] and len(path) - 1 >= min_bonds:
                paths.append(path)
    return paths


def spacer_metrics(
    structure: Structure,
    components: list[Component],
    adjacency: list[list[tuple[int, tuple[int, int, int], np.ndarray, float]]],
    config: DescriptorConfig,
) -> tuple[float, float]:
    weighted_extension = 0.0
    total_contour = 0.0
    spacer_atoms: set[int] = set()
    finite_atoms = sum(len(component.nodes) for component in components if not component.periodic)
    for component in components:
        if component.periodic:
            continue
        positions = _unwrapped_component_positions(structure, component)
        for path in _spacer_paths(component, adjacency, config.min_spacer_bonds):
            contour = 0.0
            for u, v in zip(path[:-1], path[1:], strict=True):
                contour += float(np.linalg.norm(positions[v] - positions[u]))
            if contour <= 1e-12:
                continue
            end_to_end = float(np.linalg.norm(positions[path[-1]] - positions[path[0]]))
            extension = float(np.clip(end_to_end / contour, 0.0, 1.0))
            weighted_extension += contour * extension
            total_contour += contour
            spacer_atoms.update(path[1:-1])
    extension_value = weighted_extension / total_contour if total_contour else 0.0
    fraction_value = len(spacer_atoms) / finite_atoms if finite_atoms else 0.0
    return extension_value, float(np.clip(fraction_value, 0.0, 1.0))

