"""Periodic pair enumeration, radius graph construction, and component unwrapping."""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Iterator

import numpy as np

from .elements import ATOMIC_NUMBERS, COVALENT_RADII_ANGSTROM, VDW_RADII_ANGSTROM
from .model import BondEdge, Component, DescriptorConfig, DescriptorError, Structure

def _tuple3(values: Iterable[int]) -> tuple[int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 3:
        raise AssertionError("expected three integers")
    return result  # type: ignore[return-value]


def _add_shift(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub_shift(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _neg_shift(a: tuple[int, int, int]) -> tuple[int, int, int]:
    return (-a[0], -a[1], -a[2])


def _is_zero_shift(a: tuple[int, int, int]) -> bool:
    return a == (0, 0, 0)


def _canonical_pair(i: int, j: int, shift: tuple[int, int, int]) -> bool:
    """Choose one member of each periodic undirected-pair equivalence class."""

    if _is_zero_shift(shift):
        return i < j
    for value in shift:
        if value:
            return value > 0
    return False


def _image_limits(structure: Structure, cutoff: float, config: DescriptorConfig) -> tuple[int, int, int]:
    try:
        inverse = np.linalg.inv(structure.lattice)
    except np.linalg.LinAlgError as exc:  # parser already checks, defensive here
        raise DescriptorError("lattice inversion failed") from exc
    limits = []
    for axis in range(3):
        reciprocal_height_inverse = float(np.linalg.norm(inverse[:, axis]))
        limit = int(math.floor(cutoff * reciprocal_height_inverse)) + 1
        if limit > config.max_image_repetitions:
            raise DescriptorError(
                "cell is too thin/skewed for the fixed periodic image budget: "
                f"axis {axis} needs {limit}, limit is {config.max_image_repetitions}"
            )
        limits.append(max(1, limit))
    return _tuple3(limits)


def _iter_candidate_pairs(
    structure: Structure,
    cutoff: float,
    config: DescriptorConfig,
) -> Iterator[tuple[int, int, tuple[int, int, int], np.ndarray, float]]:
    """Yield canonical atom-image pairs no farther than ``cutoff``."""

    limits = _image_limits(structure, cutoff, config)
    positions = structure.cart_coords
    n_atoms = structure.n_atoms
    cell = structure.lattice
    checked = 0
    for sx in range(-limits[0], limits[0] + 1):
        for sy in range(-limits[1], limits[1] + 1):
            for sz in range(-limits[2], limits[2] + 1):
                shift = (sx, sy, sz)
                translation = np.asarray(shift, dtype=float) @ cell
                for i in range(n_atoms):
                    for j in range(n_atoms):
                        if not _canonical_pair(i, j, shift):
                            continue
                        checked += 1
                        if checked > config.max_unique_pairs:
                            raise DescriptorError(
                                "periodic pair budget exceeded; reduce cell size/pathology or "
                                "increase DescriptorConfig.max_unique_pairs explicitly"
                            )
                        vector = positions[j] + translation - positions[i]
                        distance = float(np.linalg.norm(vector))
                        if 1e-8 < distance <= cutoff + 1e-12:
                            yield i, j, shift, vector, distance


def _radii(structure: Structure) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numbers = np.asarray([ATOMIC_NUMBERS[symbol] for symbol in structure.species], dtype=int)
    covalent = np.asarray([COVALENT_RADII_ANGSTROM[z] for z in numbers], dtype=float)
    vdw = np.asarray([VDW_RADII_ANGSTROM[z] for z in numbers], dtype=float)
    if np.any(covalent <= 0.0) or np.any(vdw <= 0.0):
        raise DescriptorError("radius lookup contains a non-positive value")
    return numbers, covalent, vdw


def build_bonds(structure: Structure, config: DescriptorConfig) -> tuple[list[BondEdge], list[list[tuple[int, tuple[int, int, int], np.ndarray, float]]]]:
    """Construct a periodic covalent-radius graph."""

    config.validate()
    _, covalent, _ = _radii(structure)
    max_cutoff = float(2.0 * config.bond_scale * np.max(covalent))
    edges: list[BondEdge] = []
    adjacency: list[list[tuple[int, tuple[int, int, int], np.ndarray, float]]] = [
        [] for _ in range(structure.n_atoms)
    ]
    for i, j, shift, vector, distance in _iter_candidate_pairs(structure, max_cutoff, config):
        radius_sum = float(covalent[i] + covalent[j])
        threshold = config.bond_scale * radius_sum
        if distance > threshold + 1e-12:
            continue
        normalized = distance / radius_sum
        edge = BondEdge(i, j, shift, vector.copy(), distance, normalized)
        edges.append(edge)
        adjacency[i].append((j, shift, vector.copy(), distance))
        adjacency[j].append((i, _neg_shift(shift), -vector.copy(), distance))
    return edges, adjacency


def find_components(
    structure: Structure,
    adjacency: list[list[tuple[int, tuple[int, int, int], np.ndarray, float]]],
) -> tuple[list[Component], list[int]]:
    """Find graph components and detect translationally periodic bond cycles."""

    assigned = [-1] * structure.n_atoms
    components: list[Component] = []
    for root in range(structure.n_atoms):
        if assigned[root] != -1:
            continue
        component_index = len(components)
        offsets: dict[int, tuple[int, int, int]] = {root: (0, 0, 0)}
        nodes: list[int] = []
        periodic = False
        queue: deque[int] = deque([root])
        assigned[root] = component_index
        while queue:
            u = queue.popleft()
            nodes.append(u)
            for v, shift, _vector, _distance in adjacency[u]:
                expected = _add_shift(offsets[u], shift)
                if v not in offsets:
                    offsets[v] = expected
                elif offsets[v] != expected:
                    periodic = True
                if assigned[v] == -1:
                    assigned[v] = component_index
                    queue.append(v)
                elif assigned[v] != component_index:
                    raise DescriptorError("internal component-label inconsistency")
        components.append(
            Component(
                index=component_index,
                nodes=tuple(sorted(nodes)),
                offsets=offsets,
                periodic=periodic,
            )
        )
    return components, assigned


def _unwrapped_component_positions(structure: Structure, component: Component) -> dict[int, np.ndarray]:
    positions = structure.cart_coords
    return {
        node: positions[node] + np.asarray(component.offsets[node], dtype=float) @ structure.lattice
        for node in component.nodes
    }

