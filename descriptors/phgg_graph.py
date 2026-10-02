"""Periodic graph and void/contact geometry primitives for PHGG."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterable, Sequence

from .phgg_radii import covalent_radius, surface_radius
from .phgg_types import (
    DescriptorError,
    IntVector3,
    Matrix3,
    Settings,
    StructureData,
    Vector3,
    frac_to_cart,
    norm,
)


@dataclass(frozen=True)
class Edge:
    i: int
    j: int
    shift: IntVector3
    distance_A: float
    cutoff_A: float


@dataclass(frozen=True)
class Component:
    atom_indices: tuple[int, ...]
    translation_rank: int
    offsets: dict[int, IntVector3]
    cycle_translations: tuple[IntVector3, ...]


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.weight = [1] * size

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.weight[a] < self.weight[b]:
            a, b = b, a
        self.parent[b] = a
        self.weight[a] += self.weight[b]


def _ivec_add(a: IntVector3, b: IntVector3) -> IntVector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _ivec_sub(a: IntVector3, b: IntVector3) -> IntVector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _ivec_neg(a: IntVector3) -> IntVector3:
    return (-a[0], -a[1], -a[2])


def _canonical_positive(shift: IntVector3) -> bool:
    for value in shift:
        if value:
            return value > 0
    return False


def _column_norm(matrix: Matrix3, column: int) -> float:
    return math.sqrt(sum(matrix[row][column] ** 2 for row in range(3)))


def _translation_ranges(delta: Vector3, inverse_lattice: Matrix3, cutoff_A: float) -> tuple[range, range, range]:
    ranges: list[range] = []
    for axis in range(3):
        radius = cutoff_A * _column_norm(inverse_lattice, axis)
        low = math.ceil(-delta[axis] - radius - 1.0e-12)
        high = math.floor(-delta[axis] + radius + 1.0e-12)
        ranges.append(range(low, high + 1))
    return ranges[0], ranges[1], ranges[2]


def build_periodic_bond_graph(structure: StructureData, settings: Settings) -> tuple[list[Edge], tuple[str, ...]]:
    """Build a radius-gated periodic multigraph with explicit image shifts."""
    radii: list[float] = []
    fallback_elements: set[str] = set()
    for symbol in structure.species:
        radius, fallback = covalent_radius(symbol)
        radii.append(radius)
        if fallback:
            fallback_elements.add(symbol)
    edges: list[Edge] = []
    for i in range(structure.n_atoms):
        for j in range(i, structure.n_atoms):
            cutoff = settings.bond_scale * (radii[i] + radii[j]) + settings.bond_offset_A
            delta = (
                structure.fractional[j][0] - structure.fractional[i][0],
                structure.fractional[j][1] - structure.fractional[i][1],
                structure.fractional[j][2] - structure.fractional[i][2],
            )
            for shift in itertools.product(*_translation_ranges(delta, structure.inverse_lattice, cutoff)):
                typed_shift: IntVector3 = (int(shift[0]), int(shift[1]), int(shift[2]))
                if i == j and (typed_shift == (0, 0, 0) or not _canonical_positive(typed_shift)):
                    continue
                displaced = (delta[0] + typed_shift[0], delta[1] + typed_shift[1], delta[2] + typed_shift[2])
                distance = norm(frac_to_cart(displaced, structure.lattice))
                if 1.0e-9 < distance <= cutoff + 1.0e-12:
                    edges.append(Edge(i, j, typed_shift, distance, cutoff))
                    if len(edges) > settings.max_edges:
                        raise DescriptorError(f"periodic bond graph exceeds max_edges={settings.max_edges}")
    edges.sort(key=lambda edge: (edge.i, edge.j, edge.shift, edge.distance_A))
    return edges, tuple(sorted(fallback_elements))


def _integer_rank(vectors: Iterable[IntVector3]) -> int:
    unique = [vector for vector in dict.fromkeys(vectors) if vector != (0, 0, 0)]
    if not unique:
        return 0
    first = unique[0]
    independent_two: IntVector3 | None = None
    for vector in unique[1:]:
        cross = (
            first[1] * vector[2] - first[2] * vector[1],
            first[2] * vector[0] - first[0] * vector[2],
            first[0] * vector[1] - first[1] * vector[0],
        )
        if cross != (0, 0, 0):
            independent_two = vector
            break
    if independent_two is None:
        return 1
    for vector in unique:
        determinant = (
            first[0] * (independent_two[1] * vector[2] - independent_two[2] * vector[1])
            - first[1] * (independent_two[0] * vector[2] - independent_two[2] * vector[0])
            + first[2] * (independent_two[0] * vector[1] - independent_two[1] * vector[0])
        )
        if determinant != 0:
            return 3
    return 2


def find_components(structure: StructureData, edges: Sequence[Edge]) -> tuple[Component, ...]:
    union = _UnionFind(structure.n_atoms)
    adjacency: list[list[tuple[int, IntVector3]]] = [[] for _ in range(structure.n_atoms)]
    for edge in edges:
        union.union(edge.i, edge.j)
        adjacency[edge.i].append((edge.j, edge.shift))
        adjacency[edge.j].append((edge.i, _ivec_neg(edge.shift)))
    groups: dict[int, list[int]] = {}
    for atom in range(structure.n_atoms):
        groups.setdefault(union.find(atom), []).append(atom)

    components: list[Component] = []
    for indices in sorted(groups.values(), key=lambda values: min(values)):
        root = min(indices)
        offsets: dict[int, IntVector3] = {root: (0, 0, 0)}
        stack = [root]
        cycles: list[IntVector3] = []
        while stack:
            current = stack.pop()
            for neighbor, shift in adjacency[current]:
                expected = _ivec_add(offsets[current], shift)
                if neighbor not in offsets:
                    offsets[neighbor] = expected
                    stack.append(neighbor)
                else:
                    residual = _ivec_sub(expected, offsets[neighbor])
                    if residual != (0, 0, 0):
                        cycles.append(residual)
        components.append(
            Component(
                atom_indices=tuple(indices),
                translation_rank=_integer_rank(cycles),
                offsets=offsets,
                cycle_translations=tuple(dict.fromkeys(cycles)),
            )
        )
    return tuple(components)


def partition_host_guest(components: Sequence[Component]) -> tuple[tuple[int, ...], tuple[Component, ...], str]:
    """Return host atoms, finite guest components, and selection mode.

    Every periodic component is host, preserving interpenetrated/disconnected
    periodic nets.  If no component is periodic, every component tied for the
    largest atom count is host and only smaller components are guests.  The tie
    rule avoids assigning host/guest roles from atom order when the structure
    itself provides no periodic distinction.
    """
    periodic = [component for component in components if component.translation_rank > 0]
    if periodic:
        host_atoms = tuple(sorted(atom for component in periodic for atom in component.atom_indices))
        guests = tuple(component for component in components if component.translation_rank == 0)
        return host_atoms, guests, "all_periodic_components"
    largest = max(len(component.atom_indices) for component in components)
    host_components = [component for component in components if len(component.atom_indices) == largest]
    host_atoms = tuple(sorted(atom for component in host_components for atom in component.atom_indices))
    guests = tuple(component for component in components if len(component.atom_indices) < largest)
    mode = "largest_component_fallback" if len(host_components) == 1 else "all_largest_components_fallback"
    return host_atoms, guests, mode


def minimum_image_vector(
    frac_from: Vector3,
    frac_to: Vector3,
    lattice: Matrix3,
    inverse_lattice: Matrix3,
) -> tuple[Vector3, float, IntVector3]:
    """Return the exact deterministic closest image under a general 3-D cell.

    A rounded image gives an initial upper bound.  For any better image, each
    fractional component is bounded by that distance times the corresponding
    reciprocal-column norm, so a finite exhaustive search is exact rather than
    relying on a fixed ``[-1, 1]`` image shell.
    """
    delta = (
        frac_to[0] - frac_from[0],
        frac_to[1] - frac_from[1],
        frac_to[2] - frac_from[2],
    )
    initial_shift: IntVector3 = tuple(int(round(-value)) for value in delta)  # type: ignore[assignment]
    initial_frac = tuple(delta[axis] + initial_shift[axis] for axis in range(3))
    best_vector = frac_to_cart(initial_frac, lattice)  # type: ignore[arg-type]
    best_distance = norm(best_vector)
    best_shift = initial_shift
    ranges = _translation_ranges(delta, inverse_lattice, best_distance + 1.0e-12)
    for shift in itertools.product(*ranges):
        typed_shift: IntVector3 = (int(shift[0]), int(shift[1]), int(shift[2]))
        shifted = (delta[0] + typed_shift[0], delta[1] + typed_shift[1], delta[2] + typed_shift[2])
        vector = frac_to_cart(shifted, lattice)
        distance = norm(vector)
        if distance < best_distance - 1.0e-14 or (
            abs(distance - best_distance) <= 1.0e-14 and typed_shift < best_shift
        ):
            best_vector = vector
            best_distance = distance
            best_shift = typed_shift
    if not math.isfinite(best_distance):
        raise DescriptorError("minimum-image search failed")
    return best_vector, best_distance, best_shift


def unwrapped_component_cartesian(structure: StructureData, component: Component) -> tuple[Vector3, ...]:
    coords: list[Vector3] = []
    for atom in component.atom_indices:
        offset = component.offsets.get(atom, (0, 0, 0))
        frac = (
            structure.fractional[atom][0] + offset[0],
            structure.fractional[atom][1] + offset[1],
            structure.fractional[atom][2] + offset[2],
        )
        coords.append(frac_to_cart(frac, structure.lattice))
    return tuple(coords)


def _grid_dimensions(structure: StructureData, settings: Settings) -> tuple[int, int, int]:
    dimensions = [max(1, int(math.ceil(norm(vector) / settings.grid_spacing_A))) for vector in structure.lattice]
    while dimensions[0] * dimensions[1] * dimensions[2] > settings.max_grid_points:
        candidates = sorted(range(3), key=lambda axis: (dimensions[axis], axis), reverse=True)
        for axis in candidates:
            if dimensions[axis] > 1:
                dimensions[axis] -= 1
                break
        else:
            break
    return dimensions[0], dimensions[1], dimensions[2]


def _wrapped_delta(value: float) -> float:
    wrapped = value - math.floor(value)
    if wrapped >= 1.0 - 1.0e-12 or abs(wrapped) <= 1.0e-12:
        return 0.0
    return wrapped


def _canonical_host_anchor(structure: StructureData, host_atoms: Sequence[int]) -> Vector3:
    """Choose a translation-covariant, atom-order-invariant grid phase."""
    best_signature: tuple[tuple[object, ...], ...] | None = None
    best_anchor: Vector3 | None = None
    for candidate in host_atoms:
        anchor = structure.fractional[candidate]
        signature = tuple(
            sorted(
                (
                    structure.species[atom],
                    round(_wrapped_delta(structure.fractional[atom][0] - anchor[0]), 12),
                    round(_wrapped_delta(structure.fractional[atom][1] - anchor[1]), 12),
                    round(_wrapped_delta(structure.fractional[atom][2] - anchor[2]), 12),
                )
                for atom in host_atoms
            )
        )
        if best_signature is None or signature < best_signature:
            best_signature = signature
            best_anchor = anchor
    if best_anchor is None:
        raise DescriptorError("cannot anchor an empty host")
    return best_anchor


def host_grid_clearance(
    structure: StructureData,
    host_atoms: Sequence[int],
    settings: Settings,
    surface_radii: Sequence[float],
) -> tuple[float, float, float, tuple[int, int, int], Vector3]:
    dims = _grid_dimensions(structure, settings)
    maximum = -math.inf
    positive_sum = 0.0
    positive_count = 0
    accessible_count = 0
    total = dims[0] * dims[1] * dims[2]
    anchor = _canonical_host_anchor(structure, host_atoms)
    for ix in range(dims[0]):
        for iy in range(dims[1]):
            for iz in range(dims[2]):
                point = (
                    (anchor[0] + (ix + 0.5) / dims[0]) % 1.0,
                    (anchor[1] + (iy + 0.5) / dims[1]) % 1.0,
                    (anchor[2] + (iz + 0.5) / dims[2]) % 1.0,
                )
                clearance = math.inf
                for atom in host_atoms:
                    _, distance, _ = minimum_image_vector(
                        point, structure.fractional[atom], structure.lattice, structure.inverse_lattice
                    )
                    clearance = min(clearance, distance - surface_radii[atom])
                maximum = max(maximum, clearance)
                if clearance > 0.0:
                    positive_sum += clearance
                    positive_count += 1
                if clearance > settings.probe_radius_A:
                    accessible_count += 1
    max_diameter = 2.0 * max(0.0, maximum)
    void_fraction = accessible_count / total if total else 0.0
    mean_positive = positive_sum / positive_count if positive_count else 0.0
    return max_diameter, void_fraction, mean_positive, dims, anchor


def contact_directionality_anisotropy(direction_groups: Sequence[Sequence[Vector3]]) -> float:
    """Return a 0--1 anisotropy of guest-normalized nearest-contact directions.

    Every guest atom contributes one equally weighted second-moment tensor.  If
    several host images tie for closest normalized surface contact, their unit
    dyads are averaged before the guest is combined with other guest atoms.
    """
    group_tensors: list[list[list[float]]] = []
    for group in direction_groups:
        unit_vectors: list[Vector3] = []
        for vector in group:
            length = norm(vector)
            if length > 1.0e-12:
                unit_vectors.append((vector[0] / length, vector[1] / length, vector[2] / length))
        if not unit_vectors:
            continue
        tensor = [[0.0] * 3 for _ in range(3)]
        for vector in unit_vectors:
            for row in range(3):
                for col in range(3):
                    tensor[row][col] += vector[row] * vector[col] / len(unit_vectors)
        group_tensors.append(tensor)
    if not group_tensors:
        return 0.0
    tensor = [[0.0] * 3 for _ in range(3)]
    for group in group_tensors:
        for row in range(3):
            for col in range(3):
                tensor[row][col] += group[row][col] / len(group_tensors)
    squared = 0.0
    for row in range(3):
        for col in range(3):
            value = tensor[row][col] - (1.0 / 3.0 if row == col else 0.0)
            squared += value * value
    return min(1.0, math.sqrt(1.5 * squared))


def build_surface_radii(structure: StructureData) -> tuple[list[float], tuple[str, ...]]:
    values: list[float] = []
    fallback: set[str] = set()
    for symbol in structure.species:
        radius, used_fallback = surface_radius(symbol)
        values.append(radius)
        if used_fallback:
            fallback.add(symbol)
    return values, tuple(sorted(fallback))
