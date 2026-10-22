"""Internal periodic bond graph and component/layer geometry."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import math
from typing import Iterable, Mapping, Sequence

from _layer_bridge_model import (
    DEFAULT_BOND_SCALE,
    Edge,
    ElementData,
    Structure,
    _add_int,
    _cross,
    _determinant,
    _frac_vector_to_cart,
    _inverse3,
    _is_finite_real,
    _norm,
    _sub_int,
    default_element_table,
)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _image_limits(
    lattice: Sequence[Sequence[float]], cutoff: float
) -> tuple[int, int, int]:
    volume = abs(_determinant(lattice))
    heights = []
    for index in range(3):
        other = [lattice[row] for row in range(3) if row != index]
        area = _norm(_cross(other[0], other[1]))
        if area <= 1e-15:
            raise ValueError("invalid lattice plane area")
        heights.append(volume / area)
    return tuple(
        max(1, int(math.ceil(cutoff / height)) + 1) for height in heights
    )


def build_periodic_bonds(
    structure: Structure,
    *,
    element_table: Mapping[str, ElementData] | None = None,
    bond_scale: float = DEFAULT_BOND_SCALE,
    minimum_distance: float = 0.35,
) -> list[Edge]:
    """Infer periodic bonds using a scaled covalent-radius sum cutoff."""
    if not _is_finite_real(bond_scale) or float(bond_scale) <= 0:
        raise ValueError("bond_scale must be a finite positive number")
    if not _is_finite_real(minimum_distance) or float(minimum_distance) < 0:
        raise ValueError("minimum_distance must be a finite nonnegative number")
    table = dict(element_table or default_element_table())
    missing = sorted({symbol for symbol in structure.species if symbol not in table})
    if missing:
        raise ValueError("missing covalent-radius data for: " + ", ".join(missing))
    maximum_cutoff = max(
        float(bond_scale)
        * (table[left].covalent_radius + table[right].covalent_radius)
        for left in structure.species
        for right in structure.species
    )
    limits = _image_limits(structure.lattice, maximum_cutoff)
    edges: list[Edge] = []
    atom_count = len(structure.species)
    for i in range(atom_count):
        for j in range(i, atom_count):
            cutoff = float(bond_scale) * (
                table[structure.species[i]].covalent_radius
                + table[structure.species[j]].covalent_radius
            )
            for sx in range(-limits[0], limits[0] + 1):
                for sy in range(-limits[1], limits[1] + 1):
                    for sz in range(-limits[2], limits[2] + 1):
                        shift = (sx, sy, sz)
                        if i == j:
                            if shift == (0, 0, 0):
                                continue
                            # Keep one orientation for each self-image bond.
                            first_nonzero = next(value for value in shift if value != 0)
                            if first_nonzero < 0:
                                continue
                        delta = tuple(
                            structure.frac_coords[j][axis]
                            + shift[axis]
                            - structure.frac_coords[i][axis]
                            for axis in range(3)
                        )
                        vector = _frac_vector_to_cart(delta, structure.lattice)
                        distance = _norm(vector)
                        if minimum_distance < distance <= cutoff + 1e-12:
                            edges.append(Edge(i, j, shift, distance, vector))
    edges.sort(
        key=lambda edge: (edge.i, edge.j, edge.shift, round(edge.distance, 12))
    )
    return edges


def _adjacency(atom_count: int, edges: Sequence[Edge]):
    graph: list[
        list[tuple[int, tuple[int, int, int], float, tuple[float, float, float]]]
    ] = [[] for _ in range(atom_count)]
    for edge in edges:
        graph[edge.i].append((edge.j, edge.shift, edge.distance, edge.vector))
        inverse_shift = tuple(-value for value in edge.shift)
        inverse_vector = tuple(-value for value in edge.vector)
        graph[edge.j].append(
            (edge.i, inverse_shift, edge.distance, inverse_vector)
        )
    for neighbors in graph:
        neighbors.sort(key=lambda item: (item[0], item[1], round(item[2], 12)))
    return graph


def _rank_and_basis(
    vectors: Iterable[Sequence[int]],
) -> tuple[int, list[tuple[int, int, int]]]:
    basis: list[tuple[int, int, int]] = []
    rank = 0
    for vector in vectors:
        candidate = tuple(int(value) for value in vector)
        if candidate == (0, 0, 0):
            continue
        trial = basis + [candidate]
        new_rank = _matrix_rank(trial)
        if new_rank > rank:
            basis.append(candidate)
            rank = new_rank
            if rank == 3:
                break
    return rank, basis


def _matrix_rank(
    rows: Sequence[Sequence[float]], tolerance: float = 1e-10
) -> int:
    matrix = [
        list(map(float, row))
        for row in rows
        if any(abs(float(value)) > tolerance for value in row)
    ]
    if not matrix:
        return 0
    row = 0
    columns = len(matrix[0])
    for column in range(columns):
        pivot = max(
            range(row, len(matrix)),
            key=lambda index: abs(matrix[index][column]),
            default=row,
        )
        if pivot >= len(matrix) or abs(matrix[pivot][column]) <= tolerance:
            continue
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
        divisor = matrix[row][column]
        matrix[row] = [value / divisor for value in matrix[row]]
        for other in range(len(matrix)):
            if other == row:
                continue
            factor = matrix[other][column]
            if abs(factor) > tolerance:
                matrix[other] = [
                    matrix[other][index] - factor * matrix[row][index]
                    for index in range(columns)
                ]
        row += 1
        if row == len(matrix):
            break
    return row


def component_dimensionalities(
    structure: Structure, edges: Sequence[Edge]
) -> list[dict[str, object]]:
    """Return quotient-graph components and translation-rank dimensionality."""
    atom_count = len(structure.species)
    graph = _adjacency(atom_count, edges)
    union = _UnionFind(atom_count)
    for edge in edges:
        union.union(edge.i, edge.j)
    groups: dict[int, list[int]] = defaultdict(list)
    for site in range(atom_count):
        groups[union.find(site)].append(site)

    components = []
    for sites in sorted(
        groups.values(), key=lambda values: (min(values), len(values))
    ):
        allowed = set(sites)
        root = min(sites)
        offsets: dict[int, tuple[int, int, int]] = {root: (0, 0, 0)}
        queue = deque([root])
        cycles: list[tuple[int, int, int]] = []
        while queue:
            current = queue.popleft()
            for neighbor, shift, _distance, _vector in graph[current]:
                if neighbor not in allowed:
                    continue
                proposed = _add_int(offsets[current], shift)
                if neighbor not in offsets:
                    offsets[neighbor] = proposed
                    queue.append(neighbor)
                else:
                    cycle = _sub_int(proposed, offsets[neighbor])
                    if cycle != (0, 0, 0):
                        cycles.append(cycle)
        rank, basis = _rank_and_basis(cycles)
        record: dict[str, object] = {
            "site_indices": sites,
            "atom_count": len(sites),
            "dimensionality": rank,
            "translation_basis": basis,
        }
        if rank == 2:
            hkl = _primitive_normal(basis)
            if hkl is not None:
                record["layer_hkl"] = hkl
        components.append(record)
    return components


def _primitive_normal(
    basis: Sequence[Sequence[int]],
) -> tuple[int, int, int] | None:
    for left_index in range(len(basis)):
        for right_index in range(left_index + 1, len(basis)):
            cross = _cross(basis[left_index], basis[right_index])
            integer = tuple(int(round(value)) for value in cross)
            if integer != (0, 0, 0):
                divisor = (
                    math.gcd(
                        math.gcd(abs(integer[0]), abs(integer[1])), abs(integer[2])
                    )
                    or 1
                )
                reduced = tuple(value // divisor for value in integer)
                first_nonzero = next(value for value in reduced if value != 0)
                if first_nonzero < 0:
                    reduced = tuple(-value for value in reduced)
                return reduced
    return None


def _largest_circular_clearance(
    structure: Structure,
    hkl: Sequence[int],
    table: Mapping[str, ElementData],
) -> tuple[float, float]:
    inverse = _inverse3(structure.lattice)
    reciprocal_normal = tuple(
        sum(inverse[row][column] * hkl[column] for column in range(3))
        for row in range(3)
    )
    reciprocal_length = _norm(reciprocal_normal)
    if reciprocal_length <= 1e-15:
        return 0.0, 0.0
    repeat = 1.0 / reciprocal_length
    intervals: list[tuple[float, float]] = []
    for symbol, frac in zip(structure.species, structure.frac_coords):
        center = (
            sum(hkl[axis] * frac[axis] for axis in range(3)) % 1.0
        ) * repeat
        radius = table[symbol].covalent_radius
        if 2.0 * radius >= repeat:
            return repeat, 0.0
        for image in (-1, 0, 1):
            start = center - radius + image * repeat
            end = center + radius + image * repeat
            clipped_start = max(0.0, start)
            clipped_end = min(repeat, end)
            if clipped_start < clipped_end:
                intervals.append((clipped_start, clipped_end))
    if not intervals:
        return repeat, repeat
    intervals.sort()
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 1e-12:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    # The interval is circular: combine the tail and head at the boundary.
    largest = merged[0][0] + repeat - merged[-1][1]
    for left, right in zip(merged, merged[1:]):
        largest = max(largest, right[0] - left[1])
    return repeat, max(0.0, largest)


def _component_summary(
    structure: Structure,
    components: Sequence[Mapping[str, object]],
    table: Mapping[str, ElementData],
) -> tuple[dict[str, float], dict[str, object]]:
    total = len(structure.species)
    counts = Counter()
    for component in components:
        counts[int(component["dimensionality"])] += int(component["atom_count"])
    maximum = max(counts) if counts else 0
    mean_dimension = (
        sum(dimension * count for dimension, count in counts.items()) / total
    )
    fractions = {dimension: counts[dimension] / total for dimension in range(4)}
    entropy = max(
        0.0,
        -sum(value * math.log(value) for value in fractions.values() if value > 0)
        / math.log(4.0),
    )

    layered_components = [
        component
        for component in components
        if int(component["dimensionality"]) == 2
    ]
    repeat = 0.0
    clearance = 0.0
    chosen_hkl = None
    if layered_components:
        chosen = sorted(
            (
                component
                for component in layered_components
                if "layer_hkl" in component
            ),
            key=lambda component: (
                -int(component["atom_count"]),
                tuple(component["layer_hkl"]),
            ),
        )
        if chosen:
            chosen_hkl = tuple(chosen[0]["layer_hkl"])
            repeat, clearance = _largest_circular_clearance(
                structure, chosen_hkl, table
            )
    descriptors = {
        "framework_dimensionality_max": float(maximum),
        "framework_dimensionality_mean": float(mean_dimension),
        "layered_atom_fraction": float(fractions[2]),
        "mixed_dimensionality_entropy": float(entropy),
        "layer_repeat_angstrom": float(repeat),
        "interlayer_clearance_angstrom": float(clearance),
        "interlayer_clearance_fraction": float(
            clearance / repeat if repeat > 0 else 0.0
        ),
    }
    orientations = {
        tuple(component.get("layer_hkl", ()))
        for component in layered_components
        if component.get("layer_hkl")
    }
    diagnostics = {
        "component_count": len(components),
        "component_atom_counts_by_dimensionality": {
            str(key): counts[key] for key in range(4)
        },
        "dominant_layer_hkl": (
            list(chosen_hkl) if chosen_hkl is not None else None
        ),
        "layer_orientation_count": len(orientations),
    }
    return descriptors, diagnostics
