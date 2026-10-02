"""Fixed, interpretable descriptors for periodic framework/polyhedron/void geometry.

The implementation consumes only a fully occupied periodic structure and static
per-element lookup values.  It deliberately does not infer synthesis history,
macroscopic monolith porosity, defects, catalytic activity, or durability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import acos, degrees
from typing import Iterable, Mapping, Sequence

import numpy as np

from .elements import ElementRecord, get_element
from .poscar import PeriodicStructure

_EPS = 1.0e-12
_IMAGE_OFFSETS_27 = np.asarray(list(product((-1, 0, 1), repeat=3)), dtype=int)


@dataclass(frozen=True)
class DescriptorConfig:
    """Numerical settings with conservative, deterministic defaults."""

    bond_scale: float = 1.25
    min_bond_distance: float = 0.25
    max_image_range: int = 4
    max_edges: int = 250_000
    grid_spacing: float = 0.80
    max_grid_points: int = 32_768
    grid_chunk_size: int = 512
    probe_radii: tuple[float, ...] = (0.0, 0.5, 1.0, 1.2, 1.4, 1.8)

    def __post_init__(self) -> None:
        if not 0.8 <= self.bond_scale <= 2.0:
            raise ValueError("bond_scale must be between 0.8 and 2.0")
        if self.min_bond_distance <= 0.0:
            raise ValueError("min_bond_distance must be positive")
        if self.max_image_range < 1:
            raise ValueError("max_image_range must be positive")
        if self.max_edges < 1:
            raise ValueError("max_edges must be positive")
        if self.grid_spacing <= 0.0:
            raise ValueError("grid_spacing must be positive")
        if self.max_grid_points < 27:
            raise ValueError("max_grid_points must be at least 27")
        if self.grid_chunk_size < 1:
            raise ValueError("grid_chunk_size must be positive")
        if tuple(sorted(self.probe_radii)) != self.probe_radii or any(x < 0 for x in self.probe_radii):
            raise ValueError("probe_radii must be sorted and non-negative")


@dataclass(frozen=True)
class BondEdge:
    """One quotient-graph edge from site ``i`` to image ``j + shift``."""

    i: int
    j: int
    shift: tuple[int, int, int]
    distance: float
    normalized_distance: float


@dataclass(frozen=True)
class DescriptorResult:
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if len(self.feature_names) != len(self.values):
            raise ValueError("feature_names and values differ in length")
        arr = np.asarray(self.values, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("descriptor contains non-finite values")

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.values))

    def to_jsonable(self) -> dict[str, object]:
        return {
            "schema": "phosphate-framework-pore-descriptor/1",
            "feature_names": list(self.feature_names),
            "values": list(self.values),
            "features": self.as_dict(),
            "metadata": dict(self.metadata),
        }


class _FeatureBuilder:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.values: list[float] = []

    def add(self, name: str, value: float | int | np.floating) -> None:
        number = float(value)
        if not np.isfinite(number):
            number = 0.0
        self.names.append(name)
        self.values.append(number)

    def add_stats(self, prefix: str, values: Sequence[float], *, quantiles: Sequence[float] = ()) -> None:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            self.add(prefix + "_mean", float(np.mean(arr)))
            self.add(prefix + "_std", float(np.std(arr)))
            for q in quantiles:
                self.add(prefix + f"_q{int(round(100 * q)):02d}", float(np.quantile(arr, q, method="nearest")))
        else:
            self.add(prefix + "_mean", 0.0)
            self.add(prefix + "_std", 0.0)
            for q in quantiles:
                self.add(prefix + f"_q{int(round(100 * q)):02d}", 0.0)


@dataclass
class _GraphSummary:
    degrees: np.ndarray
    largest_component_fraction: float
    component_density_per_node: float
    cycle_rank_per_node: float
    periodicity_rank: int
    axis_span_fraction: float


@dataclass
class _PolyhedronSummary:
    poly_centers: set[int]
    low_centers: set[int]
    high_centers: set[int]
    cn4_centers: set[int]
    cn6_centers: set[int]
    radial_cv: list[float]
    normalized_strain_rms: list[float]
    directional_anisotropy: list[float]
    tetra_q: list[float]
    tetra_angle_rmse: list[float]
    octa_cosine_rmse: list[float]
    octa_angle_rmse: list[float]


def featurize_structure(
    structure: PeriodicStructure,
    config: DescriptorConfig | None = None,
) -> DescriptorResult:
    """Compute the fixed descriptor vector.

    All features are static geometrical or lookup-derived scalars.  Grid pore
    values describe the atomistic periodic unit cell and are not measurements of
    meso/macroporosity in a processed monolith.
    """

    cfg = config or DescriptorConfig()
    records = [get_element(symbol) for symbol in structure.species]
    edges, neighbors = _build_bond_graph(structure, records, cfg)
    graph = _summarize_graph(structure.n_sites, edges)
    poly = _summarize_polyhedra(structure, records, neighbors)
    bridge = _summarize_bridges(structure.n_sites, records, neighbors, poly)
    grid = _summarize_void_grid(structure, records, cfg)
    grid_shape = tuple(grid.pop("_grid_shape"))
    grid_points = int(grid.pop("_grid_points"))

    fb = _FeatureBuilder()
    n = structure.n_sites
    z = np.asarray([x.atomic_number for x in records], dtype=float)
    cov = np.asarray([x.covalent_radius for x in records], dtype=float)
    vdw = np.asarray([x.vdw_radius for x in records], dtype=float)
    outer = np.asarray([x.outer_electrons for x in records], dtype=float)

    # Composition and cell-density block.  Values are invariant to integer supercells.
    fb.add("structure_valid", 1.0)
    fb.add("volume_per_atom_ang3", structure.volume / n)
    fb.add("number_density_per_ang3", n / structure.volume)
    fb.add("n_unique_elements", len(set(structure.species)))
    fb.add("metal_fraction", np.mean([x.is_metal for x in records]))
    fb.add("f_block_fraction", np.mean([x.is_f_block for x in records]))
    fb.add_stats("atomic_number", z)
    fb.add_stats("covalent_radius_ang", cov)
    fb.add_stats("vdw_radius_ang", vdw)
    fb.add_stats("outer_electrons", outer)

    # Radius-graph block.
    fb.add_stats("coordination", graph.degrees, quantiles=(0.10, 0.50, 0.90))
    fb.add("coordination_le2_fraction", np.mean(graph.degrees <= 2))
    fb.add("coordination_ge6_fraction", np.mean(graph.degrees >= 6))
    fb.add("bond_edges_per_atom", len(edges) / n)
    fb.add_stats("bond_length_ang", [e.distance for e in edges])
    fb.add_stats("bond_length_normalized", [e.normalized_distance for e in edges])
    fb.add(
        "heteroelement_bond_fraction",
        np.mean([structure.species[e.i] != structure.species[e.j] for e in edges]) if edges else 0.0,
    )
    fb.add("bond_graph_component_dominance", graph.largest_component_fraction)
    fb.add("bond_graph_component_density_per_atom", graph.component_density_per_node)
    fb.add("bond_graph_cycle_rank_per_atom", graph.cycle_rank_per_node)
    fb.add("bond_graph_periodicity_rank", graph.periodicity_rank)
    fb.add("bond_graph_axis_span_fraction", graph.axis_span_fraction)

    # Local coordination-polyhedron block.
    fb.add("polyhedron_center_fraction", len(poly.poly_centers) / n)
    fb.add("low_cn_center_fraction", len(poly.low_centers) / n)
    fb.add("high_cn_center_fraction", len(poly.high_centers) / n)
    fb.add("cn4_center_fraction", len(poly.cn4_centers) / n)
    fb.add("cn6_center_fraction", len(poly.cn6_centers) / n)
    fb.add_stats("polyhedron_radial_cv", poly.radial_cv)
    fb.add_stats("polyhedron_normalized_strain_rms", poly.normalized_strain_rms)
    fb.add_stats("polyhedron_directional_anisotropy", poly.directional_anisotropy)
    fb.add_stats("tetrahedral_q", poly.tetra_q)
    fb.add_stats("tetrahedral_angle_rmse_deg", poly.tetra_angle_rmse)
    fb.add_stats("octahedral_cosine_rmse", poly.octa_cosine_rmse)
    fb.add_stats("octahedral_angle_rmse_deg", poly.octa_angle_rmse)
    _add_role_element_features(fb, records, poly.low_centers, poly.high_centers)

    # Ligand-mediated framework connectivity block.
    for name, value in bridge.items():
        fb.add(name, value)

    # Periodic atomistic-void block.
    for name, value in grid.items():
        fb.add(name, value)

    metadata = {
        "descriptor": "periodic_framework_polyhedron_void_v1",
        "paper_inspiration": "10.3390/molecules30020331",
        "n_sites": n,
        "n_edges": len(edges),
        "grid_shape": list(grid_shape),
        "grid_points": grid_points,
        "config": asdict(cfg),
        "scope_note": (
            "Static fully occupied periodic-cell geometry only; no inference of macroscopic monolith "
            "porosity, oxygen-vacancy concentration, synthesis/reduction history, catalytic activity, or durability."
        ),
    }
    return DescriptorResult(tuple(fb.names), tuple(fb.values), metadata)


def _build_bond_graph(
    structure: PeriodicStructure,
    records: Sequence[ElementRecord],
    cfg: DescriptorConfig,
) -> tuple[list[BondEdge], list[list[tuple[int, np.ndarray, float, float]]]]:
    lattice = structure.lattice
    inv = np.linalg.inv(lattice)
    plane_spacings = np.asarray([1.0 / max(np.linalg.norm(inv[:, i]), _EPS) for i in range(3)])
    max_radius = max(x.covalent_radius for x in records)
    max_cutoff = 2.0 * max_radius * cfg.bond_scale
    ranges = np.minimum(
        cfg.max_image_range,
        np.maximum(1, np.ceil(max_cutoff / np.maximum(plane_spacings, _EPS)).astype(int)),
    )
    shifts = [np.asarray(x, dtype=int) for x in product(*(range(-int(r), int(r) + 1) for r in ranges))]

    edges: list[BondEdge] = []
    neighbors: list[list[tuple[int, np.ndarray, float, float]]] = [[] for _ in range(structure.n_sites)]
    frac = structure.frac_coords
    for i in range(structure.n_sites):
        for j in range(i, structure.n_sites):
            cutoff = cfg.bond_scale * (records[i].covalent_radius + records[j].covalent_radius)
            if cutoff <= cfg.min_bond_distance:
                continue
            for shift in shifts:
                if i == j:
                    if np.all(shift == 0) or not _canonical_positive_shift(shift):
                        continue
                delta = frac[j] + shift - frac[i]
                distance = float(np.linalg.norm(delta @ lattice))
                if distance < cfg.min_bond_distance or distance > cutoff + 1e-10:
                    continue
                normalized = distance / max(records[i].covalent_radius + records[j].covalent_radius, _EPS)
                edge = BondEdge(i, j, tuple(int(x) for x in shift), distance, normalized)
                edges.append(edge)
                neighbors[i].append((j, shift.copy(), distance, normalized))
                neighbors[j].append((i, -shift.copy(), distance, normalized))
                if len(edges) > cfg.max_edges:
                    raise ValueError("bond graph exceeds max_edges; check cell or bond_scale")
    # Stable sorting makes atom-order permutations converge to the same aggregate arithmetic order.
    for row in neighbors:
        row.sort(key=lambda item: (round(item[2], 12), item[0], tuple(int(x) for x in item[1])))
    return edges, neighbors


def _canonical_positive_shift(shift: np.ndarray) -> bool:
    for value in shift:
        if value:
            return bool(value > 0)
    return False


def _summarize_graph(n_nodes: int, edges: Sequence[BondEdge | tuple[int, int, Sequence[int]]]) -> _GraphSummary:
    degrees = np.zeros(n_nodes, dtype=int)
    adjacency: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(n_nodes)]
    normalized_edges: list[tuple[int, int, np.ndarray]] = []
    for edge in edges:
        if isinstance(edge, BondEdge):
            i, j, shift = edge.i, edge.j, np.asarray(edge.shift, dtype=int)
        else:
            i, j, raw = edge
            shift = np.asarray(raw, dtype=int)
        normalized_edges.append((int(i), int(j), shift))
        degrees[int(i)] += 1
        degrees[int(j)] += 1
        adjacency[int(i)].append((int(j), shift))
        adjacency[int(j)].append((int(i), -shift))

    visited = np.zeros(n_nodes, dtype=bool)
    potentials = np.zeros((n_nodes, 3), dtype=int)
    components: list[list[int]] = []
    cycle_vectors: list[np.ndarray] = []
    for start in range(n_nodes):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        component: list[int] = []
        while stack:
            u = stack.pop()
            component.append(u)
            for v, shift in adjacency[u]:
                proposal = potentials[u] + shift
                if not visited[v]:
                    visited[v] = True
                    potentials[v] = proposal
                    stack.append(v)
                else:
                    loop = proposal - potentials[v]
                    if np.any(loop):
                        cycle_vectors.append(loop.astype(float))
        components.append(component)

    periodicity_rank = _vector_rank(cycle_vectors)
    axis_flags = np.zeros(3, dtype=bool)
    for vector in cycle_vectors:
        axis_flags |= np.asarray(vector) != 0
    n_components = len(components)
    largest = max((len(c) for c in components), default=0)
    cycle_rank = max(0, len(normalized_edges) - n_nodes + n_components)
    return _GraphSummary(
        degrees=degrees,
        largest_component_fraction=(largest * n_components) / max(n_nodes, 1),
        component_density_per_node=n_components / max(n_nodes, 1),
        cycle_rank_per_node=cycle_rank / max(n_nodes, 1),
        periodicity_rank=periodicity_rank,
        axis_span_fraction=float(np.mean(axis_flags)),
    )


def _vector_rank(vectors: Sequence[np.ndarray]) -> int:
    if not vectors:
        return 0
    matrix = np.asarray(vectors, dtype=float).reshape((-1, 3))
    return int(np.linalg.matrix_rank(matrix, tol=1e-9))


def _summarize_polyhedra(
    structure: PeriodicStructure,
    records: Sequence[ElementRecord],
    neighbors: Sequence[Sequence[tuple[int, np.ndarray, float, float]]],
) -> _PolyhedronSummary:
    poly_centers: set[int] = set()
    low_centers: set[int] = set()
    high_centers: set[int] = set()
    cn4: set[int] = set()
    cn6: set[int] = set()
    radial_cv: list[float] = []
    normalized_strain_rms: list[float] = []
    directional_anisotropy: list[float] = []
    tetra_q: list[float] = []
    tetra_angle_rmse: list[float] = []
    octa_cosine_rmse: list[float] = []
    octa_angle_rmse: list[float] = []

    for center, row in enumerate(neighbors):
        cn = len(row)
        if cn < 3 or cn > 12:
            continue
        center_radius = records[center].covalent_radius
        ligand_like = [
            records[j].ligand_priority or records[j].covalent_radius <= 0.92 * center_radius
            for j, _shift, _distance, _normalized in row
        ]
        ligand_fraction = float(np.mean(ligand_like)) if ligand_like else 0.0
        if ligand_fraction < 0.60:
            continue
        poly_centers.add(center)
        if 3 <= cn <= 5:
            low_centers.add(center)
        if cn >= 6:
            high_centers.add(center)
        if cn == 4:
            cn4.add(center)
        if cn == 6:
            cn6.add(center)

        distances = np.asarray([x[2] for x in row], dtype=float)
        normalized = np.asarray([x[3] for x in row], dtype=float)
        radial_cv.append(float(np.std(distances) / max(np.mean(distances), _EPS)))
        normalized_strain_rms.append(float(np.sqrt(np.mean((normalized - 1.0) ** 2))))
        vectors = np.asarray(
            [(structure.frac_coords[j] + shift - structure.frac_coords[center]) @ structure.lattice for j, shift, *_ in row],
            dtype=float,
        )
        unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1)[:, None], _EPS)
        moment = unit.T @ unit / cn
        eig = np.linalg.eigvalsh(moment)
        directional_anisotropy.append(float(np.sqrt(1.5 * np.sum((eig - 1.0 / 3.0) ** 2))))
        if cn == 4:
            cosines = _pair_cosines(unit)
            q = 1.0 - 3.0 / 8.0 * float(np.sum((cosines + 1.0 / 3.0) ** 2))
            tetra_q.append(q)
            angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
            tetra_angle_rmse.append(float(np.sqrt(np.mean((angles - 109.471220634) ** 2))))
        if cn == 6:
            cosines = np.sort(_pair_cosines(unit))
            ideal_cosines = np.asarray([-1.0] * 3 + [0.0] * 12)
            octa_cosine_rmse.append(float(np.sqrt(np.mean((cosines - ideal_cosines) ** 2))))
            angles = np.sort(np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0))))
            ideal_angles = np.asarray([90.0] * 12 + [180.0] * 3)
            octa_angle_rmse.append(float(np.sqrt(np.mean((angles - ideal_angles) ** 2))))

    return _PolyhedronSummary(
        poly_centers, low_centers, high_centers, cn4, cn6,
        radial_cv, normalized_strain_rms, directional_anisotropy,
        tetra_q, tetra_angle_rmse, octa_cosine_rmse, octa_angle_rmse,
    )


def _pair_cosines(unit_vectors: np.ndarray) -> np.ndarray:
    result = []
    for i in range(len(unit_vectors)):
        for j in range(i + 1, len(unit_vectors)):
            result.append(float(np.dot(unit_vectors[i], unit_vectors[j])))
    return np.asarray(result, dtype=float)


def _add_role_element_features(
    fb: _FeatureBuilder,
    records: Sequence[ElementRecord],
    low_centers: set[int],
    high_centers: set[int],
) -> None:
    def values(indices: set[int], attr: str) -> np.ndarray:
        return np.asarray([float(getattr(records[i], attr)) for i in sorted(indices)], dtype=float)

    low_z = values(low_centers, "atomic_number")
    high_z = values(high_centers, "atomic_number")
    low_r = values(low_centers, "covalent_radius")
    high_r = values(high_centers, "covalent_radius")
    fb.add("low_center_atomic_number_mean", np.mean(low_z) if low_z.size else 0.0)
    fb.add("high_center_atomic_number_mean", np.mean(high_z) if high_z.size else 0.0)
    fb.add(
        "center_atomic_number_contrast_abs",
        abs(float(np.mean(high_z) - np.mean(low_z))) if low_z.size and high_z.size else 0.0,
    )
    fb.add("low_center_covalent_radius_mean_ang", np.mean(low_r) if low_r.size else 0.0)
    fb.add("high_center_covalent_radius_mean_ang", np.mean(high_r) if high_r.size else 0.0)
    fb.add(
        "center_covalent_radius_contrast_abs_ang",
        abs(float(np.mean(high_r) - np.mean(low_r))) if low_r.size and high_r.size else 0.0,
    )
    fb.add(
        "high_center_f_block_fraction",
        np.mean([records[i].is_f_block for i in high_centers]) if high_centers else 0.0,
    )


def _summarize_bridges(
    n_sites: int,
    records: Sequence[ElementRecord],
    neighbors: Sequence[Sequence[tuple[int, np.ndarray, float, float]]],
    poly: _PolyhedronSummary,
) -> dict[str, float]:
    explicit_coordinators = {
        j for center in poly.poly_centers for j, *_ in neighbors[center] if j not in poly.poly_centers
    }
    ligand_indices = {
        i for i, record in enumerate(records)
        if i not in poly.poly_centers and (record.ligand_priority or i in explicit_coordinators)
    }
    # Explicit coordinating atoms are included even when their element is not in the
    # priority list, while candidate centers are never re-labelled as ligands.
    bridge_ligands = 0
    low_degrees: list[int] = []
    high_degrees: list[int] = []
    low_total_bonds = 0
    low_ligand_bonds = 0
    high_total_bonds = 0
    high_ligand_bonds = 0
    low_high_edges: list[tuple[int, int, np.ndarray]] = []
    shared: dict[tuple[int, int], set[int]] = {}

    for center in poly.low_centers:
        low_total_bonds += len(neighbors[center])
        low_ligand_bonds += sum(j in ligand_indices for j, *_ in neighbors[center])
    for center in poly.high_centers:
        high_total_bonds += len(neighbors[center])
        high_ligand_bonds += sum(j in ligand_indices for j, *_ in neighbors[center])

    for ligand in sorted(ligand_indices):
        low = [(j, shift) for j, shift, *_ in neighbors[ligand] if j in poly.low_centers]
        high = [(j, shift) for j, shift, *_ in neighbors[ligand] if j in poly.high_centers]
        centers = [(j, shift) for j, shift, *_ in neighbors[ligand] if j in poly.poly_centers]
        if low and high:
            bridge_ligands += 1
            low_degrees.append(len(low))
            high_degrees.append(len(high))
            for i, shift_i in low:
                for j, shift_j in high:
                    low_high_edges.append((i, j, np.asarray(shift_j, dtype=int) - np.asarray(shift_i, dtype=int)))
        for a in range(len(centers)):
            for b in range(a + 1, len(centers)):
                i, _si = centers[a]
                j, _sj = centers[b]
                if i == j:
                    continue
                key = (i, j) if i < j else (j, i)
                shared.setdefault(key, set()).add(ligand)

    center_nodes = sorted(poly.low_centers | poly.high_centers)
    remap = {site: idx for idx, site in enumerate(center_nodes)}
    remapped_edges = [(remap[i], remap[j], shift) for i, j, shift in low_high_edges if i in remap and j in remap]
    graph = _summarize_graph(len(center_nodes), remapped_edges) if center_nodes else _summarize_graph(1, [])

    counts = np.asarray([len(v) for v in shared.values()], dtype=float)
    corner = np.mean(counts == 1) if counts.size else 0.0
    edge = np.mean(counts == 2) if counts.size else 0.0
    face = np.mean(counts >= 3) if counts.size else 0.0
    pair_types = []
    for i, j in shared:
        if i in poly.low_centers and j in poly.low_centers:
            pair_types.append("low_low")
        elif i in poly.high_centers and j in poly.high_centers:
            pair_types.append("high_high")
        else:
            pair_types.append("low_high")

    denom_centers = max(len(center_nodes), 1)
    return {
        "coordinating_ligand_fraction": len(ligand_indices) / max(n_sites, 1),
        "low_high_bridge_ligand_fraction": bridge_ligands / max(len(ligand_indices), 1),
        "bridge_ligand_low_center_degree_mean": float(np.mean(low_degrees)) if low_degrees else 0.0,
        "bridge_ligand_high_center_degree_mean": float(np.mean(high_degrees)) if high_degrees else 0.0,
        "low_center_ligand_saturation": low_ligand_bonds / max(low_total_bonds, 1),
        "high_center_ligand_saturation": high_ligand_bonds / max(high_total_bonds, 1),
        "low_high_bridge_edges_per_center": len(low_high_edges) / denom_centers,
        "low_high_graph_component_dominance": graph.largest_component_fraction if center_nodes else 0.0,
        "low_high_graph_cycle_rank_per_center": graph.cycle_rank_per_node if center_nodes else 0.0,
        "low_high_graph_periodicity_rank": float(graph.periodicity_rank if center_nodes else 0),
        "low_high_graph_axis_span_fraction": graph.axis_span_fraction if center_nodes else 0.0,
        "polyhedron_pair_corner_sharing_fraction": float(corner),
        "polyhedron_pair_edge_sharing_fraction": float(edge),
        "polyhedron_pair_face_sharing_fraction": float(face),
        "polyhedron_pair_shared_ligands_mean": float(np.mean(counts)) if counts.size else 0.0,
        "polyhedron_pair_low_low_fraction": pair_types.count("low_low") / max(len(pair_types), 1),
        "polyhedron_pair_high_high_fraction": pair_types.count("high_high") / max(len(pair_types), 1),
        "polyhedron_pair_low_high_fraction": pair_types.count("low_high") / max(len(pair_types), 1),
    }


def _summarize_void_grid(
    structure: PeriodicStructure,
    records: Sequence[ElementRecord],
    cfg: DescriptorConfig,
) -> dict[str, float | tuple[int, int, int] | int]:
    shape = _choose_grid_shape(structure.lattice, cfg.grid_spacing, cfg.max_grid_points)
    axes = [(np.arange(n, dtype=float) + 0.5) / n for n in shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    anchor = _canonical_anchor(structure, records)
    points = np.mod(np.stack(mesh, axis=-1).reshape((-1, 3)) + anchor[None, :], 1.0)
    clearance = _grid_clearance(points, structure, records, cfg.grid_chunk_size)
    positive = clearance[clearance > 0.0]
    q = np.quantile(clearance, [0.50, 0.90, 0.95, 0.99])
    axial = _periodic_bottleneck_radii(clearance, shape)

    result: dict[str, float | tuple[int, int, int] | int] = {
        "void_clearance_max_ang": float(np.max(clearance)),
        "void_clearance_positive_mean_ang": float(np.mean(positive)) if positive.size else 0.0,
        "void_clearance_positive_std_ang": float(np.std(positive)) if positive.size else 0.0,
        "void_clearance_q50_ang": float(q[0]),
        "void_clearance_q90_ang": float(q[1]),
        "void_clearance_q95_ang": float(q[2]),
        "void_clearance_q99_ang": float(q[3]),
    }
    for probe in cfg.probe_radii:
        tag = str(probe).replace(".", "p")
        result[f"void_fraction_probe_{tag}_ang"] = float(np.mean(clearance >= probe))
    for axis, label in enumerate("abc"):
        result[f"axial_bottleneck_{label}_radius_ang"] = float(axial[axis])
    result.update(
        {
            "axial_bottleneck_radius_mean_ang": float(np.mean(axial)),
            "axial_bottleneck_radius_std_ang": float(np.std(axial)),
            "axial_bottleneck_radius_min_ang": float(np.min(axial)),
            "axial_bottleneck_radius_max_ang": float(np.max(axial)),
            "grid_largest_included_sphere_diameter_ang": 2.0 * max(0.0, float(np.max(clearance))),
            "grid_largest_free_sphere_diameter_ang": 2.0 * max(0.0, float(np.max(axial))),
            "axial_bottleneck_anisotropy": float(np.std(axial) / max(abs(float(np.mean(axial))), 1e-8)),
        }
    )
    for probe in (0.0, 1.0, 1.2, 1.4):
        tag = str(probe).replace(".", "p")
        result[f"channel_axis_fraction_probe_{tag}_ang"] = float(np.mean(axial >= probe))
    result["_grid_shape"] = tuple(int(x) for x in shape)
    result["_grid_points"] = int(points.shape[0])
    return result



def _canonical_anchor(structure: PeriodicStructure, records: Sequence[ElementRecord]) -> np.ndarray:
    """Choose an atom-attached grid origin invariant to translation and atom order.

    A site's key uses its element and its complete labelled minimum-image radial
    environment.  Ties correspond to indistinguishable sites; choosing any tied
    site changes the grid only by a crystallographic-equivalent origin in exact
    structures.
    """

    lattice = structure.lattice
    signatures: list[tuple[tuple[object, ...], int]] = []
    for i, record in enumerate(records):
        environment: list[tuple[int, float]] = []
        for j, other in enumerate(records):
            if i == j:
                continue
            base = structure.frac_coords[j] - structure.frac_coords[i]
            base -= np.round(base)
            candidates = base[None, :] + _IMAGE_OFFSETS_27
            distances = np.linalg.norm(candidates @ lattice, axis=1)
            environment.append((other.atomic_number, round(float(np.min(distances)), 8)))
        # Prefer the chemically heaviest site, then the lexicographically complete
        # environment.  Negating Z lets Python's min() implement that preference.
        key: tuple[object, ...] = (-record.atomic_number, tuple(sorted(environment)))
        signatures.append((key, i))
    _key, index = min(signatures, key=lambda item: item[0])
    return structure.frac_coords[index].copy()

def _choose_grid_shape(lattice: np.ndarray, spacing: float, max_points: int) -> tuple[int, int, int]:
    lengths = np.linalg.norm(lattice, axis=1)
    shape = np.maximum(3, np.ceil(lengths / spacing).astype(int))
    while int(np.prod(shape)) > max_points:
        axis = int(np.argmax(shape))
        if shape[axis] <= 3:
            break
        shape[axis] -= 1
    return tuple(int(x) for x in shape)


def _grid_clearance(
    points: np.ndarray,
    structure: PeriodicStructure,
    records: Sequence[ElementRecord],
    chunk_size: int,
) -> np.ndarray:
    result = np.full(points.shape[0], np.inf, dtype=float)
    lattice = structure.lattice
    offsets = _IMAGE_OFFSETS_27.astype(float)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        chunk = points[start:stop]
        best_surface = np.full(len(chunk), np.inf, dtype=float)
        for frac_atom, record in zip(structure.frac_coords, records):
            base = chunk - frac_atom
            base -= np.round(base)
            candidates = base[:, None, :] + offsets[None, :, :]
            cart = candidates @ lattice
            distances = np.sqrt(np.sum(cart * cart, axis=2))
            nearest = np.min(distances, axis=1) - record.vdw_radius
            best_surface = np.minimum(best_surface, nearest)
        result[start:stop] = best_surface
    return result



class _PeriodicDSU:
    """Disjoint set with integer lattice potentials and cycle translations."""

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=int)
        self.rank = np.zeros(n, dtype=np.int8)
        # diff[x] = potential(x) - potential(parent[x]) in unit-cell translations.
        self.diff = np.zeros((n, 3), dtype=int)
        self.basis: list[list[np.ndarray]] = [[] for _ in range(n)]

    def find(self, x: int) -> tuple[int, np.ndarray]:
        if self.parent[x] == x:
            return x, np.zeros(3, dtype=int)
        parent = int(self.parent[x])
        root, parent_potential = self.find(parent)
        total = self.diff[x] + parent_potential
        self.parent[x] = root
        self.diff[x] = total
        return root, total.copy()

    @staticmethod
    def _independent_basis(vectors: Sequence[np.ndarray]) -> list[np.ndarray]:
        basis: list[np.ndarray] = []
        rank = 0
        for raw in vectors:
            vector = np.asarray(raw, dtype=int)
            if not np.any(vector):
                continue
            candidate = basis + [vector]
            new_rank = _vector_rank(candidate)
            if new_rank > rank:
                basis.append(vector.copy())
                rank = new_rank
            if rank == 3:
                break
        return basis

    def union_constraint(self, u: int, v: int, translation: np.ndarray) -> int:
        """Impose potential(v) - potential(u) = translation and return root."""

        ru, du = self.find(u)
        rv, dv = self.find(v)
        t = np.asarray(translation, dtype=int)
        if ru == rv:
            cycle = du + t - dv
            if np.any(cycle):
                self.basis[ru] = self._independent_basis(self.basis[ru] + [cycle])
            return ru

        if self.rank[ru] < self.rank[rv]:
            # potential(ru) - potential(rv) = dv - t - du
            self.parent[ru] = rv
            self.diff[ru] = dv - t - du
            self.basis[rv] = self._independent_basis(self.basis[rv] + self.basis[ru])
            return rv
        self.parent[rv] = ru
        # potential(rv) - potential(ru) = t + du - dv
        self.diff[rv] = t + du - dv
        self.basis[ru] = self._independent_basis(self.basis[ru] + self.basis[rv])
        if self.rank[ru] == self.rank[rv]:
            self.rank[ru] += 1
        return ru


def _periodic_bottleneck_radii(clearance: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Maximum clearance at which a periodic grid component winds along each axis.

    Grid points are activated from high to low clearance.  Integer-potential
    union-find detects non-contractible cycles on the three-torus, avoiding an
    arbitrary unit-cell cut and making the result stable under aligned integer
    supercells.
    """

    n = int(np.prod(shape))
    order = np.argsort(-clearance, kind="mergesort")
    active = np.zeros(n, dtype=bool)
    dsu = _PeriodicDSU(n)
    thresholds = np.full(3, np.nan, dtype=float)
    for raw_index in order:
        index = int(raw_index)
        coord = list(np.unravel_index(index, shape))
        active[index] = True
        current = float(clearance[index])
        touched_roots: set[int] = {index}
        for axis in range(3):
            for delta in (-1, 1):
                other = coord.copy()
                other[axis] += delta
                translation = np.zeros(3, dtype=int)
                if other[axis] < 0:
                    other[axis] += shape[axis]
                    translation[axis] = -1
                elif other[axis] >= shape[axis]:
                    other[axis] -= shape[axis]
                    translation[axis] = 1
                neighbor = int(np.ravel_multi_index(tuple(other), shape))
                if active[neighbor]:
                    root = dsu.union_constraint(index, neighbor, translation)
                    touched_roots.add(root)
        for candidate in touched_roots:
            root, _ = dsu.find(candidate)
            flags = np.zeros(3, dtype=bool)
            for vector in dsu.basis[root]:
                flags |= np.asarray(vector) != 0
            newly = flags & np.isnan(thresholds)
            thresholds[newly] = current
        if np.all(np.isfinite(thresholds)):
            break
    thresholds[~np.isfinite(thresholds)] = float(np.min(clearance))
    return thresholds
