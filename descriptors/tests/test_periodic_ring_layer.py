from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from descriptors.periodic_ring_layer import (
    COLUMNS,
    InputError,
    Settings,
    StructureData,
    analyze_structure,
    parse_poscar,
)


def square_checkerboard(*, corrugation: float = 0.0, striped: bool = False) -> StructureData:
    lattice = np.diag([2.84, 2.84, 8.0])
    fractional = np.asarray(
        [
            (0.0, 0.0, 0.5 + corrugation),
            (0.5, 0.0, 0.5 - corrugation),
            (0.0, 0.5, 0.5 - corrugation),
            (0.5, 0.5, 0.5 + corrugation),
        ]
    )
    species = ("C", "C", "N", "N") if striped else ("C", "N", "N", "C")
    return StructureData(lattice, species, fractional, "checkerboard")


def square_single() -> StructureData:
    return StructureData(np.diag([1.42, 1.42, 8.0]), ("C",), np.asarray([(0.0, 0.0, 0.5)]), "square")


def simple_cubic() -> StructureData:
    return StructureData(np.diag([1.42, 1.42, 1.42]), ("C",), np.asarray([(0.0, 0.0, 0.0)]), "cubic")


def chain() -> StructureData:
    return StructureData(np.diag([1.42, 8.0, 8.0]), ("C",), np.asarray([(0.0, 0.5, 0.5)]), "chain")


def isolated_hexagon() -> StructureData:
    lattice = np.diag([12.0, 12.0, 12.0])
    radius = 1.40
    cart = np.asarray(
        [
            (6.0 + radius * math.cos(index * math.pi / 3.0), 6.0 + radius * math.sin(index * math.pi / 3.0), 6.0)
            for index in range(6)
        ]
    )
    return StructureData(lattice, ("C",) * 6, cart @ np.linalg.inv(lattice), "hexagon")


def shift_origin(structure: StructureData, shift: tuple[float, float, float]) -> StructureData:
    return StructureData(structure.lattice, structure.species, structure.fractional + np.asarray(shift), structure.name + "-shift")


def reorder(structure: StructureData, order: np.ndarray) -> StructureData:
    return StructureData(
        structure.lattice,
        tuple(structure.species[int(index)] for index in order),
        structure.fractional[order],
        structure.name + "-reorder",
    )


def transform_basis(structure: StructureData, matrix: np.ndarray) -> StructureData:
    matrix = np.asarray(matrix, dtype=int)
    inverse = np.linalg.inv(matrix)
    return StructureData(matrix @ structure.lattice, structure.species, structure.fractional @ inverse, structure.name + "-basis")


def rotate_structure(structure: StructureData) -> StructureData:
    angle = 0.37
    axis = np.asarray([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    rotation = np.asarray(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )
    return StructureData(structure.lattice @ rotation, structure.species, structure.fractional, structure.name + "-rotate")


def make_supercell(structure: StructureData, repetitions: tuple[int, int, int]) -> StructureData:
    diagonal = np.diag(repetitions)
    inverse = np.linalg.inv(diagonal)
    species: list[str] = []
    fractional: list[np.ndarray] = []
    for i in range(repetitions[0]):
        for j in range(repetitions[1]):
            for k in range(repetitions[2]):
                translation = np.asarray([i, j, k], dtype=float)
                for symbol, coordinate in zip(structure.species, structure.fractional, strict=True):
                    species.append(symbol)
                    fractional.append((coordinate + translation) @ inverse)
    return StructureData(diagonal @ structure.lattice, tuple(species), np.asarray(fractional), structure.name + "-supercell")


def assert_scalar_maps_close(testcase: unittest.TestCase, first: dict, second: dict, tolerance: float = 1.0e-8) -> None:
    testcase.assertEqual(set(first), set(COLUMNS))
    testcase.assertEqual(set(second), set(COLUMNS))
    for name in COLUMNS:
        left = first[name]
        right = second[name]
        if left is None or right is None:
            testcase.assertIs(left, right, name)
        else:
            testcase.assertAlmostEqual(float(left), float(right), delta=tolerance, msg=name)


class PeriodicRingLayerTests(unittest.TestCase):
    def test_square_layer_exact_mechanism(self) -> None:
        result = analyze_structure(square_single())
        self.assertEqual(result["status"], "ok")
        scalars = result["scalars"]
        self.assertEqual(set(scalars), set(COLUMNS))
        self.assertEqual(scalars["prlt_max_translation_rank"], 2)
        self.assertEqual(scalars["prlt_shortest_ring_size"], 4)
        self.assertAlmostEqual(scalars["prlt_ring_count_per_atom"], 1.0)
        self.assertAlmostEqual(scalars["prlt_ring_planarity_rms_norm_mean"], 0.0)
        self.assertAlmostEqual(scalars["prlt_rank2_ring_normal_alignment_mean"], 1.0)
        self.assertIsNone(scalars["prlt_element_assortativity_all_bonds"])
        self.assertEqual(result["unavailable_reasons"]["prlt_element_assortativity_all_bonds"], "assortativity_degenerate")

    def test_checkerboard_and_stripe_chemistry(self) -> None:
        checker = analyze_structure(square_checkerboard())
        stripe = analyze_structure(square_checkerboard(striped=True))
        self.assertAlmostEqual(checker["scalars"]["prlt_element_assortativity_all_bonds"], -1.0)
        self.assertAlmostEqual(checker["scalars"]["prlt_ring_heteroedge_fraction"], 1.0)
        self.assertAlmostEqual(checker["scalars"]["prlt_ring_species_diversity_mean"], 0.5)
        self.assertGreater(stripe["scalars"]["prlt_element_assortativity_all_bonds"], checker["scalars"]["prlt_element_assortativity_all_bonds"])
        self.assertLess(stripe["scalars"]["prlt_ring_heteroedge_fraction"], 1.0)

    def test_corrugation_changes_geometry_not_topology(self) -> None:
        flat = analyze_structure(square_checkerboard())
        corrugated = analyze_structure(square_checkerboard(corrugation=0.05))
        self.assertEqual(flat["scalars"]["prlt_shortest_ring_size"], corrugated["scalars"]["prlt_shortest_ring_size"])
        self.assertGreater(corrugated["scalars"]["prlt_ring_planarity_rms_norm_mean"], 0.1)
        self.assertGreater(corrugated["scalars"]["prlt_rank2_bond_out_of_plane_rms"], 0.1)
        self.assertAlmostEqual(corrugated["scalars"]["prlt_rank2_ring_normal_alignment_mean"], 1.0)

    def test_rank_three_control(self) -> None:
        result = analyze_structure(simple_cubic())
        self.assertEqual(result["scalars"]["prlt_max_translation_rank"], 3)
        self.assertEqual(result["scalars"]["prlt_rank2_atom_fraction"], 0.0)
        self.assertEqual(result["status"], "domain_no_rank2_ring_layer")
        self.assertEqual(result["scalars"]["prlt_shortest_ring_size"], 4)

    def test_rank_one_chain_has_typed_no_ring(self) -> None:
        result = analyze_structure(chain())
        self.assertEqual(result["scalars"]["prlt_max_translation_rank"], 1)
        self.assertEqual(result["status"], "domain_no_shortest_path_rings")
        self.assertEqual(result["scalars"]["prlt_ring_count_per_atom"], 0.0)
        self.assertIsNone(result["scalars"]["prlt_ring_size_mean"])
        self.assertEqual(result["unavailable_reasons"]["prlt_ring_size_mean"], "no_admitted_rings")

    def test_finite_hexagon(self) -> None:
        result = analyze_structure(isolated_hexagon())
        self.assertEqual(result["scalars"]["prlt_max_translation_rank"], 0)
        self.assertEqual(result["scalars"]["prlt_shortest_ring_size"], 6)
        self.assertAlmostEqual(result["scalars"]["prlt_ring_count_per_atom"], 1.0 / 6.0)
        self.assertEqual(result["status"], "domain_no_rank2_ring_layer")

    def test_origin_permutation_rotation_invariance(self) -> None:
        base = square_checkerboard(corrugation=0.03)
        reference = analyze_structure(base)["scalars"]
        variants = [
            shift_origin(base, (0.37, -0.21, 0.46)),
            reorder(base, np.asarray([2, 0, 3, 1])),
            rotate_structure(base),
        ]
        for variant in variants:
            assert_scalar_maps_close(self, reference, analyze_structure(variant)["scalars"], 2.0e-8)

    def test_unimodular_basis_invariance(self) -> None:
        base = square_checkerboard(corrugation=0.03)
        transformed = transform_basis(base, np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]]))
        assert_scalar_maps_close(self, analyze_structure(base)["scalars"], analyze_structure(transformed)["scalars"], 2.0e-8)

    def test_supercell_invariance(self) -> None:
        base = square_checkerboard(corrugation=0.03)
        reference = analyze_structure(base)["scalars"]
        for repetitions in ((2, 1, 1), (1, 2, 1), (2, 2, 1)):
            variant = make_supercell(base, repetitions)
            assert_scalar_maps_close(self, reference, analyze_structure(variant)["scalars"], 2.0e-8)

    def test_resource_limited_ring_search_is_not_partial_data(self) -> None:
        result = analyze_structure(square_single(), settings=Settings(max_degree_for_ring_search=3))
        self.assertEqual(result["status"], "resource_limited_partial")
        self.assertEqual(result["scalars"]["prlt_max_translation_rank"], 2)
        self.assertIsNone(result["scalars"]["prlt_ring_count_per_atom"])
        self.assertEqual(result["unavailable_reasons"]["prlt_ring_count_per_atom"], "ring_search_resource_limit")

    def test_poscar_parser_direct_and_selective(self) -> None:
        text = """checkerboard
1.0
2.84 0 0
0 2.84 0
0 0 8
C N
2 2
Selective dynamics
Direct
0 0 0.5 T T T
0.5 0.5 0.5 T T T
0.5 0 0.5 T T T
0 0.5 0.5 T T T
"""
        structure = parse_poscar(text, is_text=True)
        result = analyze_structure(structure)
        self.assertEqual(result["scalars"]["prlt_shortest_ring_size"], 4)
        self.assertAlmostEqual(result["scalars"]["prlt_element_assortativity_all_bonds"], -1.0)

    def test_poscar_parser_cartesian_scale(self) -> None:
        text = """square
2.0
0.71 0 0
0 0.71 0
0 0 4
C
1
Cartesian
0 0 2
"""
        structure = parse_poscar(text, is_text=True)
        self.assertTrue(np.allclose(structure.lattice, np.diag([1.42, 1.42, 8.0])))
        self.assertTrue(np.allclose(structure.fractional[0], (0.0, 0.0, 0.5)))

    def test_malformed_input(self) -> None:
        with self.assertRaises(InputError):
            StructureData(np.zeros((3, 3)), ("C",), np.asarray([(0, 0, 0)]))
        with self.assertRaises(InputError):
            StructureData(np.eye(3), ("C2+",), np.asarray([(0, 0, 0)]))
        with self.assertRaises(InputError):
            parse_poscar("too short", is_text=True)

    def test_no_bond_domain(self) -> None:
        structure = StructureData(np.diag([10.0, 10.0, 10.0]), ("C",), np.asarray([(0.0, 0.0, 0.0)]), "isolated")
        result = analyze_structure(structure)
        self.assertEqual(result["status"], "domain_no_bonds")
        self.assertEqual(result["scalars"]["prlt_bonded_atom_fraction"], 0.0)
        self.assertEqual(result["scalars"]["prlt_ring_count_per_atom"], 0.0)


if __name__ == "__main__":
    unittest.main()
