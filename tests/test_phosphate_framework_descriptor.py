from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phosphate_framework_descriptor import (
    DescriptorConfig,
    PeriodicStructure,
    featurize_structure,
    parse_poscar_text,
    write_poscar,
)


def ideal_tetrahedron(cell: float = 8.0) -> PeriodicStructure:
    lattice = np.eye(3) * cell
    center = np.array([0.5, 0.5, 0.5])
    frac_length = 1.55 / math.sqrt(3.0) / cell
    directions = np.array(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
    )
    coords = np.vstack([center, center + directions * frac_length])
    return PeriodicStructure(lattice, ("P", "O", "O", "O", "O"), coords, "ideal PO4")


def ideal_octahedron(cell: float = 9.0) -> PeriodicStructure:
    lattice = np.eye(3) * cell
    center = np.array([0.5, 0.5, 0.5])
    frac_length = 1.95 / cell
    directions = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    coords = np.vstack([center, center + directions * frac_length])
    return PeriodicStructure(lattice, ("Ti",) + ("O",) * 6, coords, "ideal TiO6")


class PoscarTests(unittest.TestCase):
    def test_direct_round_trip(self) -> None:
        structure = ideal_tetrahedron()
        recovered = parse_poscar_text(write_poscar(structure))
        self.assertEqual(recovered.species, structure.species)
        np.testing.assert_allclose(recovered.lattice, structure.lattice, atol=1e-10)
        np.testing.assert_allclose(recovered.frac_coords, structure.frac_coords, atol=1e-10)

    def test_cartesian_and_selective_dynamics(self) -> None:
        text = """cartesian test
2.0
1 0 0
0 1 0
0 0 1
Si O
1 1
Selective dynamics
Cartesian
0.0 0.0 0.0 T T T
0.5 0.5 0.5 F F F
"""
        structure = parse_poscar_text(text)
        np.testing.assert_allclose(structure.lattice, np.eye(3) * 2.0)
        np.testing.assert_allclose(structure.frac_coords[1], [0.5, 0.5, 0.5])

    def test_vasp4_requires_explicit_species(self) -> None:
        text = """vasp4
1
3 0 0
0 3 0
0 0 3
1 1
Direct
0 0 0
0.5 0.5 0.5
"""
        with self.assertRaisesRegex(ValueError, "species_override"):
            parse_poscar_text(text)
        parsed = parse_poscar_text(text, species_override=["Na", "Cl"])
        self.assertEqual(parsed.species, ("Na", "Cl"))


class DescriptorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DescriptorConfig(grid_spacing=1.0, max_grid_points=16_384)

    def test_fixed_finite_vector(self) -> None:
        result = featurize_structure(ideal_tetrahedron(), self.config)
        self.assertEqual(len(result.feature_names), 103)
        self.assertEqual(len(set(result.feature_names)), 103)
        self.assertTrue(np.all(np.isfinite(result.values)))
        self.assertEqual(result.to_jsonable()["schema"], "phosphate-framework-pore-descriptor/1")

    def test_ideal_tetrahedral_order(self) -> None:
        features = featurize_structure(ideal_tetrahedron(), self.config).as_dict()
        self.assertAlmostEqual(features["tetrahedral_q_mean"], 1.0, places=10)
        self.assertLess(features["tetrahedral_angle_rmse_deg_mean"], 1e-6)
        self.assertGreater(features["cn4_center_fraction"], 0.0)

    def test_ideal_octahedral_order(self) -> None:
        features = featurize_structure(ideal_octahedron(), self.config).as_dict()
        self.assertLess(features["octahedral_cosine_rmse_mean"], 1e-10)
        self.assertLess(features["octahedral_angle_rmse_deg_mean"], 1e-8)
        self.assertGreater(features["cn6_center_fraction"], 0.0)

    def test_permutation_and_translation_invariance(self) -> None:
        structure = ideal_tetrahedron()
        baseline = featurize_structure(structure, self.config)
        order = [4, 2, 0, 3, 1]
        permuted = featurize_structure(structure.reordered(order), self.config)
        translated = featurize_structure(structure.translated([0.137, 0.212, 0.311]), self.config)
        np.testing.assert_allclose(permuted.values, baseline.values, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(translated.values, baseline.values, atol=1e-10, rtol=1e-10)

    def test_aligned_supercell_invariance(self) -> None:
        structure = ideal_tetrahedron()
        baseline = featurize_structure(structure, self.config)
        supercell = featurize_structure(structure.supercell([2, 1, 1]), self.config)
        np.testing.assert_allclose(supercell.values, baseline.values, atol=1e-9, rtol=1e-9)

    def test_periodic_graph_rank(self) -> None:
        # The one-site quotient graph has one self-image bond along each axis.
        structure = PeriodicStructure(np.eye(3) * 2.2, ("Si",), np.array([[0.0, 0.0, 0.0]]))
        features = featurize_structure(structure, DescriptorConfig(grid_spacing=0.8)).as_dict()
        self.assertEqual(features["bond_graph_periodicity_rank"], 3.0)
        self.assertEqual(features["bond_graph_axis_span_fraction"], 1.0)

    def test_void_geometry_distinguishes_cell_scale(self) -> None:
        small = PeriodicStructure(np.eye(3) * 4.0, ("Xe",), np.array([[0.0, 0.0, 0.0]]))
        large = PeriodicStructure(np.eye(3) * 8.0, ("Xe",), np.array([[0.0, 0.0, 0.0]]))
        f_small = featurize_structure(small, self.config).as_dict()
        f_large = featurize_structure(large, self.config).as_dict()
        self.assertGreater(
            f_large["grid_largest_included_sphere_diameter_ang"],
            f_small["grid_largest_included_sphere_diameter_ang"],
        )
        self.assertGreater(f_large["void_fraction_probe_1p4_ang"], f_small["void_fraction_probe_1p4_ang"])

    def test_feature_namespace_does_not_claim_unavailable_history(self) -> None:
        names = featurize_structure(ideal_tetrahedron(), self.config).feature_names
        forbidden = ("catal", "durability", "monolith", "vacancy", "defect", "reaction", "surface_area")
        self.assertFalse(any(token in name.lower() for token in forbidden for name in names))


if __name__ == "__main__":
    unittest.main()
