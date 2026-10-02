from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from descriptors.periodic_host_guest_geometry import FEATURE_NAMES, compute
from descriptors.phgg_graph import contact_directionality_anisotropy, minimum_image_vector
from descriptors.phgg_radii import surface_radius
from descriptors.phgg_types import DescriptorError, frac_to_cart, inverse, norm


def make_poscar(
    title: str,
    lattice: list[tuple[float, float, float]],
    species: list[str],
    counts: list[int],
    coords: list[tuple[float, float, float]],
    *,
    mode: str = "Direct",
    scale: float = 1.0,
) -> str:
    return "\n".join(
        [
            title,
            f"{scale:.12g}",
            *(" ".join(f"{value:.12g}" for value in vector) for vector in lattice),
            " ".join(species),
            " ".join(str(value) for value in counts),
            mode,
            *(" ".join(f"{value:.12g}" for value in coord) for coord in coords),
        ]
    ) + "\n"


class PeriodicHostGuestGeometryTests(unittest.TestCase):
    def assert_fixed_finite_schema(self, result) -> None:
        self.assertEqual(tuple(result.features), FEATURE_NAMES)
        self.assertTrue(all(isinstance(value, float) and math.isfinite(value) for value in result.features.values()))
        self.assertEqual(result.metadata["fixed_feature_names"], list(FEATURE_NAMES))

    def test_one_dimensional_periodic_host_and_finite_guest(self) -> None:
        text = make_poscar(
            "one dimensional chain",
            [(1.8, 0, 0), (0, 10, 0), (0, 0, 10)],
            ["C", "He"],
            [1, 1],
            [(0, 0, 0), (0.5, 0.5, 0.5)],
        )
        result = compute(text, {"max_grid_points": 64})
        self.assert_fixed_finite_schema(result)
        self.assertEqual(result.features["host_network_dimensionality"], 1.0)
        self.assertEqual(result.features["host_selection_periodic_ratio"], 1.0)
        self.assertEqual(result.features["finite_guest_component_count_per_cell"], 1.0)
        self.assertEqual(result.features["host_atom_fraction_ratio"], 0.5)

    def test_two_dimensional_periodic_host(self) -> None:
        text = make_poscar(
            "two dimensional net",
            [(1.8, 0, 0), (0, 1.8, 0), (0, 0, 10)],
            ["C", "He"],
            [1, 1],
            [(0, 0, 0), (0.5, 0.5, 0.5)],
        )
        result = compute(text, {"max_grid_points": 64})
        self.assertEqual(result.features["host_network_dimensionality"], 2.0)
        cycles = result.metadata["components"][0]["cycle_translations"]
        self.assertTrue(any(vector[0] for vector in cycles))
        self.assertTrue(any(vector[1] for vector in cycles))

    def test_three_dimensional_host_guest_distance_is_analytic(self) -> None:
        text = make_poscar(
            "cubic host with explicit guest",
            [(3, 0, 0), (0, 3, 0), (0, 0, 3)],
            ["Cu", "He"],
            [1, 1],
            [(0, 0, 0), (0.5, 0.5, 0.5)],
        )
        result = compute(text, {"max_grid_points": 64})
        expected_distance = 1.5 * math.sqrt(3.0)
        expected_gap = expected_distance - surface_radius("Cu")[0] - surface_radius("He")[0]
        self.assertEqual(result.features["host_network_dimensionality"], 3.0)
        self.assertAlmostEqual(result.features["host_guest_min_distance_A"], expected_distance, places=12)
        self.assertAlmostEqual(result.features["host_guest_min_surface_gap_A"], expected_gap, places=12)
        self.assertEqual(result.features["host_guest_contact_atom_fraction_ratio"], 1.0)

    def test_no_guest_has_typed_zero_channels(self) -> None:
        text = make_poscar(
            "guest free cubic host",
            [(3, 0, 0), (0, 3, 0), (0, 0, 3)],
            ["Cu"],
            [1],
            [(0, 0, 0)],
        )
        result = compute(text, {"max_grid_points": 32})
        self.assertEqual(result.features["host_guest_present_ratio"], 0.0)
        self.assertEqual(result.features["finite_guest_component_count_per_cell"], 0.0)
        self.assertEqual(result.features["host_guest_min_distance_A"], 0.0)
        self.assertFalse(result.metadata["host_guest_valid"])

    def test_all_periodic_components_are_host_not_guest(self) -> None:
        text = make_poscar(
            "two detached periodic layers",
            [(1.7, 0, 0), (0, 1.7, 0), (0, 0, 12)],
            ["C", "N"],
            [1, 1],
            [(0, 0, 0.2), (0.5, 0.5, 0.7)],
        )
        result = compute(text, {"max_grid_points": 32})
        self.assertEqual(result.features["component_count_per_cell"], 2.0)
        self.assertEqual(result.features["host_atom_fraction_ratio"], 1.0)
        self.assertEqual(result.features["finite_guest_atom_fraction_ratio"], 0.0)
        self.assertEqual(result.metadata["host_selection_mode"], "all_periodic_components")

    def test_equal_largest_finite_components_are_all_fallback_host(self) -> None:
        text = make_poscar(
            "two equal finite dimers",
            [(20, 0, 0), (0, 20, 0), (0, 0, 20)],
            ["C"],
            [4],
            [(0.10, 0.10, 0.10), (0.17, 0.10, 0.10), (0.60, 0.60, 0.60), (0.67, 0.60, 0.60)],
        )
        result = compute(text, {"max_grid_points": 32})
        self.assertEqual(result.features["host_selection_periodic_ratio"], 0.0)
        self.assertEqual(result.features["host_atom_fraction_ratio"], 1.0)
        self.assertEqual(result.features["host_guest_present_ratio"], 0.0)
        self.assertEqual(result.metadata["host_selection_mode"], "all_largest_components_fallback")

    def test_radius_gate_boundary_changes_component_count(self) -> None:
        cutoff = 1.18 * (0.76 + 0.76) + 0.10
        cell = 20.0
        below = make_poscar(
            "below radius threshold",
            [(cell, 0, 0), (0, cell, 0), (0, 0, cell)],
            ["C"],
            [2],
            [(0, 0, 0), ((cutoff - 1e-4) / cell, 0, 0)],
        )
        above = make_poscar(
            "above radius threshold",
            [(cell, 0, 0), (0, cell, 0), (0, 0, cell)],
            ["C"],
            [2],
            [(0, 0, 0), ((cutoff + 1e-4) / cell, 0, 0)],
        )
        below_result = compute(below, {"max_grid_points": 8})
        above_result = compute(above, {"max_grid_points": 8})
        self.assertEqual(below_result.features["component_count_per_cell"], 1.0)
        self.assertEqual(above_result.features["component_count_per_cell"], 2.0)

    def test_finite_component_unwraps_across_cell_boundary(self) -> None:
        text = make_poscar(
            "guest dimer across c boundary",
            [(3, 0, 0), (0, 3, 0), (0, 0, 10)],
            ["Cu", "N"],
            [1, 2],
            [(0, 0, 0.5), (0.5, 0.5, 0.97), (0.5, 0.5, 0.03)],
        )
        result = compute(text, {"max_grid_points": 32})
        self.assertEqual(result.features["host_network_dimensionality"], 2.0)
        self.assertEqual(result.features["finite_guest_component_count_per_cell"], 1.0)
        self.assertAlmostEqual(
            result.features["guest_surface_envelope_diameter_A"],
            2 * (0.3 + surface_radius("N")[0]),
            places=12,
        )

    def test_skew_cell_translation_invariance(self) -> None:
        lattice = [(3.0, 0.0, 0.0), (1.1, 3.2, 0.0), (0.4, 0.7, 3.4)]
        base = make_poscar(
            "skew",
            lattice,
            ["Cu", "He"],
            [1, 1],
            [(0.02, 0.03, 0.04), (0.92, 0.91, 0.90)],
        )
        shifted = make_poscar(
            "skew translated",
            lattice,
            ["Cu", "He"],
            [1, 1],
            [(0.39, 0.22, 0.53), (1.29, 1.10, 1.39)],
        )
        left = compute(base, {"max_grid_points": 64})
        right = compute(shifted, {"max_grid_points": 64})
        for name in FEATURE_NAMES:
            self.assertAlmostEqual(left.features[name], right.features[name], places=10, msg=name)

    def test_host_atom_permutation_invariance_including_grid_phase(self) -> None:
        lattice = [(3.0, 0.0, 0.0), (0.2, 10.0, 0.0), (0.0, 0.3, 10.0)]
        left_text = make_poscar(
            "ordered host",
            lattice,
            ["C", "He"],
            [2, 1],
            [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.3, 0.5, 0.5)],
        )
        right_text = make_poscar(
            "permuted host",
            lattice,
            ["C", "He"],
            [2, 1],
            [(0.5, 0.0, 0.0), (0.0, 0.0, 0.0), (0.3, 0.5, 0.5)],
        )
        left = compute(left_text, {"max_grid_points": 64})
        right = compute(right_text, {"max_grid_points": 64})
        for name in FEATURE_NAMES:
            self.assertAlmostEqual(left.features[name], right.features[name], places=12, msg=name)

    def test_exact_minimum_image_matches_wide_brute_force_for_skew_cell(self) -> None:
        lattice = ((4.0, 0.0, 0.0), (3.91, 0.43, 0.0), (0.27, 0.19, 5.7))
        inv = inverse(lattice)
        start = (0.07, 0.11, 0.13)
        end = (0.83, 0.79, 0.91)
        vector, distance, shift = minimum_image_vector(start, end, lattice, inv)
        brute = []
        delta = tuple(end[axis] - start[axis] for axis in range(3))
        for image in itertools.product(range(-8, 9), repeat=3):
            candidate = frac_to_cart(tuple(delta[axis] + image[axis] for axis in range(3)), lattice)
            brute.append((norm(candidate), image, candidate))
        expected_distance, expected_shift, expected_vector = min(brute, key=lambda item: (item[0], item[1]))
        self.assertAlmostEqual(distance, expected_distance, places=12)
        self.assertEqual(shift, expected_shift)
        for actual, expected in zip(vector, expected_vector):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_multiple_guest_components_use_atom_weighted_cavity_average(self) -> None:
        text = make_poscar(
            "two finite guests",
            [(3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 3.0)],
            ["Cu", "He"],
            [1, 2],
            [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.5, 0.5, 0.15)],
        )
        result = compute(text, {"max_grid_points": 64})
        first = 2.0 * ((3.0 * math.sqrt(3.0) / 2.0) - 1.96)
        second = 2.0 * (math.sqrt(1.5**2 + 1.5**2 + 0.45**2) - 1.96)
        self.assertEqual(result.features["finite_guest_component_count_per_cell"], 2.0)
        self.assertAlmostEqual(
            result.features["guest_centered_cavity_diameter_A"],
            (first + second) / 2.0,
            places=12,
        )

    def test_contact_directionality_treats_tied_symmetric_contacts_without_index_bias(self) -> None:
        axes = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
        self.assertAlmostEqual(contact_directionality_anisotropy([axes]), 0.0, places=12)
        self.assertAlmostEqual(contact_directionality_anisotropy([[(1.0, 0.0, 0.0)]]), 1.0, places=12)
        self.assertEqual(contact_directionality_anisotropy([]), 0.0)

    def test_negative_volume_scale_and_cartesian_mode(self) -> None:
        direct = make_poscar(
            "negative scale direct",
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            ["Cu"],
            [1],
            [(0, 0, 0)],
            scale=-27.0,
        )
        cartesian = make_poscar(
            "positive scale Cartesian",
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            ["Cu"],
            [1],
            [(0, 0, 0)],
            mode="Cartesian",
            scale=3.0,
        )
        left = compute(direct, {"max_grid_points": 8})
        right = compute(cartesian, {"max_grid_points": 8})
        self.assertEqual(left.metadata["cell_volume_A3"], 27.0)
        self.assertEqual(left.features, right.features)

    def test_invalid_parameters_and_vasp4_are_rejected(self) -> None:
        valid = make_poscar(
            "valid",
            [(5, 0, 0), (0, 5, 0), (0, 0, 5)],
            ["C"],
            [1],
            [(0, 0, 0)],
        )
        with self.assertRaises(DescriptorError):
            compute(valid, {"bond_scale": 3.0})
        with self.assertRaises(DescriptorError):
            compute(valid, {"max_grid_points": 8.5})
        vasp4 = "x\n1\n5 0 0\n0 5 0\n0 0 5\n1\nDirect\n0 0 0\n"
        with self.assertRaises(DescriptorError):
            compute(vasp4)

    def test_cli_emits_same_fixed_schema(self) -> None:
        text = make_poscar(
            "cli",
            [(3, 0, 0), (0, 3, 0), (0, 0, 3)],
            ["Cu", "He"],
            [1, 1],
            [(0, 0, 0), (0.5, 0.5, 0.5)],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "POSCAR"
            path.write_text(text, encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "descriptors.periodic_host_guest_geometry",
                    "--poscar",
                    str(path),
                    "--params",
                    '{"max_grid_points":32}',
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        payload = json.loads(completed.stdout)
        self.assertEqual(tuple(payload["features"]), tuple(sorted(FEATURE_NAMES)))
        self.assertEqual(payload["metadata"]["feature_id"], "periodic-host-guest-geometry/1")


if __name__ == "__main__":
    unittest.main()
