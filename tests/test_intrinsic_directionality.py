"""Synthetic contract tests for intrinsic_directionality.

The repository's lightweight unit-test job deliberately carries no scientific
Python stack. These tests therefore skip there and execute in environments
that provide the public pinned NumPy/pymatgen runtime. The formal frozen-1000
run is recorded separately in the task provenance.
"""

from __future__ import annotations

import math
import unittest

try:
    import numpy as np
    from pymatgen.core import Lattice, Structure

    from intrinsic_directionality import (
        DescriptorUnavailable,
        compare_explicit_structures,
        compute_descriptor,
    )

    HAVE_SCIENTIFIC_RUNTIME = True
except ImportError:
    HAVE_SCIENTIFIC_RUNTIME = False


@unittest.skipUnless(HAVE_SCIENTIFIC_RUNTIME, "requires NumPy and pymatgen")
class IntrinsicDirectionalitySyntheticTests(unittest.TestCase):
    def assert_feature_close(
        self, descriptor: dict, feature: str, expected: float, places: int = 12
    ) -> None:
        value = descriptor["features"][feature]
        self.assertIsNotNone(value)
        self.assertAlmostEqual(float(value), expected, places=places)

    def test_ideal_axis_plane_and_cubic_fabric_limits(self) -> None:
        chain = compute_descriptor(
            Structure(Lattice.orthorhombic(2.0, 9.0, 9.0), ["Si"], [[0, 0, 0]])
        )
        plane = compute_descriptor(
            Structure(Lattice.orthorhombic(2.0, 2.0, 9.0), ["Si"], [[0, 0, 0]])
        )
        cubic = compute_descriptor(
            Structure(Lattice.cubic(2.0), ["Si"], [[0, 0, 0]])
        )

        self.assert_feature_close(chain, "first_shell_coordination_mean", 2.0)
        self.assert_feature_close(chain, "bond_fabric_q2", 1.0)
        self.assert_feature_close(chain, "bond_fabric_linearity", 1.0)
        self.assert_feature_close(chain, "bond_fabric_planarity", 0.0)
        self.assert_feature_close(chain, "bond_fabric_isotropy", 0.0)

        self.assert_feature_close(plane, "first_shell_coordination_mean", 4.0)
        self.assert_feature_close(plane, "bond_fabric_q2", 0.5)
        self.assert_feature_close(plane, "bond_fabric_linearity", 0.0)
        self.assert_feature_close(plane, "bond_fabric_planarity", 1.0)
        self.assert_feature_close(plane, "bond_fabric_isotropy", 0.0)

        self.assert_feature_close(cubic, "first_shell_coordination_mean", 6.0)
        self.assert_feature_close(cubic, "bond_fabric_q2", 0.0)
        self.assert_feature_close(cubic, "bond_fabric_linearity", 0.0)
        self.assert_feature_close(cubic, "bond_fabric_planarity", 0.0)
        self.assert_feature_close(cubic, "bond_fabric_isotropy", 1.0)

    def test_rigid_rotation_and_fractional_translation_are_invariant(self) -> None:
        reference = Structure(
            Lattice.from_parameters(4.1, 5.2, 6.3, 83.0, 97.0, 108.0),
            ["Na", "Cl", "O"],
            [[0.0, 0.0, 0.0], [0.41, 0.37, 0.23], [0.17, 0.72, 0.63]],
        )
        baseline = compute_descriptor(reference)

        axis = np.asarray([1.0, 2.0, 3.0])
        axis /= np.linalg.norm(axis)
        angle = 0.713
        x, y, z = axis
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1.0 - c
        rotation = np.asarray(
            [
                [c + x * x * t, x * y * t - z * s, x * z * t + y * s],
                [y * x * t + z * s, c + y * y * t, y * z * t - x * s],
                [z * x * t - y * s, z * y * t + x * s, c + z * z * t],
            ]
        )
        rotated = Structure(
            Lattice(np.asarray(reference.lattice.matrix) @ rotation),
            [site.specie for site in reference],
            reference.frac_coords,
        )
        translated = Structure(
            reference.lattice,
            [site.specie for site in reference],
            np.mod(reference.frac_coords + [0.173, 0.297, 0.419], 1.0),
        )

        for transformed in (rotated, translated):
            result = compute_descriptor(transformed)
            for name, value in baseline["features"].items():
                other = result["features"][name]
                self.assertEqual(value is None, other is None, name)
                if value is not None:
                    self.assertAlmostEqual(float(value), float(other), places=9, msg=name)

    def test_paired_extension_requires_explicit_correspondence(self) -> None:
        reference = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
        target = Structure(Lattice.cubic(4.08), ["Si"], [[0, 0, 0]])
        with self.assertRaises(DescriptorUnavailable):
            compare_explicit_structures(reference, target)

        result = compare_explicit_structures(
            reference, target, assume_lattice_correspondence=True
        )
        self.assertAlmostEqual(
            result["lattice_change"]["log_volume_strain"],
            3.0 * math.log(1.02),
            places=12,
        )
        self.assertAlmostEqual(
            result["lattice_change"]["hencky_deviatoric_norm"], 0.0, places=12
        )


if __name__ == "__main__":
    unittest.main()
