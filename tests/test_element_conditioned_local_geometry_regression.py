from __future__ import annotations

import importlib.util
import unittest


DEPENDENCIES_AVAILABLE = (
    importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("pymatgen") is not None
)

if DEPENDENCIES_AVAILABLE:
    import numpy as np
    from pymatgen.core import Lattice, Structure

    from element_conditioned_local_geometry import (
        ELEMENT_CONDITIONED_VARIANCE_FLOOR,
        FEATURE_NAMES,
        TIE_ABSOLUTE_TOLERANCE_ANGSTROM,
        compute_descriptor,
    )


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    "scientific dependency pins are exercised by the formal private Actions run",
)
class ElementConditionedLocalGeometryRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        # Frozen-development regression dft_3d_JVASP-100493.  Its 12th--14th
        # periodic distances are degenerate to a few micro-angstrom and its
        # scale/radial group variances are numerical zero.  Earlier code could
        # therefore change shell membership or divide two round-off variances
        # after a representation-only transformation.
        self.structure = Structure(
            Lattice(
                [
                    [4.61862159617113, 8.211756e-10, 2.66656229639524],
                    [1.5395405328312552, 4.354478512507363, 2.6665622963952402],
                    [2.3226356e-09, 1.6423515e-09, 5.333125584744634],
                ]
            ),
            ["Al", "Li", "Pr", "Pr"],
            [
                [0.49999971242912994, 0.500000170517446, 0.5000000418821359],
                [0.0, 0.0, 0.0],
                [0.24999985621456497, 0.250000085258723, 0.25000002094106794],
                [0.7499995686436951, 0.7500002557761691, 0.7500000628232035],
            ],
            coords_are_cartesian=False,
        )

    def assert_features_close(self, left, right, tolerance=3.0e-7):
        for name in FEATURE_NAMES:
            self.assertAlmostEqual(
                float(left["features"][name]),
                float(right["features"][name]),
                delta=tolerance,
                msg=name,
            )

    def test_near_degenerate_shell_and_zero_variance_are_representation_stable(self):
        reference = compute_descriptor(self.structure)
        self.assertEqual(
            reference["parameters"]["tie_absolute_tolerance_angstrom"],
            TIE_ABSOLUTE_TOLERANCE_ANGSTROM,
        )
        self.assertEqual(
            reference["parameters"]["element_conditioned_variance_floor"],
            ELEMENT_CONDITIONED_VARIANCE_FLOOR,
        )
        self.assertEqual(
            reference["features"]["shell_scale_element_between_fraction"], 0.0
        )
        self.assertEqual(
            reference["features"]["radial_disorder_element_between_fraction"], 0.0
        )

        lattice = np.asarray(self.structure.lattice.matrix, dtype=np.float64)
        fractions = np.asarray(self.structure.frac_coords, dtype=np.float64)
        species = [site.specie for site in self.structure]

        axis = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        angle = 0.713
        x, y, z = axis
        c, s, one = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
        rotation = np.asarray(
            [
                [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
                [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
                [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
            ]
        )
        rotated = Structure(
            Lattice(lattice @ rotation), species, fractions, coords_are_cartesian=False
        )
        translated = Structure(
            Lattice(lattice),
            species,
            np.mod(fractions + np.asarray([0.173, 0.297, 0.419]), 1.0),
            coords_are_cartesian=False,
        )
        order = np.arange(len(self.structure) - 1, -1, -1, dtype=np.int64)
        permuted = Structure(
            Lattice(lattice),
            [species[int(index)] for index in order],
            fractions[order],
            coords_are_cartesian=False,
        )
        supercell = self.structure.copy()
        supercell.make_supercell([2, 1, 1])

        for changed in (rotated, translated, permuted, supercell):
            self.assert_features_close(reference, compute_descriptor(changed))


if __name__ == "__main__":
    unittest.main()
