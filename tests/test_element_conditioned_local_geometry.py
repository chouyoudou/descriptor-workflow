from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


DEPENDENCIES_AVAILABLE = (
    importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("pymatgen") is not None
)

if DEPENDENCIES_AVAILABLE:
    import numpy as np
    from pymatgen.core import Lattice, Structure

    from element_conditioned_local_geometry import (
        FEATURE_NAMES,
        compute_descriptor,
        descriptor_from_record,
        main,
    )


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    "scientific dependency pins are exercised by the formal private Actions run",
)
class ElementConditionedLocalGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.binary = Structure(
            Lattice.cubic(4.0),
            ["Mg", "Zn"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        )

    def assert_features_close(self, left, right, tolerance=2.0e-8):
        for name in FEATURE_NAMES:
            self.assertAlmostEqual(
                float(left["features"][name]),
                float(right["features"][name]),
                delta=tolerance,
                msg=name,
            )

    def test_monatomic_chemical_channels_are_zero(self):
        descriptor = compute_descriptor(
            Structure(Lattice.cubic(3.0), ["Mg"], [[0.0, 0.0, 0.0]])
        )
        for name in (
            "composition_atomic_number_std_norm",
            "composition_pairwise_atomic_number_contrast",
            "site_atomic_number_contrast_mean",
            "site_atomic_number_contrast_std",
            "site_contrast_radial_coupling_mean",
            "site_contrast_radial_coupling_std",
            "rarity_weighted_atomic_number_contrast_shift",
            "rarity_weighted_contrast_geometry_shift",
            "z_weighted_shell_scale_separation",
            "z_weighted_radial_disorder_separation",
            "z_weighted_atomic_number_contrast_separation",
            "z_weighted_contrast_geometry_separation",
            "z_weighted_element_environment_separation",
        ):
            self.assertEqual(descriptor["features"][name], 0.0, name)
        self.assertFalse(descriptor["diagnostics"]["rarity_weighting_active"])

    def test_binary_atomic_number_contrast_is_positive(self):
        features = compute_descriptor(self.binary)["features"]
        self.assertGreater(features["composition_atomic_number_std_norm"], 0.0)
        self.assertGreater(features["composition_pairwise_atomic_number_contrast"], 0.0)
        self.assertGreater(features["site_atomic_number_contrast_mean"], 0.0)

    def test_representation_invariance(self):
        reference = compute_descriptor(self.binary)
        lattice = np.asarray(self.binary.lattice.matrix, dtype=np.float64)
        fractions = np.asarray(self.binary.frac_coords, dtype=np.float64)
        species = [site.specie for site in self.binary]

        translated = Structure(
            Lattice(lattice),
            species,
            np.mod(fractions + np.asarray([0.17, 0.29, 0.41]), 1.0),
            coords_are_cartesian=False,
        )
        permuted = Structure(
            Lattice(lattice),
            list(reversed(species)),
            fractions[::-1],
            coords_are_cartesian=False,
        )
        angle = 0.61
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotated = Structure(
            Lattice(lattice @ rotation),
            species,
            fractions,
            coords_are_cartesian=False,
        )
        supercell = self.binary.copy()
        supercell.make_supercell([2, 1, 1])

        for changed in (translated, permuted, rotated, supercell):
            self.assert_features_close(reference, compute_descriptor(changed))

    def test_uniform_scale_changes_only_absolute_distance(self):
        reference = compute_descriptor(self.binary)
        factor = 1.08
        scaled = Structure(
            Lattice(np.asarray(self.binary.lattice.matrix) * factor),
            [site.specie for site in self.binary],
            self.binary.frac_coords,
            coords_are_cartesian=False,
        )
        observed = compute_descriptor(scaled)
        self.assertAlmostEqual(
            observed["features"]["knn_distance_mean_angstrom"],
            reference["features"]["knn_distance_mean_angstrom"] * factor,
            delta=2.0e-8,
        )
        for name in FEATURE_NAMES:
            if name != "knn_distance_mean_angstrom":
                self.assertAlmostEqual(
                    observed["features"][name],
                    reference["features"][name],
                    delta=2.0e-8,
                    msg=name,
                )

    def test_same_composition_geometry_change_affects_coupling(self):
        distorted = Structure(
            Lattice.cubic(4.0),
            ["Mg", "Zn"],
            [[0.0, 0.0, 0.0], [0.43, 0.47, 0.5]],
        )
        regular = compute_descriptor(self.binary)["features"]
        changed = compute_descriptor(distorted)["features"]
        self.assertNotAlmostEqual(
            regular["site_contrast_radial_coupling_mean"],
            changed["site_contrast_radial_coupling_mean"],
            delta=1.0e-9,
        )

    def test_json_record_and_poscar_cli(self):
        record = {
            "lattice": self.binary.lattice.matrix.tolist(),
            "species": [site.specie.symbol for site in self.binary],
            "fractional_coordinates": self.binary.frac_coords.tolist(),
        }
        descriptor = descriptor_from_record(record)
        self.assertEqual(descriptor["schema"], "element-conditioned-local-geometry/1")
        self.assertEqual(tuple(descriptor["features"]), FEATURE_NAMES)

        poscar_text = """binary\n1.0\n4 0 0\n0 4 0\n0 0 4\nMg Zn\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            poscar = root / "POSCAR"
            output = root / "descriptor.json"
            poscar.write_text(poscar_text, encoding="utf-8")
            self.assertEqual(main([str(poscar), "--output", str(output)]), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["descriptor"]["schema"], descriptor["schema"])


if __name__ == "__main__":
    unittest.main()
