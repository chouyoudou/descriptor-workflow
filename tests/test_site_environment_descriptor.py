from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:  # Repository-wide tests intentionally install no extras.
    raise unittest.SkipTest(
        "site_environment_descriptor tests require the task-pinned NumPy runtime"
    ) from exc

from site_environment_descriptor import (
    FEATURE_NAMES,
    DescriptorConfig,
    PeriodicStructure,
    ResourceLimitError,
    compute_site_environments,
    evaluate_probe_elements,
    featurize_structure,
    parse_poscar_text,
    smooth_contact_weight,
    write_poscar,
)
from site_environment_descriptor.cli import main as cli_main


def _assert_results_close(
    case: unittest.TestCase,
    left,
    right,
    *,
    atol: float = 1.0e-9,
) -> None:
    case.assertEqual(left.feature_names, right.feature_names)
    for name, a, b in zip(left.feature_names, left.values, right.values):
        case.assertEqual(a is None, b is None, name)
        if a is not None:
            case.assertAlmostEqual(float(a), float(b), delta=atol, msg=name)


def _octahedral_cage() -> PeriodicStructure:
    lattice = np.diag([8.0, 8.0, 8.0])
    center = np.array([0.5, 0.5, 0.5])
    directions = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    cart = [center @ lattice] + [center @ lattice + vector * 2.4 for vector in directions]
    frac = np.asarray(cart) @ np.linalg.inv(lattice)
    return PeriodicStructure(lattice, ("La",) + ("O",) * 6, frac, "LaO6 test cage")


class PoscarTests(unittest.TestCase):
    def test_direct_cartesian_three_scale_and_negative_volume(self) -> None:
        direct = """direct
1.0
1 0 0
0 1 0
0 0 1
C O
1 1
Direct
0.25 0.50 0.75
0.00 0.00 0.00
"""
        structure = parse_poscar_text(direct)
        np.testing.assert_allclose(structure.frac_coords[0], [0.25, 0.5, 0.75])

        cartesian = """cart
2 3 4
1 0 0
0 1 0
0 0 1
C
1
Selective dynamics
Cartesian
0.5 0.5 0.5 T F T
"""
        structure = parse_poscar_text(cartesian)
        np.testing.assert_allclose(structure.lattice, np.diag([2.0, 3.0, 4.0]))
        np.testing.assert_allclose(structure.frac_coords[0], [0.5, 0.5, 0.5])

        negative = """negative volume
-64
1 0 0
0 1 0
0 0 1
C
1
Direct
0 0 0
"""
        structure = parse_poscar_text(negative)
        self.assertAlmostEqual(structure.volume, 64.0, places=10)

    def test_vasp4_requires_explicit_species(self) -> None:
        text = """vasp4
1
4 0 0
0 4 0
0 0 4
1 2
Direct
0 0 0
0.25 0.25 0.25
0.75 0.75 0.75
"""
        with self.assertRaisesRegex(ValueError, "species_override"):
            parse_poscar_text(text)
        structure = parse_poscar_text(text, species_override=["Na", "Cl"])
        self.assertEqual(structure.species, ("Na", "Cl", "Cl"))

    def test_roundtrip_grouping(self) -> None:
        structure = PeriodicStructure(
            np.diag([4.0, 5.0, 6.0]),
            ("O", "La", "O"),
            np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5], [0.8, 0.7, 0.6]]),
        )
        text = write_poscar(structure)
        parsed = parse_poscar_text(text)
        self.assertEqual(parsed.species, ("O", "O", "La"))
        expected = structure.frac_coords[[0, 2, 1]]
        np.testing.assert_allclose(parsed.frac_coords, expected)


class GeometryTests(unittest.TestCase):
    def test_contact_gate_endpoints(self) -> None:
        cfg = DescriptorConfig()
        self.assertEqual(smooth_contact_weight(cfg.normalized_inner - 0.1, cfg), 1.0)
        self.assertEqual(smooth_contact_weight(cfg.normalized_outer, cfg), 0.0)
        midpoint = (cfg.normalized_inner + cfg.normalized_outer) / 2.0
        self.assertAlmostEqual(smooth_contact_weight(midpoint, cfg), 0.5)

    def test_simple_cubic_is_isotropic(self) -> None:
        structure = PeriodicStructure(
            np.eye(3) * 1.5,
            ("C",),
            np.array([[0.0, 0.0, 0.0]]),
        )
        env = compute_site_environments(structure)[0]
        self.assertTrue(env.valid)
        self.assertGreater(env.channels["effective_coordination"], 5.7)
        self.assertLess(env.channels["effective_coordination"], 6.01)
        self.assertAlmostEqual(env.channels["offcentering"], 0.0, places=12)
        self.assertAlmostEqual(env.channels["orientation_anisotropy"], 0.0, places=12)
        self.assertAlmostEqual(env.channels["orientation_linearity"], 0.0, places=12)

    def test_chain_is_linear_and_layer_is_planar(self) -> None:
        chain = PeriodicStructure(
            np.diag([1.5, 8.0, 8.0]),
            ("C",),
            np.array([[0.17, 0.33, 0.71]]),
        )
        chain_env = compute_site_environments(chain)[0]
        self.assertAlmostEqual(chain_env.channels["orientation_anisotropy"], 1.0, places=12)
        self.assertAlmostEqual(chain_env.channels["orientation_linearity"], 1.0, places=12)
        self.assertAlmostEqual(chain_env.channels["orientation_planarity"], 0.0, places=12)

        layer = PeriodicStructure(
            np.diag([1.5, 1.5, 8.0]),
            ("C",),
            np.array([[0.27, 0.43, 0.68]]),
        )
        layer_env = compute_site_environments(layer)[0]
        self.assertAlmostEqual(layer_env.channels["orientation_linearity"], 0.0, places=12)
        self.assertAlmostEqual(layer_env.channels["orientation_planarity"], 0.5, places=12)
        self.assertAlmostEqual(layer_env.channels["orientation_anisotropy"], 0.25, places=12)

    def test_heteroelement_channels_and_species_separation(self) -> None:
        result = featurize_structure(_octahedral_cage())
        features = result.as_dict()
        self.assertEqual(features["unique_element_count"], 2.0)
        self.assertEqual(features["multi_species_structure"], 1.0)
        self.assertGreater(features["heteroelement_contact_weight_fraction"], 0.99)
        self.assertGreater(features["local_element_effective_count_mean"], 0.99)
        self.assertIsNotNone(features["species_eta2_effective_coordination"])
        self.assertGreater(features["species_eta2_effective_coordination"], 0.9)

    def test_isolated_site_has_typed_nulls(self) -> None:
        structure = PeriodicStructure(
            np.eye(3) * 20.0,
            ("C",),
            np.array([[0.0, 0.0, 0.0]]),
        )
        result = featurize_structure(structure)
        features = result.as_dict()
        self.assertEqual(features["environment_valid_fraction"], 0.0)
        self.assertEqual(features["environment_missing_fraction"], 1.0)
        self.assertEqual(features["contact_weight_per_site_mean"], 0.0)
        self.assertIsNone(features["effective_coordination_mean"])
        self.assertIsNone(features["heteroelement_contact_weight_fraction"])
        self.assertEqual(result.metadata["missing_reasons"], {"no_weighted_contacts": 1})

    def test_overlap_and_resource_guards_are_explicit(self) -> None:
        overlapping = PeriodicStructure(
            np.eye(3) * 5.0,
            ("C", "O"),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )
        with self.assertRaisesRegex(ValueError, "overlapping_periodic_sites"):
            featurize_structure(overlapping)

        with self.assertRaises(ResourceLimitError):
            featurize_structure(
                _octahedral_cage(),
                DescriptorConfig(max_neighbor_evaluations=1),
            )


class ContractAndInvarianceTests(unittest.TestCase):
    def test_fixed_contract(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), 94)
        self.assertEqual(len(set(FEATURE_NAMES)), 94)
        result = featurize_structure(_octahedral_cage())
        self.assertEqual(result.feature_names, FEATURE_NAMES)
        self.assertEqual(len(result.values), 94)
        json.dumps(result.to_jsonable(), allow_nan=False)

    def test_translation_permutation_rotation_and_supercell(self) -> None:
        structure = _octahedral_cage()
        reference = featurize_structure(structure)
        translated = featurize_structure(structure.translated([1.37, -2.41, 0.73]))
        _assert_results_close(self, reference, translated, atol=2.0e-10)

        order = [6, 0, 4, 2, 5, 1, 3]
        permuted = featurize_structure(structure.reordered(order))
        _assert_results_close(self, reference, permuted, atol=2.0e-10)

        angle = 0.617
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotated = featurize_structure(structure.rotated(rotation))
        _assert_results_close(self, reference, rotated, atol=2.0e-10)

        expanded = featurize_structure(structure.supercell((2, 1, 1)))
        _assert_results_close(self, reference, expanded, atol=2.0e-10)

    def test_probe_extension_is_explicit_and_separate(self) -> None:
        structure = _octahedral_cage()
        core_before = featurize_structure(structure)
        probes = evaluate_probe_elements(structure, ["Cr", "Eu", "Cr"])
        core_after = featurize_structure(structure)
        _assert_results_close(self, core_before, core_after)
        self.assertEqual(tuple(probes.probes), ("Cr", "Eu"))
        self.assertGreater(probes.probes["Cr"]["valid_cage_count"], 0)
        self.assertIsNotNone(probes.probes["Eu"]["absolute_residual_mean"])
        json.dumps(probes.to_jsonable(), allow_nan=False)

    def test_cli_emits_strict_json(self) -> None:
        structure = _octahedral_cage()
        with tempfile.TemporaryDirectory() as tmp:
            poscar = Path(tmp) / "POSCAR"
            output = Path(tmp) / "result.json"
            write_poscar(structure, poscar)
            code = cli_main(
                [str(poscar), "--probes", "Cr,Eu", "--output", str(output)]
            )
            self.assertEqual(code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["descriptor"]["schema"], "periodic-site-environment/1")
            self.assertEqual(
                payload["explicit_probes"]["schema"],
                "explicit-frozen-cage-probes/1",
            )


if __name__ == "__main__":
    unittest.main()
