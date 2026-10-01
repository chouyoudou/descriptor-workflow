from __future__ import annotations

import contextlib
import io
import json
import math
import pathlib
import tempfile
import unittest

from descriptors.bv_directionality import (
    ContactConfig,
    DescriptorError,
    ExplicitBondValenceProtocol,
    PairParameter,
    Structure,
    _cart_to_frac,
    _frac_to_cart,
    analyze_explicit_bond_valence_site,
    analyze_radius_contact_site,
    analyze_structure,
    load_explicit_protocol,
    main,
    parse_poscar,
    run_synthetic_benchmark,
)


SYMMETRIC_POSCAR = """symmetric SnF6 cage
1.0
14 0 0
0 14 0
0 0 14
Sn F
1 6
Direct
0.5 0.5 0.5
0.6428571428571429 0.5 0.5
0.3571428571428571 0.5 0.5
0.5 0.6428571428571429 0.5
0.5 0.3571428571428571 0.5
0.5 0.5 0.6428571428571429
0.5 0.5 0.3571428571428571
"""

OPEN_POSCAR = """one-sided open SnF6 cage
1.0
14 0 0
0 14 0
0 0 14
Sn F
1 6
Direct
0.5 0.5 0.5
0.75 0.5 0.5
0.3571428571428571 0.5 0.5
0.5 0.6428571428571429 0.5
0.5 0.3571428571428571 0.5
0.5 0.5 0.6428571428571429
0.5 0.5 0.3571428571428571
"""


def _two_coordinate_structure(asymmetric: bool = False) -> Structure:
    minus_distance = 3.0 if asymmetric else 2.0
    return Structure(
        lattice=((12.0, 0.0, 0.0), (0.0, 12.0, 0.0), (0.0, 0.0, 12.0)),
        species=("Sn", "F", "F"),
        frac_coords=(
            (0.5, 0.5, 0.5),
            ((6.0 + 2.0) / 12.0, 0.5, 0.5),
            ((6.0 - minus_distance) / 12.0, 0.5, 0.5),
        ),
        comment="two-coordinate SnF2",
    )


def _protocol(structure: Structure) -> ExplicitBondValenceProtocol:
    protocol = ExplicitBondValenceProtocol(
        site_oxidation_states=(2, -1, -1),
        pair_parameters=(
            PairParameter(
                cation="Sn",
                cation_oxidation=2,
                anion="F",
                anion_oxidation=-1,
                r0_angstrom=2.0,
                b_angstrom=0.37,
                source="synthetic unit-test parameter; not a literature value",
            ),
        ),
        candidate_site_indices=(0,),
        minimum_bond_valence=1e-5,
        maximum_distance_angstrom=4.0,
        vector_cutoff=0.5,
        dummy_offset_angstrom=1.0,
    )
    protocol.validate(structure)
    return protocol


class VectorMathTests(unittest.TestCase):
    def test_skew_round_trip(self) -> None:
        lattice = ((2.0, 0.0, 0.0), (0.5, 3.0, 0.0), (0.2, 0.3, 4.0))
        fractional = (0.2, 0.3, 0.4)
        reconstructed = _cart_to_frac(_frac_to_cart(fractional, lattice), lattice)
        for expected, observed in zip(fractional, reconstructed, strict=True):
            self.assertAlmostEqual(expected, observed, places=12)


class PoscarAndContactTests(unittest.TestCase):
    def test_symmetric_cage_has_zero_first_moment(self) -> None:
        structure = parse_poscar(SYMMETRIC_POSCAR)
        site = analyze_radius_contact_site(structure, 0)
        self.assertEqual(site["neighbor_count"], 6)
        self.assertFalse(site["fallback_used"])
        self.assertLess(site["contact_vector_imbalance"], 1e-12)
        self.assertIsNone(site["open_axis_cartesian"])
        self.assertIsNone(site["virtual_point_fractional"])

    def test_open_cage_axis_points_toward_lengthened_contact(self) -> None:
        structure = parse_poscar(OPEN_POSCAR)
        site = analyze_radius_contact_site(structure, 0)
        self.assertGreater(site["contact_vector_imbalance"], 0.10)
        axis = site["open_axis_cartesian"]
        self.assertIsNotNone(axis)
        self.assertGreater(axis[0], 0.99)
        self.assertAlmostEqual(axis[1], 0.0, places=10)
        self.assertAlmostEqual(axis[2], 0.0, places=10)
        self.assertGreater(site["opposed_void_support"], 0.0)

    def test_fractional_translation_invariance(self) -> None:
        structure = parse_poscar(OPEN_POSCAR)
        shifted = Structure(
            lattice=structure.lattice,
            species=structure.species,
            frac_coords=tuple(
                ((x + 0.173) % 1.0, (y + 0.291) % 1.0, (z + 0.407) % 1.0)
                for x, y, z in structure.frac_coords
            ),
            comment="shifted",
        )
        original = analyze_radius_contact_site(structure, 0)
        translated = analyze_radius_contact_site(shifted, 0)
        for key in (
            "contact_vector_imbalance",
            "effective_coordination",
            "mean_normalized_contact_distance",
            "radial_dispersion",
            "cavity_radius_angstrom",
            "radius_fit_mismatch",
            "virtual_point_clearance_angstrom",
            "opposed_void_support",
        ):
            self.assertAlmostEqual(original[key], translated[key], places=10, msg=key)

    def test_rigid_rotation_invariance(self) -> None:
        structure = parse_poscar(OPEN_POSCAR)
        # Apply a 90-degree rotation about z to every lattice row. Fractional
        # coordinates stay unchanged and all scalar descriptors must be invariant.
        def rotate(vector: tuple[float, float, float]) -> tuple[float, float, float]:
            return (-vector[1], vector[0], vector[2])

        rotated = Structure(
            lattice=tuple(rotate(vector) for vector in structure.lattice),
            species=structure.species,
            frac_coords=structure.frac_coords,
            comment="rotated",
        )
        original = analyze_radius_contact_site(structure, 0)
        transformed = analyze_radius_contact_site(rotated, 0)
        for key in (
            "contact_vector_imbalance",
            "effective_coordination",
            "mean_normalized_contact_distance",
            "radial_dispersion",
            "cavity_radius_angstrom",
            "radius_fit_mismatch",
            "virtual_point_clearance_angstrom",
            "opposed_void_support",
        ):
            self.assertAlmostEqual(original[key], transformed[key], places=10, msg=key)
        self.assertAlmostEqual(transformed["open_axis_cartesian"][0], 0.0, places=10)
        self.assertGreater(transformed["open_axis_cartesian"][1], 0.99)

    def test_structure_output_has_fixed_scalar_contract(self) -> None:
        result = analyze_structure(parse_poscar(OPEN_POSCAR), site_indices=(0,))
        self.assertEqual(result["schema"], "bond-valence-directionality/1")
        self.assertIsNone(result["explicit_bond_valence"])
        scalars = result["radius_contact"]["scalars"]
        self.assertEqual(
            set(scalars),
            {
                "analyzed_site_count",
                "contact_vector_imbalance_mean",
                "contact_vector_imbalance_p90",
                "contact_vector_imbalance_max",
                "effective_coordination_mean",
                "radius_fit_mismatch_mean",
                "virtual_point_clearance_mean_angstrom",
                "opposed_void_support_mean",
                "opposed_void_support_max",
                "fallback_site_fraction",
                "zero_vector_site_fraction",
            },
        )


class ExplicitBondValenceTests(unittest.TestCase):
    def test_symmetric_explicit_bvs_matches_closed_form(self) -> None:
        structure = _two_coordinate_structure(asymmetric=False)
        site = analyze_explicit_bond_valence_site(structure, 0, _protocol(structure))
        self.assertEqual(site["parameterized_contribution_count"], 2)
        self.assertAlmostEqual(site["bond_valence_sum"], 2.0, places=12)
        self.assertAlmostEqual(site["bond_valence_sum_mismatch"], 0.0, places=12)
        self.assertAlmostEqual(site["bond_valence_vector_magnitude"], 0.0, places=12)
        self.assertFalse(site["dummy_site_created"])

    def test_asymmetric_explicit_vector_creates_empirical_dummy(self) -> None:
        structure = _two_coordinate_structure(asymmetric=True)
        site = analyze_explicit_bond_valence_site(structure, 0, _protocol(structure))
        expected_weak = math.exp((2.0 - 3.0) / 0.37)
        self.assertAlmostEqual(site["bond_valence_sum"], 1.0 + expected_weak, places=12)
        self.assertAlmostEqual(site["bond_valence_vector_magnitude"], 1.0 - expected_weak, places=12)
        self.assertTrue(site["dummy_site_created"])
        # Stronger bond is in +x, hence the empirical dummy is placed in -x.
        self.assertLess(site["dummy_site_fractional"][0], 0.5)
        self.assertIsNotNone(site["dummy_site_clearance_angstrom"])

    def test_protocol_loader_requires_explicit_sources(self) -> None:
        structure = _two_coordinate_structure()
        data = {
            "schema": "explicit-bond-valence-protocol/1",
            "site_oxidation_states": [2, -1, -1],
            "pair_parameters": [
                {
                    "cation": "Sn",
                    "cation_oxidation": 2,
                    "anion": "F",
                    "anion_oxidation": -1,
                    "r0_angstrom": 2.0,
                    "b_angstrom": 0.37,
                    "source": "",
                }
            ],
        }
        with self.assertRaises(DescriptorError):
            load_explicit_protocol(data, structure)

    def test_combined_output_keeps_layers_separate(self) -> None:
        structure = _two_coordinate_structure(asymmetric=True)
        result = analyze_structure(
            structure,
            site_indices=(0,),
            explicit_protocol=_protocol(structure),
        )
        self.assertIsNotNone(result["explicit_bond_valence"])
        self.assertIn("not_implemented", result["explicit_bond_valence"])
        self.assertNotEqual(
            result["radius_contact"]["sites"][0]["contact_vector_imbalance"],
            result["explicit_bond_valence"]["sites"][0]["bond_valence_vector_imbalance"],
        )


class BenchmarkAndCliTests(unittest.TestCase):
    def test_small_benchmark_is_complete_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            summary_a = run_synthetic_benchmark(
                count_per_family=3,
                seed=20251001,
                output_dir=first,
            )
            summary_b = run_synthetic_benchmark(
                count_per_family=3,
                seed=20251001,
                output_dir=second,
            )
            self.assertEqual(summary_a["successful_structure_count"], 15)
            self.assertEqual(summary_a["failure_count"], 0)
            self.assertEqual(summary_a["finite_row_count"], 15)
            self.assertEqual(summary_a["row_sha256"], summary_b["row_sha256"])
            expected_files = {
                "thousand-run-results.json",
                "thousand-run-failures.csv",
                "thousand-run-sample-map.csv",
                "thousand-run-summary.md",
                "thousand-run-analysis.md",
            }
            self.assertEqual(expected_files, {path.name for path in pathlib.Path(first).iterdir()})

    def test_cli_analyze_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            poscar = pathlib.Path(directory) / "POSCAR"
            poscar.write_text(OPEN_POSCAR, encoding="utf-8")
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                return_code = main(["analyze", str(poscar), "--site-indices", "0"])
            self.assertEqual(return_code, 0)
            payload = json.loads(stream.getvalue())
            self.assertEqual(payload["structure"]["atom_count"], 7)
            self.assertGreater(
                payload["radius_contact"]["scalars"]["contact_vector_imbalance_mean"],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
