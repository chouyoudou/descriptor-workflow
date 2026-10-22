import json
import math
import pathlib
import tempfile
import unittest

from descriptors.site_void_coupling import (
    DescriptorConfig,
    DescriptorError,
    Structure,
    analyze_site,
    analyze_structure,
    enumerate_neighbor_images,
    parse_poscar,
    run_synthetic_benchmark,
    structure_from_json_record,
)


def cage_structure(*, open_side=False, center_shift=0.0):
    length = 12.0
    lattice = ((length, 0.0, 0.0), (0.0, length, 0.0), (0.0, 0.0, length))

    def frac(x, y, z):
        return (0.5 + x / length, 0.5 + y / length, 0.5 + z / length)

    plus_x = 4.8 if open_side else 2.4
    coordinates = (
        frac(center_shift, 0.0, 0.0),
        frac(plus_x, 0.0, 0.0),
        frac(-2.4, 0.0, 0.0),
        frac(0.0, 2.4, 0.0),
        frac(0.0, -2.4, 0.0),
        frac(0.0, 0.0, 2.4),
        frac(0.0, 0.0, -2.4),
    )
    return Structure(
        lattice=lattice,
        species=("Ca",) + ("O",) * 6,
        frac_coords=coordinates,
        comment="open" if open_side else "symmetric",
    )


class SiteVoidCouplingTests(unittest.TestCase):
    def test_symmetric_octahedral_cage_is_zero_coupling(self):
        result = analyze_site(cage_structure(), 0)
        self.assertEqual(result.neighbor_count, 6)
        self.assertAlmostEqual(result.offcenter_fraction, 0.0, places=12)
        self.assertAlmostEqual(result.void_axis_uniqueness, 0.0, places=12)
        self.assertAlmostEqual(result.clearance_contrast, 0.0, places=12)
        self.assertAlmostEqual(result.site_void_coupling_signed, 0.0, places=12)

    def test_open_cage_and_shift_give_positive_coupling(self):
        result = analyze_site(
            cage_structure(open_side=True, center_shift=0.2),
            0,
            candidate_elements=("Al", "Ga"),
        )
        self.assertEqual(result.neighbor_count, 5)
        self.assertGreater(result.offcenter_fraction, 0.2)
        self.assertGreater(result.void_axis[0], 0.99)
        self.assertGreater(result.clearance_contrast, 0.25)
        self.assertGreater(result.offcenter_void_alignment, 0.99)
        self.assertGreater(result.site_void_coupling_signed, 0.03)
        self.assertTrue(result.locally_large_radius_site)
        self.assertGreater(result.center_to_neighbor_radius_ratio, 2.0)
        projections = {row.candidate: row for row in result.candidate_radius_projections}
        self.assertGreater(
            projections["Al"].minimum_radial_margin_angstrom,
            result.minimum_radial_margin_angstrom,
        )

    def test_translation_invariance(self):
        structure = cage_structure(open_side=True, center_shift=0.2)
        shifted = structure.translated((0.213, -0.377, 0.419))
        first = analyze_site(structure, 0)
        second = analyze_site(shifted, 0)
        for name in (
            "offcenter_fraction",
            "void_axis_uniqueness",
            "clearance_contrast",
            "site_void_coupling_signed",
            "radius_mismatch_fraction",
        ):
            self.assertAlmostEqual(getattr(first, name), getattr(second, name), places=11)

    def test_rigid_rotation_invariance(self):
        structure = cage_structure(open_side=True, center_shift=0.2)
        angle = 0.61
        cosine, sine = math.cos(angle), math.sin(angle)

        def rotate(vector):
            x, y, z = vector
            return (cosine * x - sine * y, sine * x + cosine * y, z)

        rotated = Structure(
            lattice=tuple(rotate(vector) for vector in structure.lattice),
            species=structure.species,
            frac_coords=structure.frac_coords,
            comment="rotated",
        )
        first = analyze_site(structure, 0)
        second = analyze_site(rotated, 0)
        for name in (
            "offcenter_fraction",
            "angular_void_anisotropy",
            "void_axis_uniqueness",
            "clearance_contrast",
            "site_void_coupling_signed",
        ):
            self.assertAlmostEqual(getattr(first, name), getattr(second, name), places=11)

    def test_skew_cell_neighbor_enumeration_includes_periodic_images(self):
        structure = Structure(
            lattice=((2.0, 0.0, 0.0), (1.7, 1.2, 0.0), (0.2, 0.3, 2.1)),
            species=("Si",),
            frac_coords=((0.0, 0.0, 0.0),),
        )
        neighbors = enumerate_neighbor_images(structure, 0, 2.2)
        self.assertGreaterEqual(len(neighbors), 4)
        self.assertTrue(all(row.atom_index == 0 for row in neighbors))
        self.assertTrue(all(row.translation != (0, 0, 0) for row in neighbors))

    def test_parse_direct_selective_dynamics_and_potcar_labels(self):
        text = """demo
1.0
12 0 0
0 12 0
0 0 12
Ca_sv O
1 6
Selective dynamics
Direct
0.5 0.5 0.5 T T T
0.7 0.5 0.5 F F F
0.3 0.5 0.5 F F F
0.5 0.7 0.5 F F F
0.5 0.3 0.5 F F F
0.5 0.5 0.7 F F F
0.5 0.5 0.3 F F F
"""
        structure = parse_poscar(text)
        self.assertEqual(structure.species[0], "Ca")
        self.assertEqual(structure.atom_count, 7)
        self.assertAlmostEqual(structure.volume_angstrom3, 1728.0)

    def test_negative_scale_is_target_volume(self):
        text = """negative scale
-64
1 0 0
0 1 0
0 0 1
Si
1
Cartesian
0.5 0.5 0.5
"""
        structure = parse_poscar(text)
        self.assertAlmostEqual(structure.volume_angstrom3, 64.0, places=10)
        self.assertAlmostEqual(structure.frac_coords[0][0], 0.5, places=10)

    def test_vasp4_without_symbols_is_rejected(self):
        text = """bad
1
1 0 0
0 1 0
0 0 1
1
Direct
0 0 0
"""
        with self.assertRaisesRegex(DescriptorError, "VASP 4"):
            parse_poscar(text)

    def test_structure_aggregation_and_site_filter(self):
        structure = cage_structure(open_side=True, center_shift=0.2)
        result = analyze_structure(structure, center_elements=("Ca",))
        self.assertEqual(result["schema"], "site-void-coupling/1")
        self.assertEqual(result["scalars"]["site_count"], 1)
        self.assertEqual(result["structure"]["selected_site_indices"], [0])
        self.assertIn("not pressure derivatives", result["interpretation_scope"])

    def test_json_record_adapter_preserves_row_identity(self):
        record = {
            "id": "fixture-1",
            "lattice": [[12, 0, 0], [0, 12, 0], [0, 0, 12]],
            "species": ["Ca", "O", "O", "O", "O"],
            "fractional_coordinates": [
                [0.5, 0.5, 0.5],
                [0.7, 0.5, 0.5],
                [0.3, 0.5, 0.5],
                [0.5, 0.7, 0.5],
                [0.5, 0.3, 0.5],
            ],
        }
        structure = structure_from_json_record(record)
        self.assertEqual(structure.comment, "fixture-1")
        self.assertEqual(structure.atom_count, 5)
        self.assertEqual(structure.species[0], "Ca")

    def test_benchmark_is_deterministic_and_writes_complete_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first = run_synthetic_benchmark(
                32,
                output_jsonl=root / "first.jsonl",
                summary_path=root / "first-summary.json",
            )
            second = run_synthetic_benchmark(
                32,
                output_jsonl=root / "second.jsonl",
                summary_path=root / "second-summary.json",
            )
            self.assertEqual(first["structure_count"], 32)
            self.assertEqual(first["finite_scalar_rows"], 32)
            self.assertEqual(first["row_sha256"], second["row_sha256"])
            rows = (root / "first.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 32)
            decoded = json.loads(rows[0])
            self.assertIn("site", decoded)
            self.assertIn("scalars", decoded)

    def test_actions_qualification_1200_structures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            summary = run_synthetic_benchmark(
                1200,
                output_jsonl=root / "qualification.jsonl",
                summary_path=root / "qualification-summary.json",
            )
            self.assertEqual(summary["structure_count"], 1200)
            self.assertEqual(summary["finite_scalar_rows"], 1200)
            self.assertEqual(
                summary["row_sha256"],
                "f7ffb41465d80fb99eead13fa6df9e84195cf08ba2a1c805698d8fe5a7d2994f",
            )
            print("SITE_VOID_QUALIFICATION=" + json.dumps(summary, sort_keys=True))

    def test_invalid_protocol_is_rejected(self):
        with self.assertRaisesRegex(DescriptorError, "exceed clearance cap"):
            DescriptorConfig(search_radius_angstrom=4.0, clearance_cap_angstrom=5.0).validate()


if __name__ == "__main__":
    unittest.main()
