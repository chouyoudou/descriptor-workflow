from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import elemental_reference_heat_capacity as descriptor


class ElementalReferenceHeatCapacityTests(unittest.TestCase):
    def test_pure_lithium_reference_not_compound_claim(self):
        result = descriptor.compute_descriptor(
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            ["Li"],
        )
        features = result["features"]
        self.assertEqual(result["schema"], descriptor.SCHEMA)
        self.assertAlmostEqual(features["ref_cp_stp_mean_j_mol_atom_k"], 24.86, places=12)
        self.assertEqual(features["ref_cp_stp_std_j_mol_atom_k"], 0.0)
        self.assertEqual(features["ref_cp_stp_range_j_mol_atom_k"], 0.0)
        self.assertEqual(features["ref_cp_stp_pairwise_abs_j_mol_atom_k"], 0.0)
        self.assertEqual(features["ref_cp_stp_relative_pairwise_abs"], 0.0)
        expected_density = 24.86 * 1.0e24 / (descriptor.AVOGADRO_CONSTANT_MOL_INV * 27.0)
        self.assertAlmostEqual(
            features["ref_cp_stp_density_proxy_mj_m3_k"], expected_density, places=12
        )
        limits = " ".join(result["metadata"]["semantic_limits"]).lower()
        self.assertIn("not the actual heat capacity", limits)
        self.assertIn("no phonons", limits)

    def test_binary_closed_form_and_order_invariance(self):
        lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
        forward = descriptor.compute_descriptor(lattice, ["Li", "Cu"])
        reverse = descriptor.compute_descriptor(lattice, ["Cu", "Li"])
        self.assertEqual(forward["features"], reverse["features"])
        features = forward["features"]
        mean = (24.86 + 24.44) / 2.0
        self.assertAlmostEqual(features["ref_cp_stp_mean_j_mol_atom_k"], mean, places=12)
        self.assertAlmostEqual(features["ref_cp_stp_std_j_mol_atom_k"], 0.21, places=12)
        self.assertAlmostEqual(features["ref_cp_stp_range_j_mol_atom_k"], 0.42, places=12)
        self.assertAlmostEqual(features["ref_cp_stp_pairwise_abs_j_mol_atom_k"], 0.21, places=12)
        self.assertAlmostEqual(features["ref_cp_stp_relative_pairwise_abs"], 0.21 / mean, places=12)

    def test_integer_supercell_invariance(self):
        primitive = descriptor.compute_descriptor(
            [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]], ["Li", "Cu"]
        )
        doubled = descriptor.compute_descriptor(
            [[6.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
            ["Li", "Cu", "Li", "Cu"],
        )
        for name in descriptor.FEATURE_NAMES:
            self.assertAlmostEqual(primitive["features"][name], doubled["features"][name], places=12)

    def test_record_validation_and_fractional_coordinates(self):
        record = {
            "id": "example",
            "lattice": [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            "species": ["Li", "Cu"],
            "fractional_coordinates": [[0, 0, 0], [0.5, 0.5, 0.5]],
        }
        result = descriptor.descriptor_from_record(record)
        self.assertEqual(result["metadata"]["element_counts"], {"Cu": 1, "Li": 1})
        record["fractional_coordinates"] = [[0, 0, 0]]
        with self.assertRaises(descriptor.DescriptorUnavailable) as context:
            descriptor.descriptor_from_record(record)
        self.assertEqual(context.exception.code, "coordinate_count_mismatch")

    def test_missing_reference_is_typed_and_not_imputed(self):
        with self.assertRaises(descriptor.DescriptorUnavailable) as context:
            descriptor.compute_descriptor([[3, 0, 0], [0, 3, 0], [0, 0, 3]], ["Li", "Tc"])
        self.assertEqual(context.exception.category, "definition_domain")
        self.assertEqual(context.exception.code, "missing_element_reference")
        self.assertIn("strict policy forbids imputation", context.exception.message)

    def test_aliases_and_oxidation_labels_are_not_silently_stripped(self):
        for value in ("li", "Fe2+", "D", "X"):
            with self.subTest(value=value):
                with self.assertRaises(descriptor.DescriptorUnavailable) as context:
                    descriptor.compute_descriptor([[3, 0, 0], [0, 3, 0], [0, 0, 3]], [value])
                self.assertIn(context.exception.code, {"invalid_element_symbol", "unknown_element_symbol"})

    def test_singular_and_ill_conditioned_cells(self):
        with self.assertRaises(descriptor.DescriptorUnavailable) as context:
            descriptor.compute_descriptor([[1, 0, 0], [2, 0, 0], [0, 0, 1]], ["Li"])
        self.assertEqual(context.exception.code, "singular_lattice")
        with self.assertRaises(descriptor.DescriptorUnavailable) as context:
            descriptor.compute_descriptor([[1.0e7, 0, 0], [0, 1.0, 0], [0, 0, 1.0e-7]], ["Li"])
        self.assertIn(context.exception.code, {"singular_lattice", "ill_conditioned_lattice"})

    def test_poscar_v5_and_negative_volume_scale(self):
        positive = """Li Cu
1.0
4 0 0
0 4 0
0 0 4
Li Cu
1 1
Selective dynamics
Direct
0 0 0 T T T
0.5 0.5 0.5 F F F
"""
        result = descriptor.descriptor_from_poscar_text(positive)
        self.assertAlmostEqual(result["metadata"]["cell_volume_angstrom3"], 64.0)
        negative = """Li
-27.0
1 0 0
0 1 0
0 0 1
Li
1
Cartesian
0 0 0
"""
        result = descriptor.descriptor_from_poscar_text(negative)
        self.assertAlmostEqual(result["metadata"]["cell_volume_angstrom3"], 27.0)

    def test_vasp4_without_symbols_is_rejected(self):
        text = """legacy
1.0
3 0 0
0 3 0
0 0 3
1
Direct
0 0 0
"""
        with self.assertRaises(descriptor.DescriptorUnavailable) as context:
            descriptor.descriptor_from_poscar_text(text)
        self.assertEqual(context.exception.code, "missing_poscar_species_symbols")

    def test_source_metadata_is_pinned_and_explicit(self):
        metadata = descriptor.REFERENCE_METADATA
        self.assertEqual(metadata["upstream_version"], "0.10.1")
        self.assertEqual(metadata["upstream_commit"], "25cd8f2778d73e54f80b38ee1983a28643368118")
        self.assertEqual(metadata["upstream_license"], "BSD-3-Clause")
        self.assertIn("STP", metadata["state_and_temperature"])
        self.assertEqual(metadata["units"], "J mol(element atoms)^-1 K^-1")
        self.assertIn("strict", metadata["missing_policy"])

    def test_jsonl_cli_preserves_rows_and_typed_failures(self):
        valid = {
            "id": "ok",
            "lattice": [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            "species": ["Li"],
            "fractional_coordinates": [[0, 0, 0]],
        }
        missing = {
            "id": "missing",
            "lattice": [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            "species": ["Tc"],
            "fractional_coordinates": [[0, 0, 0]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "structures.jsonl"
            input_path.write_text(json.dumps(valid) + "\n" + json.dumps(missing) + "\n", encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "elemental_reference_heat_capacity.py"), str(input_path), "--format", "jsonl"],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 1)
        rows = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual([row["id"] for row in rows], ["ok", "missing"])
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[1]["status"], "unavailable")
        self.assertEqual(rows[1]["error"]["code"], "missing_element_reference")


if __name__ == "__main__":
    unittest.main()
