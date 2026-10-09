import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_bridge_geometry import (
    DESCRIPTOR_NAMES,
    ElementData,
    Structure,
    analyze_cutoff_ensemble,
    analyze_structure,
    build_periodic_bonds,
    default_element_table,
    parse_poscar,
)


class LayerBridgeGeometryTests(unittest.TestCase):
    @staticmethod
    def layered_fixture(include_guest=False):
        species = ("Ni", "S", "N")
        coords = ((0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5))
        if include_guest:
            species += ("He",)
            coords += ((0.0, 0.0, 0.0),)
        return Structure(
            ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 8.0)),
            species,
            coords,
            "synthetic layer fixture",
        )

    def assert_descriptor_maps_close(self, left, right, places=10):
        self.assertEqual(tuple(left), DESCRIPTOR_NAMES)
        self.assertEqual(tuple(right), DESCRIPTOR_NAMES)
        for name in DESCRIPTOR_NAMES:
            self.assertAlmostEqual(left[name], right[name], places=places, msg=name)

    def test_known_periodic_component_ranks(self):
        cells = {
            0: ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
            1: ((1.4, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
            2: ((1.4, 0.0, 0.0), (0.0, 1.4, 0.0), (0.0, 0.0, 8.0)),
            3: ((1.4, 0.0, 0.0), (0.0, 1.4, 0.0), (0.0, 0.0, 1.4)),
        }
        for expected, lattice in cells.items():
            with self.subTest(expected=expected):
                result = analyze_structure(Structure(lattice, ("C",), ((0.23, 0.41, 0.67),)))
                self.assertEqual(result["descriptors"]["framework_dimensionality_max"], expected)
                self.assertEqual(
                    result["diagnostics"]["component_dimensionalities"][0]["dimensionality"],
                    expected,
                )
        layered = analyze_structure(Structure(cells[2], ("C",), ((0.0, 0.0, 0.0),)))
        self.assertAlmostEqual(layered["descriptors"]["layer_repeat_angstrom"], 8.0)
        self.assertAlmostEqual(layered["descriptors"]["interlayer_clearance_angstrom"], 6.48)

    def test_skew_layer_uses_reciprocal_normal_repeat(self):
        structure = Structure(
            ((1.4, 0.0, 0.0), (0.4, 1.34, 0.0), (1.1, 0.7, 8.0)),
            ("C",),
            ((0.37, 0.19, 0.61),),
        )
        result = analyze_structure(structure)
        self.assertEqual(result["descriptors"]["framework_dimensionality_max"], 2.0)
        self.assertEqual(result["diagnostics"]["dominant_layer_hkl"], [0, 0, 1])
        self.assertAlmostEqual(result["descriptors"]["layer_repeat_angstrom"], 8.0)
        self.assertAlmostEqual(result["descriptors"]["interlayer_clearance_angstrom"], 6.48)

    def test_layer_coordination_and_bridge_descriptors(self):
        result = analyze_structure(self.layered_fixture())
        values = result["descriptors"]
        self.assertEqual(tuple(values), DESCRIPTOR_NAMES)
        self.assertEqual(values["framework_dimensionality_max"], 2.0)
        self.assertEqual(values["layered_atom_fraction"], 1.0)
        self.assertAlmostEqual(values["layer_repeat_angstrom"], 8.0)
        self.assertAlmostEqual(values["interlayer_clearance_angstrom"], 5.52)
        self.assertAlmostEqual(values["metal_coordination_mean"], 4.0)
        self.assertAlmostEqual(values["bridge_path_incidence_per_metal"], 4.0)
        self.assertAlmostEqual(values["bridge_internal_atoms_mean"], 1.0)
        self.assertAlmostEqual(values["bridge_straightness_mean"], 1.0)
        self.assertAlmostEqual(values["single_atom_bridge_angle_mean_deg"], 180.0)
        self.assertFalse(result["diagnostics"]["bridge_path_truncated"])
        self.assertIn("static structure proxy only", result["diagnostics"]["scope_warning"])

    def test_explicit_gap_guest_reduces_clearance(self):
        empty = analyze_structure(self.layered_fixture())["descriptors"]
        occupied = analyze_structure(self.layered_fixture(include_guest=True))["descriptors"]
        self.assertAlmostEqual(empty["interlayer_clearance_angstrom"], 5.52)
        self.assertAlmostEqual(occupied["interlayer_clearance_angstrom"], 2.48)
        self.assertLess(
            occupied["interlayer_clearance_angstrom"], empty["interlayer_clearance_angstrom"]
        )
        self.assertLess(occupied["layered_atom_fraction"], 1.0)

    def test_atom_order_translation_rotation_and_supercell_invariance(self):
        base = self.layered_fixture()
        reference = analyze_structure(base)["descriptors"]

        permutation = (2, 0, 1)
        reordered = Structure(
            base.lattice,
            tuple(base.species[index] for index in permutation),
            tuple(base.frac_coords[index] for index in permutation),
        )
        shifted = Structure(
            base.lattice,
            base.species,
            tuple(
                tuple((value + delta) % 1.0 for value, delta in zip(coord, (0.173, 0.291, 0.419)))
                for coord in base.frac_coords
            ),
        )

        angle = math.radians(33.0)
        rotation = (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
        rotated_lattice = tuple(
            tuple(sum(row[k] * rotation[j][k] for k in range(3)) for j in range(3))
            for row in base.lattice
        )
        rotated = Structure(rotated_lattice, base.species, base.frac_coords)

        super_lattice = (
            tuple(2.0 * value for value in base.lattice[0]),
            base.lattice[1],
            base.lattice[2],
        )
        super_species = []
        super_coords = []
        for symbol, coord in zip(base.species, base.frac_coords):
            for image in (0, 1):
                super_species.append(symbol)
                super_coords.append(((coord[0] + image) / 2.0, coord[1], coord[2]))
        supercell = Structure(super_lattice, tuple(super_species), tuple(super_coords))

        for label, structure in (
            ("reordered", reordered),
            ("shifted", shifted),
            ("rotated", rotated),
            ("in-plane supercell", supercell),
        ):
            with self.subTest(label=label):
                self.assert_descriptor_maps_close(reference, analyze_structure(structure)["descriptors"])

    def test_bridge_search_limit_is_explicit(self):
        result = analyze_structure(self.layered_fixture(), max_bridge_paths=1)
        self.assertTrue(result["diagnostics"]["bridge_path_truncated"])
        self.assertEqual(result["diagnostics"]["bridge_path_count"], 1)
        self.assertEqual(result["descriptors"]["bridge_path_incidence_per_metal"], 2.0)

    def test_cutoff_ensemble_exposes_topology_sensitivity(self):
        structure = Structure(
            ((1.75, 0.0, 0.0), (0.0, 8.0, 0.0), (0.0, 0.0, 8.0)),
            ("C",),
            ((0.0, 0.0, 0.0),),
        )
        result = analyze_cutoff_ensemble(structure)
        dimensions = [run["descriptors"]["framework_dimensionality_max"] for run in result["runs"]]
        self.assertEqual(dimensions, [0.0, 1.0, 1.0])
        self.assertEqual(
            result["descriptor_ranges"]["framework_dimensionality_max"],
            {"min": 0.0, "max": 1.0, "span": 1.0},
        )

    def test_poscar_parser_direct_cartesian_selective_and_negative_scale(self):
        direct = parse_poscar(
            """fixture
1.0
4 0 0
0 4 0
0 0 8
Ni S N
1 1 1
Selective dynamics
Direct
0 0 0.5 T T T
0.5 0 0.5 F F F
0 0.5 0.5 T F T
"""
        )
        self.assertEqual(direct.species, ("Ni", "S", "N"))
        self.assertEqual(direct.frac_coords[1], (0.5, 0.0, 0.5))

        negative = parse_poscar(
            """negative volume
-64
1 0 0
0 1 0
0 0 1
C
1
Cartesian
0.5 0 0
"""
        )
        self.assertAlmostEqual(negative.lattice[0][0], 4.0)
        self.assertAlmostEqual(negative.frac_coords[0][0], 0.5)

        with self.assertRaisesRegex(ValueError, "explicit VASP 5 element symbols"):
            parse_poscar(
                """VASP4
1
1 0 0
0 1 0
0 0 1
1
Direct
0 0 0
"""
            )

    def test_missing_element_radius_requires_explicit_override(self):
        structure = Structure(
            ((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
            ("Bk",),
            ((0.0, 0.0, 0.0),),
        )
        with self.assertRaisesRegex(ValueError, "missing element data for Bk"):
            analyze_structure(structure)
        table = default_element_table()
        table["Bk"] = ElementData(97, 1.70, True)
        result = analyze_structure(structure, element_table=table)
        self.assertEqual(result["descriptors"]["metal_fraction"], 1.0)

    def test_periodic_bond_graph_is_deterministic(self):
        structure = self.layered_fixture()
        first = build_periodic_bonds(structure)
        second = build_periodic_bonds(structure)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)

    def test_cli_json_and_concise_error(self):
        poscar = ROOT / "examples" / "POSCAR.layer-bridge"
        good = subprocess.run(
            [sys.executable, str(ROOT / "layer_bridge_geometry.py"), str(poscar)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertEqual(good.stderr, "")
        result = json.loads(good.stdout)
        self.assertEqual(result["schema"], "layer-bridge-geometry/1")
        self.assertEqual(tuple(result["descriptors"]), DESCRIPTOR_NAMES)

        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "POSCAR"
            bad_path.write_text("broken\n", encoding="utf-8")
            bad = subprocess.run(
                [sys.executable, str(ROOT / "layer_bridge_geometry.py"), str(bad_path)],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(bad.returncode, 2)
        self.assertEqual(bad.stdout, "")
        self.assertEqual(bad.stderr, "error: POSCAR is too short\n")
        self.assertNotIn("Traceback", bad.stderr)

    def test_frozen_1000_static_structures(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "frozen-1000.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "examples" / "layer_bridge_frozen_1000.py"),
                    "--output",
                    str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["case_count"], 1000)
        self.assertEqual(report["exact_dimensionality_matches"], 1000)
        self.assertEqual(report["finite_descriptor_values"], 21000)
        self.assertEqual(report["layered_positive_clearance_cases"], 250)
        self.assertEqual(
            report["descriptor_sha256"],
            "a43948921b31cdaa17fee170d98597ba536d9018091fd30c75d9c31440fb4d46",
        )
        print(
            "FROZEN_1000 exact=1000/1000 finite=21000 digest="
            + report["descriptor_sha256"]
        )


if __name__ == "__main__":
    unittest.main()
