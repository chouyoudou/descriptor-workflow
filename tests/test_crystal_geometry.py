import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from crystal_geometry import cell_volume, volume_per_atom


ROOT = Path(__file__).resolve().parents[1]


class CellGeometryTests(unittest.TestCase):
    def test_cubic_cell(self):
        lattice = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        self.assertEqual(cell_volume(lattice), 8)
        self.assertEqual(volume_per_atom(lattice, 4), 2)

    def test_nonorthogonal_cell(self):
        self.assertEqual(cell_volume([[3, 0, 0], [1, 2, 0], [0.5, 0.25, 4]]), 24)

    def test_basis_orientation(self):
        self.assertEqual(cell_volume([[-2, 0, 0], [0, 2, 0], [0, 0, 2]]), 8)

    def test_volume_preserving_shear(self):
        self.assertEqual(cell_volume([[2, 20, 0], [0, 2, 0], [0, 0, 2]]), 8)

    def test_supercell_normalization(self):
        original = volume_per_atom([[2, 0, 0], [0, 2, 0], [0, 0, 2]], 4)
        repeated = volume_per_atom([[4, 0, 0], [0, 2, 0], [0, 0, 2]], 8)
        self.assertEqual(original, repeated)

    def test_uniform_scaling(self):
        self.assertEqual(volume_per_atom([[4, 0, 0], [0, 4, 0], [0, 0, 4]], 4), 16)

    def test_invalid_cells(self):
        for lattice in (
            [[1, 0, 0], [2, 0, 0], [0, 0, 1]],
            [[math.inf, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0], [0, 1]],
        ):
            with self.assertRaises(ValueError):
                cell_volume(lattice)

    def test_invalid_atom_count(self):
        for n in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                volume_per_atom([[1, 0, 0], [0, 1, 0], [0, 0, 1]], n)

    def run_cli(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_text(text, encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(ROOT / "crystal_geometry.py"), str(path)],
                text=True, capture_output=True, check=False)

    def test_cli_bad_json_is_concise(self):
        proc = self.run_cli('{"broken":')
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, "error: invalid JSON input\n")
        self.assertNotIn("Traceback", proc.stderr)

    def test_cli_invalid_cell_is_concise(self):
        proc = self.run_cli(json.dumps([
            {"id": "bad", "lattice": [[1, 0, 0], [2, 0, 0], [0, 0, 1]], "atom_count": 1}
        ]))
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, "error: cell must have a finite positive volume\n")
        self.assertNotIn("Traceback", proc.stderr)

    def test_cli_example_cells(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "crystal_geometry.py"), str(ROOT / "examples" / "cells.json")],
            text=True, capture_output=True, check=False)
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stderr, "")
        result = json.loads(proc.stdout)
        self.assertEqual([row["id"] for row in result], ["cubic", "orthorhombic", "skew"])
        self.assertEqual([row["volume_angstrom3"] for row in result], [8, 60, 24.0])
        self.assertEqual([row["volume_per_atom_angstrom3"] for row in result], [2.0, 7.5, 6.0])


if __name__ == "__main__":
    unittest.main()
