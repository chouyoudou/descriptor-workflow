import math
import unittest
from crystal_geometry import cell_volume, volume_per_atom


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


if __name__ == "__main__":
    unittest.main()
