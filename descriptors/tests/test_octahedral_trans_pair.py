from __future__ import annotations

import itertools
import math
import pathlib
import tempfile
import unittest

import numpy as np

from descriptors.octahedral_trans_pair import (
    DescriptorError,
    Settings,
    StructureData,
    analyze_six_vectors,
    analyze_structure,
    parse_poscar,
)


def local_vectors(pair_lengths=(2.0, 2.0, 2.0)):
    x, y, z = pair_lengths
    return [
        np.array([x, 0.0, 0.0]),
        np.array([-x, 0.0, 0.0]),
        np.array([0.0, y, 0.0]),
        np.array([0.0, -y, 0.0]),
        np.array([0.0, 0.0, z]),
        np.array([0.0, 0.0, -z]),
    ]


def boxed_structure(pair_lengths=(2.0, 2.0, 2.0), center=(0.5, 0.5, 0.5), extra=None):
    lattice = np.eye(3) * 20.0
    center = np.asarray(center, dtype=float)
    cart = [center @ lattice]
    species = ["Ni"]
    for vector in local_vectors(pair_lengths):
        cart.append(center @ lattice + vector)
        species.append("N")
    if extra:
        for symbol, vector in extra:
            cart.append(center @ lattice + np.asarray(vector, dtype=float))
            species.append(symbol)
    fractional = np.asarray(cart) @ np.linalg.inv(lattice)
    return StructureData(lattice, tuple(species), fractional, identifier="box")


def rotate_vectors(vectors):
    axis = np.array([1.0, 2.0, -1.0])
    axis /= np.linalg.norm(axis)
    angle = 0.73
    kx, ky, kz = axis
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    R = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    return [R @ np.asarray(v) for v in vectors]


def make_supercell_x2(structure: StructureData) -> StructureData:
    lattice = structure.lattice.copy()
    lattice[0] *= 2.0
    species = []
    fractional = []
    for rep in range(2):
        for symbol, frac in zip(structure.species, structure.fractional, strict=True):
            species.append(symbol)
            fractional.append([(frac[0] + rep) / 2.0, frac[1], frac[2]])
    return StructureData(lattice, tuple(species), np.asarray(fractional), identifier="x2")


def vectors_from_angle_table(labels, angles, lengths):
    n = len(labels)
    G = np.eye(n)
    index = {label: i for i, label in enumerate(labels)}
    for (a, b), angle in angles.items():
        i, j = index[a], index[b]
        G[i, j] = G[j, i] = math.cos(math.radians(angle))
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[-3:]
    coords = evecs[:, order] @ np.diag(np.sqrt(np.maximum(evals[order], 0.0)))
    coords /= np.linalg.norm(coords, axis=1)[:, None]
    return [coords[i] * lengths[labels[i]] for i in range(n)]


class TestOTPD(unittest.TestCase):
    def test_ideal_octahedron_is_zero(self):
        site = analyze_six_vectors("Ni", ["N"] * 6, local_vectors())
        self.assertIsNotNone(site)
        assert site is not None
        self.assertAlmostEqual(site.values["pair_range"], 0.0, places=12)
        self.assertAlmostEqual(site.values["uniaxial_polarity"], 0.0, places=12)
        self.assertAlmostEqual(site.values["trans_bend_rms_deg"], 0.0, places=12)
        self.assertAlmostEqual(site.values["cis_bend_rms_deg"], 0.0, places=12)
        self.assertAlmostEqual(site.values["baur_distortion"], 0.0, places=12)

    def test_signed_uniaxial_polarity(self):
        compressed = analyze_six_vectors("Ni", ["N"] * 6, local_vectors((1.8, 2.1, 2.1)))
        elongated = analyze_six_vectors("Ni", ["N"] * 6, local_vectors((2.0, 2.0, 2.4)))
        balanced = analyze_six_vectors("Ni", ["N"] * 6, local_vectors((1.8, 2.0, 2.2)))
        assert compressed and elongated and balanced
        self.assertGreater(compressed.values["uniaxial_polarity"], 0.0)
        self.assertLess(elongated.values["uniaxial_polarity"], 0.0)
        self.assertAlmostEqual(balanced.values["uniaxial_polarity"], 0.0, places=12)
        self.assertGreater(balanced.values["pair_range"], 0.0)

    def test_permutation_and_rotation_invariance(self):
        vectors = local_vectors((1.9, 2.1, 2.2))
        symbols = ["N"] * 6
        reference = analyze_six_vectors("Ni", symbols, vectors)
        assert reference
        perm = [4, 0, 5, 2, 1, 3]
        permuted = analyze_six_vectors("Ni", [symbols[i] for i in perm], [vectors[i] for i in perm])
        rotated = analyze_six_vectors("Ni", symbols, rotate_vectors(vectors))
        assert permuted and rotated
        for name in ("pair_range", "uniaxial_polarity", "baur_distortion", "trans_bend_rms_deg"):
            self.assertAlmostEqual(reference.values[name], permuted.values[name], places=12)
            self.assertAlmostEqual(reference.values[name], rotated.values[name], places=12)

    def test_trigonal_prism_rejected(self):
        vectors = []
        for z, phase in ((1.0, 0.0), (-1.0, 0.0)):
            for k in range(3):
                angle = phase + 2.0 * math.pi * k / 3.0
                vectors.append(np.array([math.cos(angle), math.sin(angle), z]))
        self.assertIsNone(analyze_six_vectors("Ni", ["N"] * 6, vectors))

    def test_donor_size_bond_correlation(self):
        vectors = [
            [2.0, 0, 0], [-2.4, 0, 0],
            [0, 2.0, 0], [0, -2.4, 0],
            [0, 0, 2.0], [0, 0, -2.4],
        ]
        site = analyze_six_vectors("Ni", ["O", "S", "O", "S", "O", "S"], vectors)
        assert site
        self.assertAlmostEqual(site.values["donor_size_bond_correlation"], 1.0, places=12)

    def test_plane_twist_extension(self):
        normals = [
            np.array([1, 0, 0]), np.array([0, 1, 0]),
            np.array([1, 0, 0]), np.array([1, 0, 0]),
            np.array([0, 0, 1]), np.array([0, 0, 1]),
        ]
        site = analyze_six_vectors(
            "Ni", ["N"] * 6, local_vectors(), donor_plane_normals=normals
        )
        assert site
        self.assertAlmostEqual(site.values["donor_plane_twist_mean_deg"], 30.0, places=12)

    def test_periodic_structure_and_supercell_invariance(self):
        structure = boxed_structure((1.8, 2.1, 2.1), center=(0.95, 0.95, 0.95))
        base = analyze_structure(structure, center_elements={"Ni"}, include_plane_extension=False)
        supercell = analyze_structure(make_supercell_x2(structure), center_elements={"Ni"}, include_plane_extension=False)
        self.assertEqual(base["status"], "ok")
        self.assertEqual(supercell["status"], "ok")
        self.assertEqual(base["qualified_site_count"], 1)
        self.assertEqual(supercell["qualified_site_count"], 2)
        for name in base["scalars"]:
            a, b = base["scalars"][name], supercell["scalars"][name]
            if a is None or b is None:
                self.assertIs(a, b)
            else:
                self.assertAlmostEqual(a, b, places=10, msg=name)

    def test_seventh_tied_neighbor_is_ambiguous(self):
        structure = boxed_structure(extra=[("N", [math.sqrt(2.0), math.sqrt(2.0), 0.0])])
        result = analyze_structure(structure, center_elements={"Ni"}, include_plane_extension=False)
        self.assertEqual(result["status"], "domain_no_octahedral_like_site")
        self.assertEqual(result["rejection_counts"].get("ambiguous_six_neighbor_shell"), 1)

    def test_poscar_parser(self):
        text = """NiN6 test
1.0
20 0 0
0 20 0
0 0 20
Ni N
1 6
Direct
0.5 0.5 0.5
0.6 0.5 0.5
0.4 0.5 0.5
0.5 0.6 0.5
0.5 0.4 0.5
0.5 0.5 0.6
0.5 0.5 0.4
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "POSCAR"
            path.write_text(text, encoding="utf-8")
            structure = parse_poscar(path)
            result = analyze_structure(structure, center_elements={"Ni"}, include_plane_extension=False)
        self.assertEqual(result["qualified_site_count"], 1)
        self.assertAlmostEqual(result["scalars"]["otpd_pair_range_mean"], 0.0, places=12)

    def test_invalid_and_nonfinite_structures_raise(self):
        with self.assertRaises(DescriptorError):
            StructureData(np.zeros((3, 3)), ("Ni",), np.zeros((1, 3)))
        with self.assertRaises(DescriptorError):
            StructureData(np.eye(3), ("Ni",), np.array([[math.nan, 0.0, 0.0]]))
        with self.assertRaises(DescriptorError):
            analyze_six_vectors("Ni", ["N"] * 6, local_vectors()[:-1] + [np.zeros(3)])

    def test_fewer_than_six_neighbors_is_typed_domain_absence(self):
        lattice = np.eye(3) * 20.0
        center = np.array([0.5, 0.5, 0.5])
        cart = [center @ lattice] + [center @ lattice + v for v in local_vectors()[:5]]
        structure = StructureData(
            lattice,
            ("Ni",) + ("N",) * 5,
            np.asarray(cart) @ np.linalg.inv(lattice),
            identifier="five-neighbor",
        )
        result = analyze_structure(structure, center_elements={"Ni"}, include_plane_extension=False)
        self.assertEqual(result["status"], "domain_no_octahedral_like_site")
        self.assertEqual(result["rejection_counts"].get("fewer_than_six_neighbors"), 1)

    def test_degenerate_pair_order_has_deterministic_tie_break(self):
        first = analyze_six_vectors("Ni", ["N"] * 6, local_vectors())
        second = analyze_six_vectors("Ni", ["N"] * 6, local_vectors())
        assert first and second
        self.assertEqual(first.pair_indices, second.pair_indices)
        self.assertEqual(first.short_pair_axis, second.short_pair_axis)
        self.assertEqual(first.short_pair_axis, [1.0, 0.0, 0.0])

    def test_skew_periodic_cell_matches_large_orthogonal_cell(self):
        reference = analyze_structure(
            boxed_structure((1.8, 2.1, 2.1), center=(0.96, 0.94, 0.93)),
            center_elements={"Ni"},
            include_plane_extension=False,
        )
        lattice = np.array([[20.0, 0.0, 0.0], [4.0, 19.0, 0.0], [2.0, 3.0, 18.0]])
        center_frac = np.array([0.96, 0.94, 0.93])
        center_cart = center_frac @ lattice
        cart = [center_cart] + [center_cart + v for v in local_vectors((1.8, 2.1, 2.1))]
        skew = StructureData(
            lattice,
            ("Ni",) + ("N",) * 6,
            np.asarray(cart) @ np.linalg.inv(lattice),
            identifier="skew",
        )
        observed = analyze_structure(skew, center_elements={"Ni"}, include_plane_extension=False)
        self.assertEqual(observed["status"], "ok")
        for name in reference["scalars"]:
            a, b = reference["scalars"][name], observed["scalars"][name]
            if a is None or b is None:
                self.assertIs(a, b)
            else:
                self.assertAlmostEqual(a, b, places=10, msg=name)

    def test_paper_compound_1_reconstruction(self):
        labels = ["N1", "N2", "N3", "N4", "N5", "N6"]
        lengths = {"N1": 2.111, "N2": 2.008, "N3": 2.107, "N4": 2.112, "N5": 2.003, "N6": 2.116}
        angles = {
            ("N5", "N2"): 178.7,
            ("N2", "N4"): 103.3,
            ("N5", "N3"): 101.6,
            ("N3", "N4"): 95.5,
            ("N2", "N3"): 77.6,
            ("N1", "N4"): 89.6,
            ("N5", "N1"): 102.8,
            ("N5", "N6"): 78.2,
            ("N2", "N1"): 78.1,
            ("N2", "N6"): 100.8,
            ("N3", "N1"): 155.6,
            ("N3", "N6"): 89.8,
            ("N5", "N4"): 77.7,
            ("N1", "N6"): 95.2,
            ("N4", "N6"): 155.9,
        }
        vectors = vectors_from_angle_table(labels, angles, lengths)
        site = analyze_six_vectors("Ni", ["N"] * 6, vectors)
        assert site
        paired_label_sets = {
            frozenset((labels[i], labels[j])) for i, j in site.pair_indices
        }
        self.assertEqual(
            paired_label_sets,
            {frozenset(("N2", "N5")), frozenset(("N1", "N3")), frozenset(("N4", "N6"))},
        )
        axial = (lengths["N2"] + lengths["N5"]) / 2.0
        outer_1 = (lengths["N1"] + lengths["N3"]) / 2.0
        outer_2 = (lengths["N4"] + lengths["N6"]) / 2.0
        expected = math.log(((outer_1 + outer_2) / 2.0) / axial)
        self.assertAlmostEqual(site.values["short_pair_log_compression"], expected, places=10)
        self.assertGreater(site.values["uniaxial_polarity"], 0.0)
        self.assertLess(site.values["trans_bend_rms_deg"], 25.0)
        self.assertLess(site.values["cis_bend_rms_deg"], 15.0)

    def test_angular_and_radial_channels_are_separate(self):
        vectors = local_vectors((1.8, 2.1, 2.1))
        # Bend one trans pair symmetrically without changing its two lengths.
        angle = math.radians(15.0)
        vectors[0] = np.array([1.8, 0, 0])
        vectors[1] = np.array([-1.8 * math.cos(angle), 1.8 * math.sin(angle), 0])
        site = analyze_six_vectors("Ni", ["N"] * 6, vectors)
        assert site
        self.assertGreater(site.values["trans_bend_rms_deg"], 0.0)
        expected_range = (2.1 - 1.8) / 2.0
        self.assertAlmostEqual(site.values["pair_range"], expected_range, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
