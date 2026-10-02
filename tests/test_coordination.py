from __future__ import annotations

import numpy as np

from molecular_packing_descriptor import compute_all

from conftest import write_poscar


def _tetrahedron(distances=(2.25, 2.25, 2.25, 2.25)) -> np.ndarray:
    center = np.asarray([10.0, 10.0, 10.0])
    directions = np.asarray(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
    )
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return np.vstack([center, center + directions * np.asarray(distances)[:, None]])


def test_regular_vs_distorted_tetrahedral_shell(tmp_path) -> None:
    regular = write_poscar(
        tmp_path,
        ["Fe", "Cl", "Cl", "Cl", "Cl"],
        _tetrahedron(),
        filename="regular.vasp",
    )
    distorted_coords = _tetrahedron((2.05, 2.30, 2.45, 2.65))
    distorted_coords[-1] += np.asarray([0.25, -0.10, 0.0])
    distorted = write_poscar(
        tmp_path,
        ["Fe", "Cl", "Cl", "Cl", "Cl"],
        distorted_coords,
        filename="distorted.vasp",
    )
    a = compute_all(regular)
    b = compute_all(distorted)
    assert a["coordination_shell_radial_distortion"] < 1e-10
    assert a["coordination_shell_orientation_anisotropy"] < 1e-10
    assert b["coordination_shell_radial_distortion"] > 0.05
    assert b["coordination_shell_orientation_anisotropy"] > 1e-4
