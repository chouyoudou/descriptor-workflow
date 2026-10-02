from __future__ import annotations

import numpy as np

from molecular_packing_descriptor import compute_all

from conftest import write_poscar


def test_straight_spacer_is_more_extended_and_rodlike(tmp_path) -> None:
    straight = np.asarray([[4.0 + 1.5 * i, 6.0, 6.0] for i in range(6)])
    bent = np.asarray(
        [
            [4.0, 6.0, 6.0],
            [5.5, 6.0, 6.0],
            [6.8, 6.75, 6.0],
            [6.8, 8.25, 6.0],
            [5.5, 9.00, 6.0],
            [4.0, 9.00, 6.0],
        ]
    )
    straight_path = write_poscar(tmp_path, ["C"] * 6, straight, filename="straight.vasp")
    bent_path = write_poscar(tmp_path, ["C"] * 6, bent, filename="bent.vasp")
    a = compute_all(straight_path)
    b = compute_all(bent_path)
    assert a["spacer_path_extension"] > 0.999
    assert b["spacer_path_extension"] < a["spacer_path_extension"] - 0.08
    assert a["component_shape_anisotropy"] > b["component_shape_anisotropy"]
    assert a["spacer_atom_fraction"] == b["spacer_atom_fraction"] == 4 / 6


def test_ring_is_not_mislabeled_as_spacer(tmp_path) -> None:
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    radius = 1.4
    coords = np.column_stack(
        [8 + radius * np.cos(angles), 8 + radius * np.sin(angles), np.full(6, 8.0)]
    )
    path = write_poscar(tmp_path, ["C"] * 6, coords)
    result = compute_all(path)
    assert result["spacer_path_extension"] == 0.0
    assert result["spacer_atom_fraction"] == 0.0
