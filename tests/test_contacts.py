from __future__ import annotations

import numpy as np
import pytest

from molecular_packing_descriptor import compute_all

from conftest import write_poscar


def test_contact_compactness_decreases_with_separation(tmp_path) -> None:
    near = write_poscar(
        tmp_path,
        ["H", "Cl"],
        [[8.0, 8.0, 8.0], [10.55, 8.0, 8.0]],
        filename="near.vasp",
    )
    far = write_poscar(
        tmp_path,
        ["H", "Cl"],
        [[8.0, 8.0, 8.0], [11.05, 8.0, 8.0]],
        filename="far.vasp",
    )
    a = compute_all(near)
    b = compute_all(far)
    assert a["interfragment_contact_compactness"] > b["interfragment_contact_compactness"] > 0
    assert a["interfragment_contact_anisotropy"] > 0.99
    assert a["heteroelement_contact_fraction"] == 1.0
    assert a["contact_pair_specificity"] == 1.0
    assert a["interfragment_contact_coordination"] == 1.0


def test_isotropic_six_contact_shell_has_low_anisotropy(tmp_path) -> None:
    center = np.asarray([10.0, 10.0, 10.0])
    vectors = np.asarray(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    coords = np.vstack([center, center + 3.0 * vectors])
    path = write_poscar(tmp_path, ["C"] + ["O"] * 6, coords)
    result = compute_all(path)
    assert result["interfragment_contact_anisotropy"] < 1e-10
    assert result["heteroelement_contact_fraction"] == 1.0
    assert result["interfragment_contact_coordination"] == pytest.approx(12 / 7)


def test_pair_specificity_distinguishes_mixed_contact_types(tmp_path) -> None:
    center = np.asarray([10.0, 10.0, 10.0])
    coords = np.vstack(
        [
            center,
            center + [3.0, 0, 0],
            center + [-3.0, 0, 0],
            center + [0, 3.0, 0],
            center + [0, -3.0, 0],
            center + [0, 0, 3.0],
            center + [0, 0, -3.0],
        ]
    )
    uniform = write_poscar(
        tmp_path,
        ["C"] + ["O"] * 6,
        coords,
        filename="uniform.vasp",
    )
    mixed = write_poscar(
        tmp_path,
        ["C", "O", "O", "N", "N", "F", "F"],
        coords,
        filename="mixed.vasp",
    )
    u = compute_all(uniform)
    m = compute_all(mixed)
    assert u["contact_pair_specificity"] > 0.99
    assert m["contact_pair_specificity"] < 0.25
