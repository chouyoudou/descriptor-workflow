from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from molecular_packing_descriptor import DESCRIPTOR_FUNCTIONS, DESCRIPTOR_NAMES, compute_all
from molecular_packing_descriptor.poscar import read_poscar

from conftest import write_poscar


def _base_structure(tmp_path, filename="base.vasp"):
    species = ["C"] * 5 + ["O", "N"]
    coords = np.asarray(
        [
            [4.0, 5.0, 5.0],
            [5.5, 5.0, 5.0],
            [6.7, 5.8, 5.0],
            [7.9, 6.6, 5.0],
            [9.4, 6.6, 5.0],
            [9.6, 9.3, 5.0],
            [4.0, 8.0, 5.0],
        ]
    )
    return write_poscar(tmp_path, species, coords, filename=filename)


def test_scalar_interface_and_finite_values(tmp_path) -> None:
    path = _base_structure(tmp_path)
    all_values = compute_all(path)
    assert tuple(all_values) == DESCRIPTOR_NAMES
    for name, function in DESCRIPTOR_FUNCTIONS.items():
        value = function(path)
        assert type(value) is float
        assert math.isfinite(value)
        assert value == pytest.approx(all_values[name], abs=1e-12)


def test_translation_and_species_group_order_invariance(tmp_path) -> None:
    base = _base_structure(tmp_path, "base.vasp")
    structure = read_poscar(base)
    translated_coords = structure.cart_coords + np.asarray([3.2, -1.7, 4.4])
    translated = write_poscar(
        tmp_path,
        list(structure.species),
        translated_coords,
        cell=structure.lattice,
        species_order=["N", "C", "O"],
        filename="translated.vasp",
    )
    a = compute_all(base)
    b = compute_all(translated)
    for name in DESCRIPTOR_NAMES:
        assert a[name] == pytest.approx(b[name], abs=2e-10)


def test_two_by_one_by_one_supercell_invariance(tmp_path) -> None:
    base = _base_structure(tmp_path, "base.vasp")
    structure = read_poscar(base)
    species = list(structure.species) * 2
    coords = np.vstack([structure.cart_coords, structure.cart_coords + structure.lattice[0]])
    cell = structure.lattice.copy()
    cell[0] *= 2.0
    supercell = write_poscar(
        tmp_path,
        species,
        coords,
        cell=cell,
        filename="supercell.vasp",
    )
    a = compute_all(base)
    b = compute_all(supercell)
    for name in DESCRIPTOR_NAMES:
        assert a[name] == pytest.approx(b[name], abs=2e-10)


def test_no_contact_and_periodic_network_fallbacks_are_finite(tmp_path) -> None:
    isolated = write_poscar(tmp_path, ["He"], [[5.0, 5.0, 5.0]], filename="isolated.vasp")
    # Carbon atoms spaced 1.5 A along a 3 A periodic x cell: the graph has a
    # translational cycle and is deliberately classified as periodic.
    network = write_poscar(
        tmp_path,
        ["C", "C"],
        [[0.0, 5.0, 5.0], [1.5, 5.0, 5.0]],
        cell=np.asarray([[3.0, 0, 0], [0, 12.0, 0], [0, 0, 12.0]]),
        filename="network.vasp",
    )
    for path in (isolated, network):
        values = compute_all(path)
        assert all(math.isfinite(value) for value in values.values())
    assert all(value == 0.0 for value in compute_all(isolated).values())


def test_cli_writes_contract_json(tmp_path) -> None:
    path = _base_structure(tmp_path)
    output = tmp_path / "result.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "molecular_packing_descriptor",
            str(path),
            "--descriptor",
            "spacer_path_extension",
            "--output",
            str(output),
        ],
        check=True,
    )
    data = json.loads(output.read_text())
    assert data["task_id"] == "paper-03aef80c9d0a380d998e06d1"
    assert data["descriptor"] == "spacer_path_extension"
    assert isinstance(data["value"], float)
    assert math.isfinite(data["value"])
    assert "charge" in data["interpretation_scope"]
