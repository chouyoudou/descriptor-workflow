from __future__ import annotations

import numpy as np
import pytest

from molecular_packing_descriptor import DescriptorError, parse_poscar_text

from conftest import poscar_text


def test_direct_and_cartesian_match() -> None:
    species = ["C", "H", "H"]
    coords = np.asarray([[2.0, 3.0, 4.0], [3.0, 3.0, 4.0], [1.0, 3.0, 4.0]])
    cell = np.asarray([[9.0, 0.2, 0.0], [0.0, 10.0, 0.3], [0.1, 0.0, 11.0]])
    direct = parse_poscar_text(poscar_text(species, coords, cell=cell, mode="Direct"))
    cart = parse_poscar_text(
        poscar_text(species, coords, cell=cell, mode="Cartesian", selective=True)
    )
    assert direct.species == cart.species
    assert np.allclose(direct.cart_coords, cart.cart_coords, atol=1e-10)


def test_negative_volume_scale() -> None:
    text = """C\n-1000\n1 0 0\n0 1 0\n0 0 1\nC\n1\nDirect\n0.25 0.25 0.25\n"""
    structure = parse_poscar_text(text)
    assert structure.volume == pytest.approx(1000.0)
    assert np.allclose(structure.cart_coords[0], [2.5, 2.5, 2.5])


def test_vasp4_comment_symbols() -> None:
    text = """C H molecular fragment\n1\n10 0 0\n0 10 0\n0 0 10\n1 2\nDirect\n0.1 0.1 0.1\n0.2 0.1 0.1\n0.0 0.1 0.1\n"""
    structure = parse_poscar_text(text)
    assert structure.species == ("C", "H", "H")


def test_invalid_symbol_rejected() -> None:
    text = """bad\n1\n10 0 0\n0 10 0\n0 0 10\nQq\n1\nDirect\n0 0 0\n"""
    with pytest.raises(DescriptorError):
        parse_poscar_text(text)
