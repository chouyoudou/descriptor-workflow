from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pytest


def poscar_text(
    species: Iterable[str],
    cart_coords: Iterable[Iterable[float]],
    *,
    cell: np.ndarray | None = None,
    comment: str = "synthetic descriptor test",
    species_order: list[str] | None = None,
    selective: bool = False,
    mode: str = "Direct",
) -> str:
    species = list(species)
    coords = np.asarray(list(cart_coords), dtype=float)
    cell = np.asarray(cell if cell is not None else np.eye(3) * 20.0, dtype=float)
    if len(species) != len(coords):
        raise ValueError("species/coordinate length mismatch")
    order = species_order or list(dict.fromkeys(species))
    indices = [idx for symbol in order for idx, actual in enumerate(species) if actual == symbol]
    counts = [sum(actual == symbol for actual in species) for symbol in order]
    grouped = coords[indices]
    if mode.lower().startswith("d"):
        values = grouped @ np.linalg.inv(cell)
    else:
        values = grouped
    lines = [comment, "1.0"]
    lines.extend("  ".join(f"{value:.12f}" for value in row) for row in cell)
    lines.append(" ".join(order))
    lines.append(" ".join(str(value) for value in counts))
    if selective:
        lines.append("Selective dynamics")
    lines.append(mode)
    for row in values:
        suffix = " T T T" if selective else ""
        lines.append("  ".join(f"{value:.12f}" for value in row) + suffix)
    return "\n".join(lines) + "\n"


def write_poscar(tmp_path: Path, species, coords, **kwargs) -> Path:
    path = tmp_path / kwargs.pop("filename", "POSCAR")
    path.write_text(poscar_text(species, coords, **kwargs), encoding="utf-8")
    return path


@pytest.fixture
def cubic_cell() -> np.ndarray:
    return np.eye(3) * 20.0
