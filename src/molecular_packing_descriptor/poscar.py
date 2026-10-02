"""Strict, dependency-light POSCAR reader."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np

from .elements import ATOMIC_NUMBERS, canonical_symbol
from .model import DescriptorError, Structure

_INT_RE = re.compile(r"^[+-]?\d+$")
_SYMBOL_RE = re.compile(r"[A-Z][a-z]?")


def _nonempty(lines: Iterable[str]) -> list[str]:
    return [line.rstrip("\n") for line in lines if line.strip()]


def _parse_floats(line: str, *, count: int, context: str) -> list[float]:
    tokens = line.split()
    if len(tokens) < count:
        raise DescriptorError(f"{context}: expected at least {count} numbers")
    try:
        return [float(token) for token in tokens[:count]]
    except ValueError as exc:
        raise DescriptorError(f"{context}: invalid floating-point value") from exc


def _is_count_line(tokens: list[str]) -> bool:
    return bool(tokens) and all(_INT_RE.match(token) for token in tokens)


def _symbols_from_comment(comment: str, n_types: int) -> list[str]:
    candidates = []
    for token in _SYMBOL_RE.findall(comment):
        try:
            symbol = canonical_symbol(token)
        except ValueError:
            continue
        if symbol not in candidates:
            candidates.append(symbol)
        if len(candidates) == n_types:
            return candidates
    raise DescriptorError(
        "VASP4-style POSCAR requires element symbols in the comment line; "
        f"could not resolve {n_types} species"
    )


def parse_poscar_text(text: str, *, source: str | Path | None = None) -> Structure:
    """Parse a VASP POSCAR/CONTCAR string.

    Supported: VASP 4/5 symbols, one or three scale factors, negative-volume
    scale, Direct or Cartesian coordinates, and Selective Dynamics flags.
    Partial occupancies and disordered site records are intentionally outside
    the POSCAR format handled here.
    """

    lines = _nonempty(text.splitlines())
    if len(lines) < 8:
        raise DescriptorError("POSCAR is too short")

    comment = lines[0].strip() or "POSCAR"
    scale_tokens = lines[1].split()
    if len(scale_tokens) not in (1, 3):
        raise DescriptorError("POSCAR scale line must contain one or three values")
    try:
        scale_values = np.asarray([float(token) for token in scale_tokens], dtype=float)
    except ValueError as exc:
        raise DescriptorError("invalid POSCAR scale factor") from exc
    if not np.all(np.isfinite(scale_values)):
        raise DescriptorError("POSCAR scale factor must be finite")

    raw_lattice = np.asarray(
        [_parse_floats(lines[2 + row], count=3, context=f"lattice row {row + 1}") for row in range(3)],
        dtype=float,
    )
    raw_volume = abs(float(np.linalg.det(raw_lattice)))
    if raw_volume <= 1e-10:
        raise DescriptorError("lattice is singular")

    if len(scale_values) == 1:
        scale = float(scale_values[0])
        if scale == 0.0:
            raise DescriptorError("POSCAR scale factor cannot be zero")
        if scale < 0.0:
            linear_scale = (abs(scale) / raw_volume) ** (1.0 / 3.0)
        else:
            linear_scale = scale
        lattice = raw_lattice * linear_scale
        cart_scale = np.asarray([linear_scale, linear_scale, linear_scale])
    else:
        if np.any(scale_values <= 0.0):
            raise DescriptorError("three POSCAR scale factors must all be positive")
        # VASP's three-factor form scales Cartesian x/y/z components.
        lattice = raw_lattice * scale_values[np.newaxis, :]
        cart_scale = scale_values

    if abs(float(np.linalg.det(lattice))) <= 1e-10:
        raise DescriptorError("scaled lattice is singular")

    cursor = 5
    tokens = lines[cursor].split()
    if _is_count_line(tokens):
        counts = [int(token) for token in tokens]
        symbols = _symbols_from_comment(comment, len(counts))
        cursor += 1
    else:
        try:
            symbols = [canonical_symbol(token) for token in tokens]
        except ValueError as exc:
            raise DescriptorError(str(exc)) from exc
        cursor += 1
        if cursor >= len(lines):
            raise DescriptorError("missing POSCAR atom counts")
        count_tokens = lines[cursor].split()
        if not _is_count_line(count_tokens):
            raise DescriptorError("invalid POSCAR atom-count line")
        counts = [int(token) for token in count_tokens]
        cursor += 1
        if len(symbols) != len(counts):
            raise DescriptorError("number of element symbols and counts differs")

    if not counts or any(count <= 0 for count in counts):
        raise DescriptorError("POSCAR atom counts must be positive")
    n_atoms = sum(counts)
    if n_atoms > 200_000:
        raise DescriptorError("POSCAR atom count is unreasonably large")

    if cursor < len(lines) and lines[cursor].strip().lower().startswith("s"):
        cursor += 1
    if cursor >= len(lines):
        raise DescriptorError("missing POSCAR coordinate mode")
    mode = lines[cursor].strip().lower()
    cursor += 1
    if not mode or mode[0] not in {"d", "c", "k"}:
        raise DescriptorError("coordinate mode must be Direct or Cartesian")
    if cursor + n_atoms > len(lines):
        raise DescriptorError("fewer coordinate rows than atom count")

    coords = np.asarray(
        [_parse_floats(lines[cursor + i], count=3, context=f"coordinate row {i + 1}") for i in range(n_atoms)],
        dtype=float,
    )
    if not np.all(np.isfinite(coords)):
        raise DescriptorError("coordinates must be finite")

    try:
        inverse = np.linalg.inv(lattice)
    except np.linalg.LinAlgError as exc:
        raise DescriptorError("lattice inversion failed") from exc
    if mode[0] == "d":
        frac = coords
    else:
        cart = coords * cart_scale[np.newaxis, :]
        frac = cart @ inverse
    frac = frac - np.floor(frac)

    species: list[str] = []
    for symbol, count in zip(symbols, counts, strict=True):
        if symbol not in ATOMIC_NUMBERS:
            raise DescriptorError(f"unsupported element symbol: {symbol}")
        species.extend([symbol] * count)

    return Structure(
        comment=comment,
        lattice=np.asarray(lattice, dtype=float),
        species=tuple(species),
        frac_coords=np.asarray(frac, dtype=float),
        source=Path(source) if source is not None else None,
    )


def read_poscar(path: str | Path) -> Structure:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DescriptorError(f"cannot read POSCAR {source}: {exc}") from exc
    return parse_poscar_text(text, source=source)
