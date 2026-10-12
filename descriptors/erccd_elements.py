"""Independent neutral-element references and explicit label normalization."""

from __future__ import annotations

import math
import re


class DescriptorError(ValueError):
    """Raised for malformed structures or unsupported descriptor inputs."""


CHEMICAL_SYMBOLS: tuple[str, ...] = (
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na",
    "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V",
    "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am",
    "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
    "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)

# Clean numeric snapshot of RDKit PeriodicTable.GetRcovalent queried
# 2026-10-02. RDKit is BSD-3-Clause licensed; RDKit is not a dependency.
COVALENT_RADII_ANGSTROM: tuple[float, ...] = (
    0.000, 0.310, 0.280, 1.280, 0.960, 0.840, 0.760, 0.710, 0.660, 0.570,
    0.580, 1.660, 1.410, 1.210, 1.110, 1.070, 1.050, 1.020, 1.060, 2.030,
    1.760, 1.700, 1.600, 1.520, 1.390, 1.390, 1.320, 1.260, 1.240, 1.320,
    1.220, 1.220, 1.200, 1.190, 1.200, 1.200, 1.160, 2.200, 1.950, 1.900,
    1.750, 1.640, 1.540, 1.470, 1.460, 1.420, 1.390, 1.450, 1.440, 1.420,
    1.390, 1.390, 1.380, 1.390, 1.400, 2.440, 2.150, 2.070, 2.040, 2.030,
    2.010, 1.990, 1.980, 1.980, 1.960, 1.940, 1.920, 1.920, 1.890, 1.900,
    1.870, 1.870, 1.750, 1.700, 1.620, 1.510, 1.440, 1.410, 1.360, 1.360,
    1.320, 1.450, 1.460, 1.480, 1.400, 1.500, 1.500, 2.600, 2.200, 2.150,
    2.060, 2.000, 1.960, 1.900, 1.870, 1.800, 1.690, 1.900, 1.900, 1.900,
    1.900, 1.900, 1.900, 1.900, 1.900, 1.900, 1.900, 1.900, 1.900, 1.900,
    1.900, 1.900, 1.900, 1.360, 1.430, 1.620, 1.750, 1.650, 1.570,
)

ATOMIC_NUMBERS: dict[str, int] = {
    symbol: number for number, symbol in enumerate(CHEMICAL_SYMBOLS) if number
}

_EXPLICIT_ALIASES = {
    "D": "H",
    "T": "H",
    "DEUTERIUM": "H",
    "TRITIUM": "H",
}


def canonical_symbol(raw: str) -> str:
    """Normalize bare symbols, POTCAR suffixes, and D/T aliases."""
    token = str(raw).strip()
    if not token:
        raise DescriptorError("empty element symbol")
    alias = _EXPLICIT_ALIASES.get(token.upper())
    if alias is not None:
        return alias
    match = re.fullmatch(r"([A-Za-z]{1,2})(?:[_./-].*)?", token)
    if not match:
        raise DescriptorError(f"cannot parse element label {raw!r}")
    letters = match.group(1)
    candidates = (letters[0].upper() + letters[1:].lower(), letters[0].upper())
    for symbol in candidates:
        if symbol in ATOMIC_NUMBERS:
            return symbol
    raise DescriptorError(f"unsupported element label {raw!r}")


def covalent_radius_angstrom(symbol: str) -> float:
    z = ATOMIC_NUMBERS[canonical_symbol(symbol)]
    radius = COVALENT_RADII_ANGSTROM[z]
    if not math.isfinite(radius) or radius <= 0.0:
        raise DescriptorError(f"missing positive covalent radius for {symbol!r}")
    return radius


__all__ = [
    "ATOMIC_NUMBERS",
    "CHEMICAL_SYMBOLS",
    "COVALENT_RADII_ANGSTROM",
    "DescriptorError",
    "canonical_symbol",
    "covalent_radius_angstrom",
]
