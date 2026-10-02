"""Auditable neutral-element lookup for the site-environment descriptor.

The numeric radii and outer-electron counts were generated from the RDKit
2025.09.4 periodic-table API. They are factual lookup data, not oxidation-state
assignments. See the task-specific third-party notice for provenance.
"""

from __future__ import annotations

from dataclasses import dataclass

_SYMBOLS_TEXT = "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
_RADII_TEXT = "0 0.31 0.28 1.28 0.96 0.84 0.76 0.71 0.66 0.57 0.58 1.66 1.41 1.21 1.11 1.07 1.05 1.02 1.06 2.03 1.76 1.7 1.6 1.52 1.39 1.39 1.32 1.26 1.24 1.32 1.22 1.22 1.2 1.19 1.2 1.2 1.16 2.2 1.95 1.9 1.75 1.64 1.54 1.47 1.46 1.42 1.39 1.45 1.44 1.42 1.39 1.39 1.38 1.39 1.4 2.44 2.15 2.07 2.04 2.03 2.01 1.99 1.98 1.98 1.96 1.94 1.92 1.92 1.89 1.9 1.87 1.87 1.75 1.7 1.62 1.51 1.44 1.41 1.36 1.36 1.32 1.45 1.46 1.48 1.4 1.5 1.5 2.6 2.2 2.15 2.06 2 1.96 1.9 1.87 1.8 1.69 1.9 1.9 1.9 1.9 1.9 1.9 1.9 1.9 1.9 1.9 1.9 1.9 1.9 1.9 1.9 1.9 1.36 1.43 1.62 1.75 1.65 1.57"
_OUTER_TEXT = "0 1 2 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 9 10 11 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 9 10 11 2 3 4 5 6 7 8 1 2 3 4 3 4 5 6 7 8 9 10 11 12 13 14 15 4 5 6 7 8 9 10 11 2 3 4 5 6 7 8 1 2 3 4 3 4 5 6 7 8 9 10 11 12 13 14 15 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2"

SYMBOLS: tuple[str, ...] = tuple(_SYMBOLS_TEXT.split())
COVALENT_RADII_ANGSTROM: tuple[float, ...] = tuple(float(x) for x in _RADII_TEXT.split())
OUTER_ELECTRONS: tuple[int, ...] = tuple(int(x) for x in _OUTER_TEXT.split())

if not (len(SYMBOLS) == len(COVALENT_RADII_ANGSTROM) == len(OUTER_ELECTRONS) == 119):
    raise RuntimeError(
        "element table length mismatch: "
        f"symbols={len(SYMBOLS)}, radii={len(COVALENT_RADII_ANGSTROM)}, "
        f"outer={len(OUTER_ELECTRONS)}"
    )

SYMBOL_TO_Z = {symbol: z for z, symbol in enumerate(SYMBOLS) if z}


@dataclass(frozen=True)
class ElementRecord:
    """Neutral lookup values used by the descriptor."""

    symbol: str
    atomic_number: int
    covalent_radius_angstrom: float
    outer_electrons: int


def normalize_symbol(token: str) -> str:
    """Normalize an element or common POTCAR-like token without guessing chemistry."""

    token = token.strip()
    if not token:
        raise ValueError("empty element symbol")
    letters: list[str] = []
    for char in token:
        if char.isalpha():
            letters.append(char)
        else:
            break
    if not letters:
        raise ValueError(f"invalid element token: {token!r}")
    raw = "".join(letters)
    candidate = raw[0].upper() + raw[1:].lower()
    if candidate not in SYMBOL_TO_Z and len(candidate) > 2:
        candidate = candidate[:2]
    if candidate not in SYMBOL_TO_Z:
        candidate = candidate[:1]
    if candidate not in SYMBOL_TO_Z:
        raise ValueError(f"unknown element symbol: {token!r}")
    return candidate


def get_element(token: str) -> ElementRecord:
    symbol = normalize_symbol(token)
    z = SYMBOL_TO_Z[symbol]
    radius = float(COVALENT_RADII_ANGSTROM[z])
    if not radius > 0.0:
        raise ValueError(f"no positive covalent radius for {symbol}")
    return ElementRecord(
        symbol=symbol,
        atomic_number=z,
        covalent_radius_angstrom=radius,
        outer_electrons=int(OUTER_ELECTRONS[z]),
    )
