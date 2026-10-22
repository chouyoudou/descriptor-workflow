"""Static element lookup used by the descriptor.

The radii and outer-electron values were generated from RDKit 2025.09.4
``Chem.GetPeriodicTable()``.  See ``THIRD_PARTY-paper-5a4d4d2640ecebb85628abb0.md``
and ``LICENSES/RDKit-BSD-3-Clause.txt`` for provenance and license details.
"""

from __future__ import annotations

from dataclasses import dataclass

SYMBOLS: tuple[str, ...] = ('X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og')
COVALENT_RADII: tuple[float, ...] = (0.0, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.7, 1.6, 1.52, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.2, 1.19, 1.2, 1.2, 1.16, 2.2, 1.95, 1.9, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.4, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.9, 1.87, 1.87, 1.75, 1.7, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.4, 1.5, 1.5, 2.6, 2.2, 2.15, 2.06, 2.0, 1.96, 1.9, 1.87, 1.8, 1.69, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.36, 1.43, 1.62, 1.75, 1.65, 1.57)
VDW_RADII: tuple[float, ...] = (0.0, 1.2, 1.4, 2.2, 1.9, 1.8, 1.7, 1.6, 1.55, 1.5, 1.54, 2.4, 2.2, 2.1, 2.1, 1.95, 1.8, 1.8, 1.88, 2.8, 2.4, 2.3, 2.15, 2.05, 2.05, 2.05, 2.05, 2.0, 2.0, 2.0, 2.1, 2.1, 2.1, 2.05, 1.9, 1.9, 2.02, 2.9, 2.55, 2.4, 2.3, 2.15, 2.1, 2.05, 2.05, 2.0, 2.05, 2.1, 2.2, 2.2, 2.25, 2.2, 2.1, 2.1, 2.16, 3.0, 2.7, 2.5, 2.48, 2.47, 2.45, 2.43, 2.42, 2.4, 2.38, 2.37, 2.35, 2.33, 2.32, 2.3, 2.28, 2.27, 2.25, 2.2, 2.1, 2.05, 2.0, 2.0, 2.05, 2.1, 2.05, 2.2, 2.3, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.4, 2.0, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
OUTER_ELECTRONS: tuple[int, ...] = (0, 1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)

SYMBOL_TO_Z = {symbol: z for z, symbol in enumerate(SYMBOLS) if z}

# Conventional chemistry classification used only as an interpretable lookup.
# Borderline metalloids are intentionally not counted as metals.
_METAL_Z = frozenset(
    [3, 4, 11, 12, 13, 19, 20, 31, 37, 38, 49, 50, 55, 56, 81, 82, 83, 84, 87, 88, 113, 114, 115, 116]
    + list(range(21, 31)) + list(range(39, 49)) + list(range(57, 81))
    + list(range(89, 113))
)
_F_BLOCK_Z = frozenset(list(range(57, 72)) + list(range(89, 104)))
# Elements frequently acting as anionic or terminal ligands in explicit structures.
_LIGAND_PRIORITY_Z = frozenset([1, 7, 8, 9, 16, 17, 34, 35, 52, 53])

@dataclass(frozen=True)
class ElementRecord:
    symbol: str
    atomic_number: int
    covalent_radius: float
    vdw_radius: float
    outer_electrons: int
    is_metal: bool
    is_f_block: bool
    ligand_priority: bool

def normalize_symbol(token: str) -> str:
    token = token.strip()
    if not token:
        raise ValueError("empty element symbol")
    # Accept common POTCAR-like labels such as Fe_pv or O_s.
    letters = []
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
    cov = COVALENT_RADII[z]
    vdw = VDW_RADII[z]
    if not (cov > 0.0):
        cov = 1.5
    if not (vdw > 0.0):
        vdw = max(1.5, 1.20 * cov + 0.5)
    return ElementRecord(
        symbol=symbol, atomic_number=z, covalent_radius=float(cov),
        vdw_radius=float(vdw), outer_electrons=int(OUTER_ELECTRONS[z]),
        is_metal=z in _METAL_Z, is_f_block=z in _F_BLOCK_Z,
        ligand_priority=z in _LIGAND_PRIORITY_Z,
    )
