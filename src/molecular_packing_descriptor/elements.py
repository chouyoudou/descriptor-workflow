"""Element symbols and radius lookup used by the descriptor.

The numeric table is a clean data snapshot of the public RDKit ``PeriodicTable``
``GetRcovalent`` and ``GetRvdw`` values queried on 2026-10-02.  RDKit is
BSD-3-Clause licensed.  No RDKit source code is copied and RDKit is not a
runtime dependency.  The radii are geometric lookup defaults, not oxidation
states, charges, spin states, or bond orders.
"""

from __future__ import annotations

CHEMICAL_SYMBOLS: tuple[str, ...] = (
    'X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
    'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
    'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
    'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
    'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
    'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
    'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh',
    'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
)

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

VDW_RADII_ANGSTROM: tuple[float, ...] = (
    0.000, 1.200, 1.400, 2.200, 1.900, 1.800, 1.700, 1.600, 1.550, 1.500,
    1.540, 2.400, 2.200, 2.100, 2.100, 1.950, 1.800, 1.800, 1.880, 2.800,
    2.400, 2.300, 2.150, 2.050, 2.050, 2.050, 2.050, 2.000, 2.000, 2.000,
    2.100, 2.100, 2.100, 2.050, 1.900, 1.900, 2.020, 2.900, 2.550, 2.400,
    2.300, 2.150, 2.100, 2.050, 2.050, 2.000, 2.050, 2.100, 2.200, 2.200,
    2.250, 2.200, 2.100, 2.100, 2.160, 3.000, 2.700, 2.500, 2.480, 2.470,
    2.450, 2.430, 2.420, 2.400, 2.380, 2.370, 2.350, 2.330, 2.320, 2.300,
    2.280, 2.270, 2.250, 2.200, 2.100, 2.050, 2.000, 2.000, 2.050, 2.100,
    2.050, 2.200, 2.300, 2.300, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000,
    2.400, 2.000, 2.300, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000,
    2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000,
    2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000,
)

ATOMIC_NUMBERS: dict[str, int] = {symbol: z for z, symbol in enumerate(CHEMICAL_SYMBOLS) if z}


def canonical_symbol(raw: str) -> str:
    token = raw.strip()
    if not token:
        raise ValueError("empty element symbol")
    symbol = token[0].upper() + token[1:].lower()
    if symbol not in ATOMIC_NUMBERS:
        raise ValueError(f"unsupported element symbol: {raw!r}")
    return symbol


def covalent_radius(symbol: str) -> float:
    return COVALENT_RADII_ANGSTROM[ATOMIC_NUMBERS[canonical_symbol(symbol)]]


def vdw_radius(symbol: str) -> float:
    return VDW_RADII_ANGSTROM[ATOMIC_NUMBERS[canonical_symbol(symbol)]]
