"""Shared data, element lookup, and POSCAR parsing for OTPD."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import re
from collections.abc import Mapping
from typing import Any

import numpy as np


SCHEMA = "octahedral-trans-pair-deformation/1"
ELEMENT_PROTOCOL = "cordero-covalent-radius-with-explicit-overrides/1"
STATIC_SCOPE = (
    "Static periodic geometry only. Values are not oxidation states, ligand-field "
    "splittings, magnetic anisotropy parameters, relaxation barriers, interaction "
    "energies, force constants, or dynamic response coefficients."
)

# Covalent radii in angstrom. The numerical values match the public RDKit
# PeriodicTable.GetRcovalent table used to generate this data-only lookup. For
# the ordinary elements they track the Cordero et al. crystallographic
# compilation (Dalton Trans. 2008, DOI 10.1039/B801115J). Callers can replace
# any value or load the project's reviewed element table explicitly.
DEFAULT_COVALENT_RADII: dict[str, float] = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66,
    'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05,
    'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76, 'Sc': 1.7, 'Ti': 1.6, 'V': 1.52, 'Cr': 1.39,
    'Mn': 1.39, 'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.2,
    'As': 1.19, 'Se': 1.2, 'Br': 1.2, 'Kr': 1.16, 'Rb': 2.2, 'Sr': 1.95, 'Y': 1.9, 'Zr': 1.75,
    'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
    'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.4, 'Cs': 2.44, 'Ba': 2.15,
    'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96,
    'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.9, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75,
    'Ta': 1.7, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32,
    'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.4, 'At': 1.5, 'Rn': 1.5, 'Fr': 2.6, 'Ra': 2.2,
    'Ac': 2.15, 'Th': 2.06, 'Pa': 2.0, 'U': 1.96, 'Np': 1.9, 'Pu': 1.87, 'Am': 1.8, 'Cm': 1.69,
    'Bk': 1.9, 'Cf': 1.9, 'Es': 1.9, 'Fm': 1.9, 'Md': 1.9, 'No': 1.9, 'Lr': 1.9, 'Rf': 1.9,
    'Db': 1.9, 'Sg': 1.9, 'Bh': 1.9, 'Hs': 1.9, 'Mt': 1.9, 'Ds': 1.9, 'Rg': 1.9, 'Cn': 1.9,
    'Nh': 1.36, 'Fl': 1.43, 'Mc': 1.62, 'Lv': 1.75, 'Ts': 1.65, 'Og': 1.57,
}

ELEMENT_SYMBOLS = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
    "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

CORE_SCALARS = (
    "otpd_pair_range_mean",
    "otpd_pair_range_max",
    "otpd_uniaxial_polarity_mean",
    "otpd_uniaxial_polarity_abs_mean",
    "otpd_short_pair_log_compression_mean",
    "otpd_long_pair_log_elongation_mean",
    "otpd_pair_asymmetry_rms_mean",
    "otpd_trans_bend_rms_mean_deg",
    "otpd_cis_bend_rms_mean_deg",
    "otpd_baur_distortion_mean",
)

EXTENSION_SCALARS = (
    "otpd_radius_pair_range_mean",
    "otpd_radius_uniaxial_polarity_mean",
    "otpd_strain_bend_correlation_mean",
    "otpd_donor_size_bond_correlation_mean",
    "otpd_donor_plane_twist_mean_deg",
)

ALL_SCALARS = CORE_SCALARS + EXTENSION_SCALARS


class DescriptorError(ValueError):
    """Raised for invalid structures or settings."""


@dataclasses.dataclass(frozen=True)
class Settings:
    shell_ratio_max: float = 1.55
    shell_gap_abs_min: float = 0.08
    shell_gap_ratio_min: float = 1.06
    neighbor_search_ratio: float = 1.90
    max_search_radius: float = 10.0
    trans_rms_max_deg: float = 35.0
    cis_rms_max_deg: float = 30.0
    backbone_ratio_max: float = 1.25
    plane_cross_min: float = 0.08
    numerical_tolerance: float = 1.0e-10

    def validate(self) -> None:
        values = dataclasses.asdict(self)
        for key, value in values.items():
            if not math.isfinite(float(value)):
                raise DescriptorError(f"non-finite setting {key}")
        if self.shell_ratio_max <= 0 or self.neighbor_search_ratio <= self.shell_ratio_max:
            raise DescriptorError("neighbor_search_ratio must exceed shell_ratio_max > 0")
        if self.max_search_radius <= 0:
            raise DescriptorError("max_search_radius must be positive")
        if not (0 < self.trans_rms_max_deg < 90 and 0 < self.cis_rms_max_deg < 90):
            raise DescriptorError("angle gates must lie between 0 and 90 degrees")


@dataclasses.dataclass(frozen=True)
class StructureData:
    lattice: np.ndarray
    species: tuple[str, ...]
    fractional: np.ndarray
    identifier: str | None = None

    def __post_init__(self) -> None:
        lattice = np.asarray(self.lattice, dtype=float)
        fractional = np.asarray(self.fractional, dtype=float)
        if lattice.shape != (3, 3):
            raise DescriptorError(f"lattice must have shape (3,3), got {lattice.shape}")
        if fractional.shape != (len(self.species), 3):
            raise DescriptorError(
                f"fractional coordinates must have shape ({len(self.species)},3), got {fractional.shape}"
            )
        if not np.isfinite(lattice).all() or not np.isfinite(fractional).all():
            raise DescriptorError("lattice and coordinates must be finite")
        det = float(np.linalg.det(lattice))
        if abs(det) <= 1.0e-10:
            raise DescriptorError("lattice is singular")
        canonical = tuple(canonical_symbol(s) for s in self.species)
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "fractional", np.mod(fractional, 1.0))
        object.__setattr__(self, "species", canonical)

    @property
    def atom_count(self) -> int:
        return len(self.species)


@dataclasses.dataclass(frozen=True)
class Neighbor:
    atom_index: int
    image: tuple[int, int, int]
    symbol: str
    vector: np.ndarray
    distance: float
    normalized_distance: float

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (self.atom_index, *self.image)


@dataclasses.dataclass
class SiteResult:
    center_index: int | None
    center_symbol: str
    neighbor_keys: list[tuple[int, int, int, int]]
    neighbor_symbols: list[str]
    pair_indices: list[tuple[int, int]]
    pair_neighbor_keys: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]
    pair_mean_lengths: list[float]
    pair_trans_bends_deg: list[float]
    short_pair_axis: list[float]
    values: dict[str, float | None]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_index": self.center_index,
            "center_symbol": self.center_symbol,
            "neighbor_keys": [list(k) for k in self.neighbor_keys],
            "neighbor_symbols": self.neighbor_symbols,
            "pair_indices": [list(p) for p in self.pair_indices],
            "pair_neighbor_keys": [[list(a), list(b)] for a, b in self.pair_neighbor_keys],
            "pair_mean_lengths": self.pair_mean_lengths,
            "pair_trans_bends_deg": self.pair_trans_bends_deg,
            "short_pair_axis": self.short_pair_axis,
            "values": self.values,
            "diagnostics": self.diagnostics,
        }


def canonical_symbol(token: str) -> str:
    text = str(token).strip()
    match = re.search(r"([A-Z][a-z]?)", text)
    if not match:
        match = re.search(r"([a-zA-Z]{1,2})", text)
        if not match:
            raise DescriptorError(f"cannot parse element symbol from {token!r}")
        candidate = match.group(1).capitalize()
    else:
        candidate = match.group(1)
    if candidate not in DEFAULT_COVALENT_RADII:
        raise DescriptorError(f"unknown element symbol {token!r}")
    return candidate


def radius_table(overrides: Mapping[str, float] | None = None) -> dict[str, float]:
    table = dict(DEFAULT_COVALENT_RADII)
    if overrides:
        for raw_symbol, raw_value in overrides.items():
            symbol = canonical_symbol(raw_symbol)
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0:
                raise DescriptorError(f"radius for {symbol} must be finite and positive")
            table[symbol] = value
    return table


def load_project_element_table(path: str | pathlib.Path) -> tuple[dict[str, float], dict[str, Any]]:
    """Load the project's reviewed element table.

    The table rows are ordered by atomic number. Cordero radius is primary;
    Pyykko single-bond radius is a declared fallback when Cordero is absent.
    """

    raw = pathlib.Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema") != "element_lookup_review_view_v1":
        raise DescriptorError("unexpected element-table schema")
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise DescriptorError("element table has no rows list")
    table: dict[str, float] = {}
    fallback_count = 0
    for row in rows:
        z = int(row["atomic_number"])
        if not (1 <= z < len(ELEMENT_SYMBOLS)):
            continue
        value = row.get("covalent_radius_cordero")
        source = "cordero"
        if value is None:
            value = row.get("covalent_radius_pyykko_single")
            source = "pyykko_single_fallback"
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value) and value > 0:
            table[ELEMENT_SYMBOLS[z]] = value
            fallback_count += int(source != "cordero")
    if not table:
        raise DescriptorError("element table yielded no radii")
    meta = {
        "schema": data.get("schema"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "radius_primary": "covalent_radius_cordero",
        "radius_fallback": "covalent_radius_pyykko_single",
        "resolved_count": len(table),
        "fallback_count": fallback_count,
    }
    return radius_table(table), meta


def structure_from_record(record: Mapping[str, Any]) -> StructureData:
    try:
        lattice = record["lattice"]
        species = record["species"]
        fractional = record["fractional_coordinates"]
    except KeyError as exc:
        raise DescriptorError(f"missing structure field {exc.args[0]}") from exc
    identifier = record.get("id")
    return StructureData(
        lattice=np.asarray(lattice, dtype=float),
        species=tuple(str(x) for x in species),
        fractional=np.asarray(fractional, dtype=float),
        identifier=None if identifier is None else str(identifier),
    )


def parse_poscar(path: str | pathlib.Path) -> StructureData:
    lines = [line.rstrip() for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()]
    if len(lines) < 8:
        raise DescriptorError("POSCAR is too short")
    comment = lines[0].strip() or pathlib.Path(path).name
    try:
        scale = float(lines[1].split()[0])
        raw_lattice = np.asarray([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=float)
    except Exception as exc:
        raise DescriptorError("invalid POSCAR scale or lattice") from exc
    if raw_lattice.shape != (3, 3) or abs(float(np.linalg.det(raw_lattice))) <= 1.0e-12:
        raise DescriptorError("invalid POSPÐTˆ]XÙHŠBˆYˆØØ[HOH‚ˆ˜Z\ÙH\ØÜš\Ü‘\œ›ÜŠ”ÔÐÐTˆØØ[HØ[››Ý™H™\›ÈŠBˆYˆØØ[H‚ˆ\™Ù]Ý›Û[YHHXœÊØØ[JBˆ˜XÝÜˆH
\™Ù]Ý›Û[YHÈXœÊ›Ø]
œ›[˜[Ë™]
˜]×Û]XÙJJJH
Šˆ
KŒÈËŒ
Bˆ[ÙN‚ˆ˜XÝÜˆHØØ[Bˆ]XÙHH˜]×Û]XÙH
ˆ˜XÝÜ‚‚ˆÞ[X›Û×Û[™HH[™\ÖÍWKœÜ]

BˆYˆ›ÝÞ[X›Û×Û[™HÜˆ[
Ú\×Ú[YÙ\—ÝÚÙ[ŠÚÊH›ÜˆÚÈ[ˆÞ[X›Û×Û[™JN‚ˆ˜Z\ÙH\ØÜš\Ü‘\œ›ÜŠ•TÔÔÐÐTˆÚ]Ý][[Y[Þ[X›ÛÈ\È[œÝ\ÜYŠBˆÞ[X›ÛÈHØØ[›ÛšXØ[ÜÞ[X›Û
ÚÊH›ÜˆÚÈ[ˆÞ[X›Û×Û[™WBˆžN‚ˆÛÝ[ÈHÚ[
ÚÊH›ÜˆÚÈ[ˆ[™\ÖÍ—KœÜ]

WBˆ^Ù\^Ù\[Ûˆ\È^Î‚ˆ˜Z\ÙH\ØÜš\Ü‘\œ›ÜŠš[˜[YÔÐÐTˆ]ÛHÛÝ[ÈŠHœ›ÛH^ÂˆYˆ[ŠÞ[X›ÛÊHOH[ŠÛÝ[ÊHÜˆ[žJÈ›ÜˆÈ[ˆÛÝ[ÊN‚ˆ˜Z\ÙH\ØÜš\Ü‘\œ›ÜŠ”ÔÐÐTˆÞ[X›ÛØÛÝ[Z\ÛX]ÚŠB‚ˆÝ\œÛÜˆHÂˆYˆÝ\œÛÜˆ[Š[™\ÊH[™[™\ÖØÝ\œÛÜ—KœÝš\

K›ÝÙ\Š
KœÝ\ÝÚ]
œÈŠN‚ˆÝ\œÛÜˆ
ÏHBˆYˆÝ\œÛÜˆH[Š[™\ÊN‚ˆ˜Z\ÙH\ØÜš\Ü‘\œ›ÜŠ”ÔÐÐTˆXÚÜÈÛÛÜ™[˜]H[ÙHŠBˆ[ÙHH[™\ÖØÝ\œÛÜ—KœÝš\

K›ÝÙ\Š
BˆÝ\œÛÜˆ
ÏHBˆ\™XÝH[ÙKœÝ\ÝÚ]
™ŠBˆØ\\ÚX[ˆH[ÙKœÝ\ÝÚ]
˜ÈŠHÜˆ[ÙKœÝ\ÝÚ]
šÈŠBˆYˆ›Ý
\™XÝÜˆØ\\ÚX[ŠN‚ˆ˜Z\ÙH\ØÜš\Ü‘\œ›ÜŠˆ[šÛ›ÝÛˆÔÐÐTˆÛÛÜ™[˜]H[ÙHÛ[™\ÖØÝ\œÛÜˆHWH\ŸHŠB‚ˆÝ[HÝ[JÛÝ[ÊBˆYˆ[Š[™\ÊHÝ\œÛÜˆ
ÈÝ[‚ˆ˜Z\ÙH\ØÜš\Ü‘\œ›ÜŠ”ÔÐÐTˆ\ÈÛÈ™]ÈÛÛÜ™[˜]\ÈŠBˆÛÛÜ™ÈH×Bˆ›Üˆ[™H[ˆ[™\ÖØÝ\œÛÜˆˆÝ\œÛÜˆ
ÈÝ[N‚ˆšY[ÈH[™KœÜ]

BˆYˆ[ŠšY[ÊHÎ‚ˆ˜Z\ÙH\ØÜš\Ü‘\œ›ÜŠš[˜[YÔÐÐTˆÛÛÜ™[˜]H›ÝÈŠBˆÛÛÜ™Ë˜\[™
Ù›Ø]
šY[ÖÌJK›Ø]
šY[ÖÌWJK›Ø]
šY[ÖÌ—JWJBˆÛÛÜ™×Ø\œˆHœ˜\Ø\œ˜^JÛÛÜ™Ë\OY›Ø]
BˆYˆØ\\ÚX[Ž‚ˆÛÛÜ™×Ø\œˆHÛÛÜ™×Ø\œˆ
ˆ˜XÝÜ‚ˆœ˜XÝ[Û˜[HÛÛÜ™×Ø\œˆœ›[˜[Ëš[Š]XÙJBˆ[ÙN‚ˆœ˜XÝ[Û˜[HÛÛÜ™×Ø\œ‚ˆ^[™YH\JÞ[X›Û›ÜˆÞ[X›ÛÛÝ[[ˆš\
Þ[X›ÛËÛÝ[ËÝšXÝUYJH›ÜˆÈ[ˆ˜[™ÙJÛÝ[
JBˆ™]\›ˆÝXÝ\™Q]J]XÙO[]XÙKÜXÚY\ÏY^[™Yœ˜XÝ[Û˜[Yœ˜XÝ[Û˜[Y[YšY\XÛÛ[Y[
B‚‚‚‚™YˆÚ\×Ú[YÙ\—ÝÚÙ[ŠÚÙ[ŽˆÝŠHOˆ›ÛÛ‚ˆžN‚ˆ[
ÚÙ[ŠBˆ™]\›ˆYBˆ^Ù\˜[YQ\œ›ÜŽ‚ˆ™]\›ˆ˜[ÙB