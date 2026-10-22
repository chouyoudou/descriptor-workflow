"""Cheap elemental-reference heat-capacity descriptors for periodic structures.

The columns combine explicit composition/cell volume with independent pure-element
molar heat-capacity values documented at STP.  They are reference proxies only:
not a compound heat capacity, phonon/free-energy calculation, phase model, or
CALPHAD assessment.  Missing elemental references are never imputed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

SCHEMA = "elemental-stp-heat-capacity-reference/1"
AVOGADRO_CONSTANT_MOL_INV = 6.02214076e23
CONDITION_WARNING_FROBENIUS = 1.0e8
CONDITION_ERROR_FROBENIUS = 1.0e12
MAX_SITES = 1_000_000
FEATURE_NAMES = (
    "ref_cp_stp_mean_j_mol_atom_k",
    "ref_cp_stp_std_j_mol_atom_k",
    "ref_cp_stp_range_j_mol_atom_k",
    "ref_cp_stp_pairwise_abs_j_mol_atom_k",
    "ref_cp_stp_relative_pairwise_abs",
    "ref_cp_stp_density_proxy_mj_m3_k",
)
REFERENCE_METADATA = {
    "property": "pure-element molar heat capacity reference",
    "units": "J mol(element atoms)^-1 K^-1",
    "state_and_temperature": (
        "STP as documented by the upstream matminer Magpie data README; "
        "the table is a pure-element reference, not a compound property"
    ),
    "upstream_project": "matminer",
    "upstream_version": "0.10.1",
    "upstream_commit": "25cd8f2778d73e54f80b38ee1983a28643368118",
    "upstream_table": "src/matminer/utils/data_files/magpie_elementdata/HeatCapacityMolar.table",
    "upstream_documentation": "src/matminer/utils/data_files/magpie_elementdata/README.txt",
    "upstream_source_named_by_documentation": "CRC Handbook",
    "upstream_license": "BSD-3-Clause",
    "missing_policy": "strict typed unavailability; no elemental-mean, zero, or target imputation",
}

# Atomic-number order. The upstream table is one row per atomic number.
_ALL_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_ALL_ELEMENT_SYMBOLS = frozenset(_ALL_SYMBOLS)
# Finite rows from matminer 0.10.1 HeatCapacityMolar.table, J mol^-1 K^-1.
_CP_ROWS = """
H 28.836;He 20.786;Li 24.86;Be 16.443;B 11.087;C 8.517;N 29.124;O 29.378;F 31.304;Ne 20.786
Na 28.23;Mg 24.869;Al 24.2;Si 19.99;P 23.824;S 22.7;Cl 33.949;Ar 20.786;K 29.6;Ca 25.929
Sc 25.52;Ti 25.06;V 24.89;Cr 23.35;Mn 26.32;Fe 25.1;Co 24.81;Ni 26.07;Cu 24.44;Zn 25.39
Ga 26.03;Ge 23.222;As 24.64;Se 25.363;Br 75.69;Kr 20.786;Rb 31.06;Sr 26.79;Y 26.53;Zr 25.36
Nb 24.6;Mo 24.06;Ru 24.06;Rh 24.98;Pd 25.98;Ag 25.35;Cd 26.02;In 26.74;Sn 26.99;Sb 25.23
Te 25.73;I 54.43;Xe 20.786;Cs 32.21;Ba 28.07;La 27.11;Ce 26.94;Pr 27.2;Nd 27.45;Sm 29.54
Eu 27.66;Gd 37.03;Tb 28.91;Dy 28.16;Ho 27.15;Er 28.12;Tm 27.03;Yb 26.74;Lu 26.86;Hf 25.73
Ta 25.36;W 24.27;Re 25.48;Os 24.7;Ir 25.1;Pt 25.86;Au 25.418;Hg 27.983;Tl 26.32;Pb 24.7
Bi 25.52;Rn 20.786;Ac 27.2;Th 27.32;U 27.665
"""
ELEMENTAL_CP_STP_J_MOL_K = {
    item.split()[0]: float(item.split()[1])
    for item in _CP_ROWS.replace("\n", ";").split(";") if item.strip()
}
_SYMBOL_RE = re.compile(r"^[A-Z][a-z]?$", re.ASCII)


class DescriptorUnavailable(RuntimeError):
    """Machine-readable scientific non-result."""
    def __init__(self, category: str, code: str, message: str):
        super().__init__(message)
        self.category, self.code, self.message = category, code, message

    def as_dict(self) -> dict[str, str]:
        return {"category": self.category, "code": self.code, "message": self.message}


def _unavailable(category: str, code: str, message: str) -> DescriptorUnavailable:
    return DescriptorUnavailable(category, code, message)


def _finite(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise _unavailable("input", "non_numeric_value", f"{name} must be real") from exc
    if not math.isfinite(out):
        raise _unavailable("input", "non_finite_value", f"{name} must be finite")
    return out


def _symbol(value: Any) -> str:
    if not isinstance(value, str) or not _SYMBOL_RE.fullmatch(value):
        raise _unavailable(
            "input", "invalid_element_symbol",
            "use canonical neutral symbols (for example Li or Fe); aliases, oxidation labels, isotopes, and dummy species are not stripped",
        )
    if value not in _ALL_ELEMENT_SYMBOLS:
        raise _unavailable("input", "unknown_element_symbol", f"unknown element: {value}")
    return value


def _lattice_matrix(value: Any) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise _unavailable("input", "invalid_lattice", "lattice must contain three vectors")
    rows: list[list[float]] = []
    for i, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 3:
            raise _unavailable("input", "invalid_lattice", "each lattice vector must have three entries")
        rows.append([_finite(x, f"lattice[{i}]") for x in row])
    return rows


def _det(a: Sequence[Sequence[float]]) -> float:
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def _lattice_metrics(lattice: Any) -> tuple[list[list[float]], float, float, list[str]]:
    a = _lattice_matrix(lattice)
    determinant = _det(a)
    norms = [math.sqrt(sum(x * x for x in row)) for row in a]
    scale = norms[0] * norms[1] * norms[2]
    if scale <= 0.0 or abs(determinant) / scale <= 1.0e-12:
        raise _unavailable("input", "singular_lattice", "lattice vectors are singular or nearly coplanar")
    inv = [
        [(a[1][1]*a[2][2]-a[1][2]*a[2][1])/determinant,
         (a[0][2]*a[2][1]-a[0][1]*a[2][2])/determinant,
         (a[0][1]*a[1][2]-a[0][2]*a[1][1])/determinant],
        [(a[1][2]*a[2][0]-a[1][0]*a[2][2])/determinant,
         (a[0][0]*a[2][2]-a[0][2]*a[2][0])/determinant,
         (a[0][2]*a[1][0]-a[0][0]*a[1][2])/determinant],
        [(a[1][0]*a[2][1]-a[1][1]*a[2][0])/determinant,
         (a[0][1]*a[2][0]-a[0][0]*a[2][1])/determinant,
         (a[0][0]*a[1][1]-a[0][1]*a[1][0])/determinant],
    ]
    cond = math.sqrt(sum(x*x for row in a for x in row)) * math.sqrt(sum(x*x for row in inv for x in row))
    if not math.isfinite(cond) or cond > CONDITION_ERROR_FROBENIUS:
        raise _unavailable("numerical", "ill_conditioned_lattice", f"Frobenius condition estimate {cond:.6g} exceeds {CONDITION_ERROR_FROBENIUS:.0e}")
    warnings = []
    if cond > CONDITION_WARNING_FROBENIUS:
        warnings.append({"code": "large_lattice_condition", "message": f"large Frobenius condition estimate: {cond:.6g}"})
    return a, abs(determinant), cond, warnings


def _validate_coordinates(coords: Any, count: int) -> None:
    if coords is None:
        return
    if not isinstance(coords, Sequence) or isinstance(coords, (str, bytes)) or len(coords) != count:
        raise _unavailable("input", "coordinate_count_mismatch", "coordinate count must equal species count")
    for i, row in enumerate(coords):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 3:
            raise _unavailable("input", "invalid_coordinates", "each coordinate must have three entries")
        for j, value in enumerate(row):
            _finite(value, f"coordinates[{i}][{j}]")


def compute_descriptor(lattice: Any, species: Sequence[Any]) -> dict[str, Any]:
    if not isinstance(species, Sequence) or isinstance(species, (str, bytes)) or not species:
        raise _unavailable("input", "empty_species", "species must be a non-empty sequence")
    if len(species) > MAX_SITES:
        raise _unavailable("resource", "site_limit", f"site count exceeds {MAX_SITES}")
    symbols = [_symbol(x) for x in species]
    matrix, volume, condition, warnings = _lattice_metrics(lattice)
    counts = Counter(symbols)
    missing = sorted(symbol for symbol in counts if symbol not in ELEMENTAL_CP_STP_J_MOL_K)
    if missing:
        raise _unavailable(
            "definition_domain", "missing_element_reference",
            "missing pure-element STP heat-capacity reference for " + ", ".join(missing) + "; strict policy forbids imputation",
        )
    total = len(symbols)
    weighted = [(count / total, ELEMENTAL_CP_STP_J_MOL_K[symbol]) for symbol, count in sorted(counts.items())]
    mean = sum(weight * value for weight, value in weighted)
    variance = sum(weight * (value - mean) ** 2 for weight, value in weighted)
    std = math.sqrt(max(0.0, variance))
    values = [value for _, value in weighted]
    pairwise = sum(w1 * w2 * abs(v1 - v2) for w1, v1 in weighted for w2, v2 in weighted)
    features = {
        FEATURE_NAMES[0]: mean,
        FEATURE_NAMES[1]: std,
        FEATURE_NAMES[2]: max(values) - min(values),
        FEATURE_NAMES[3]: pairwise,
        FEATURE_NAMES[4]: pairwise / mean,
        FEATURE_NAMES[5]: total * mean * 1.0e24 / (AVOGADRO_CONSTANT_MOL_INV * volume),
    }
    if any(not math.isfinite(value) for value in features.values()):
        raise _unavailable("numerical", "non_finite_result", "descriptor produced a non-finite value")
    return {
        "schema": SCHEMA,
        "features": features,
        "metadata": {
            "atom_count": total,
            "element_counts": dict(sorted(counts.items())),
            "cell_volume_angstrom3": volume,
            "lattice_condition_number_frobenius": condition,
            "warnings": warnings,
            "reference": dict(REFERENCE_METADATA),
            "semantic_limits": [
                "reference aggregates are not the actual heat capacity of the compound",
                "the density proxy assumes only additive pure-element references and is not measured volumetric heat capacity",
                "no phonons, temperature dependence, Gibbs/free energy, phase stability, or CALPHAD fitting is evaluated",
            ],
        },
    }


def descriptor_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise _unavailable("input", "invalid_record", "record must be a JSON object")
    if "lattice" not in record or "species" not in record:
        raise _unavailable("input", "missing_structure_field", "record requires lattice and species")
    species = record["species"]
    if not isinstance(species, Sequence) or isinstance(species, (str, bytes)):
        raise _unavailable("input", "invalid_species", "species must be a sequence")
    _validate_coordinates(record.get("fractional_coordinates"), len(species))
    return compute_descriptor(record["lattice"], species)


def _tokens(line: str) -> list[str]:
    return line.split("#", 1)[0].split()


def _parse_poscar(text: str) -> tuple[list[list[float]], list[str]]:
    if not isinstance(text, str):
        raise _unavailable("input", "invalid_poscar", "POSCAR must be text")
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        raise _unavailable("input", "truncated_poscar", "POSCAR is too short")
    scale_tokens = _tokens(lines[1])
    if len(scale_tokens) != 1:
        raise _unavailable("definition_domain", "unsupported_poscar_scale", "only one-component POSCAR scale is supported")
    scale = _finite(scale_tokens[0], "POSCAR scale")
    if scale == 0.0:
        raise _unavailable("input", "zero_poscar_scale", "POSCAR scale cannot be zero")
    raw = []
    for index in range(2, 5):
        tokens = _tokens(lines[index])
        if len(tokens) < 3:
            raise _unavailable("input", "invalid_poscar_lattice", "POSCAR lattice row needs three values")
        raw.append([_finite(x, "POSCAR lattice") for x in tokens[:3]])
    raw_volume = abs(_det(raw))
    if raw_volume <= 0.0:
        raise _unavailable("input", "singular_lattice", "POSCAR raw lattice is singular")
    factor = scale if scale > 0 else (-scale / raw_volume) ** (1.0 / 3.0)
    lattice = [[factor * x for x in row] for row in raw]
    symbols = _tokens(lines[5])
    if not symbols or all(re.fullmatch(r"[+-]?\d+", token) for token in symbols):
        raise _unavailable("definition_domain", "missing_poscar_species_symbols", "VASP-4 POSCAR without explicit symbols is unsupported")
    symbols = [_symbol(token) for token in symbols]
    count_tokens = _tokens(lines[6])
    if len(count_tokens) != len(symbols):
        raise _unavailable("input", "poscar_species_count_mismatch", "species and count columns differ")
    try:
        counts = [int(token) for token in count_tokens]
    except ValueError as exc:
        raise _unavailable("input", "invalid_poscar_counts", "POSCAR counts must be integers") from exc
    if any(count < 0 for count in counts) or sum(counts) < 1:
        raise _unavailable("input", "invalid_poscar_counts", "POSCAR counts must be nonnegative and nonempty")
    cursor = 7
    if lines[cursor].lstrip().lower().startswith("s"):
        cursor += 1
    if cursor >= len(lines):
        raise _unavailable("input", "missing_poscar_mode", "POSCAR coordinate mode is missing")
    mode = lines[cursor].lstrip().lower()
    if not (mode.startswith("d") or mode.startswith("c") or mode.startswith("k")):
        raise _unavailable("input", "invalid_poscar_mode", "coordinate mode must be Direct or Cartesian")
    cursor += 1
    total = sum(counts)
    if len(lines) < cursor + total:
        raise _unavailable("input", "truncated_poscar_coordinates", "POSCAR has too few coordinate rows")
    coords = []
    for line in lines[cursor:cursor + total]:
        tokens = _tokens(line)
        if len(tokens) < 3:
            raise _unavailable("input", "invalid_poscar_coordinates", "coordinate row needs three values")
        coords.append([_finite(x, "POSCAR coordinate") for x in tokens[:3]])
    _validate_coordinates(coords, total)
    species = [symbol for symbol, count in zip(symbols, counts) for _ in range(count)]
    return lattice, species


def descriptor_from_poscar_text(text: str) -> dict[str, Any]:
    lattice, species = _parse_poscar(text)
    return compute_descriptor(lattice, species)


def descriptor_from_poscar(path: str | Path) -> dict[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise _unavailable("input", "unreadable_poscar", f"cannot read POSCAR: {exc}") from exc
    return descriptor_from_poscar_text(text)


def _result_row(identifier: Any, function) -> dict[str, Any]:
    try:
        result = function()
    except DescriptorUnavailable as exc:
        return {"id": str(identifier), "status": "unavailable", "error": exc.as_dict()}
    except Exception as exc:
        return {"id": str(identifier), "status": "unavailable", "error": {"category": "algorithm", "code": "unexpected_error", "message": f"{type(exc).__name__}: {exc}"}}
    return {"id": str(identifier), "status": "ok", **result}


def _run_jsonl(path: Path, stream) -> int:
    unavailable = False
    try:
        handle = path.open("r", encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        print(f"error: cannot read input: {exc}", file=sys.stderr)
        return 2
    with handle:
        for index, line in enumerate(handle):
            identifier = f"row-{index}"
            try:
                record = json.loads(line)
                if isinstance(record, Mapping):
                    identifier = record.get("id", identifier)
                row = _result_row(identifier, lambda r=record: descriptor_from_record(r))
            except json.JSONDecodeError as exc:
                row = {"id": identifier, "status": "unavailable", "error": {"category": "input", "code": "invalid_json", "message": str(exc)}}
            unavailable |= row["status"] != "ok"
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return 1 if unavailable else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--format", choices=("poscar", "jsonl"), default="poscar")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    stream, close = sys.stdout, False
    if args.output is not None:
        try:
            stream, close = args.output.open("w", encoding="utf-8"), True
        except OSError as exc:
            print(f"error: cannot open output: {exc}", file=sys.stderr)
            return 2
    try:
        if args.format == "jsonl":
            return _run_jsonl(args.input, stream)
        row = _result_row(args.input.name, lambda: descriptor_from_poscar(args.input))
        stream.write(json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return 0 if row["status"] == "ok" else 1
    finally:
        if close:
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
