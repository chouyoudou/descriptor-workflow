# Elemental-reference STP heat-capacity descriptor

`elemental_reference_heat_capacity.py` defines six fixed, target-blind scalars from an explicit periodic structure and an independently tabulated pure-element property. The implementation accepts a VASP-5-style POSCAR or the repository's JSON structure-record convention.

## Scientific meaning

For element \(i\), let \(x_i=n_i/N\) be its atomic fraction and let \(c_i\) be the tabulated **molar heat capacity of the pure element at STP** in J mol(element atoms)\(^{-1}\) K\(^{-1}\). The table is the `HeatCapacityMolar.table` distributed with matminer 0.10.1 and documented there as sourced from the CRC Handbook.

The descriptor contains:

1. `ref_cp_stp_mean_j_mol_atom_k`
   \[
   \bar c = \sum_i x_i c_i
   \]

2. `ref_cp_stp_std_j_mol_atom_k`
   \[
   \sigma_c = \sqrt{\sum_i x_i(c_i-\bar c)^2}
   \]

3. `ref_cp_stp_range_j_mol_atom_k`
   \[
   \max_i c_i-\min_i c_i
   \]

4. `ref_cp_stp_pairwise_abs_j_mol_atom_k`
   \[
   \sum_{i,j}x_i x_j|c_i-c_j|
   =2\sum_{i<j}x_i x_j|c_i-c_j|
   \]

5. `ref_cp_stp_relative_pairwise_abs`
   \[
   \frac{\sum_{i,j}x_i x_j|c_i-c_j|}{\bar c}
   \]

6. `ref_cp_stp_density_proxy_mj_m3_k`
   \[
   \frac{N\bar c}{N_A V}\,10^{24}
   \]
   where \(V\) is the cell volume in Å³ and \(N_A\) is the exact Avogadro constant. The conversion gives MJ m\(^{-3}\) K\(^{-1}\).

The last column is deliberately named a **density proxy**. It combines a real unit-cell number density with additive pure-element references. It is invariant under an integer supercell, but it is not the measured or calculated volumetric heat capacity of the compound.

## What it does not claim

The implementation does not calculate compound heat capacity, phonons, force constants, anharmonicity, electronic heat capacity, Gibbs or Helmholtz free energy, phase stability, phase fractions, transition temperatures, or a CALPHAD assessment. It performs no fit and reads no response target. A pure-lithium thermodynamic model cannot be transferred to an arbitrary structure by this descriptor.

## Definition domain and missing values

The upstream table has finite values for 85 elements. If any constituent element lacks a value, the structure receives typed unavailability:

```json
{
  "category": "definition_domain",
  "code": "missing_element_reference"
}
```

There is no elemental-mean imputation, target imputation, zero fill, or silent row deletion. Canonical neutral element symbols are required. Oxidation-state labels, isotope aliases, and dummy species are not silently converted.

## POSCAR semantics

The POSCAR parser supports:

- one standard positive scale factor;
- one negative scale factor interpreted as the requested cell volume;
- explicit VASP-5 element and count lines;
- Direct or Cartesian coordinates;
- optional `Selective dynamics` and trailing coordinate flags.

VASP-4 files without element symbols are rejected rather than guessing species from a POTCAR or filename. Three-component POSCAR scale factors are not accepted by this fixed parser.

The lattice is validated before the volume-dependent feature is emitted. A scale-aware determinant tolerance rejects singular cells. The Frobenius-norm condition number produces a warning above \(10^8\) and a typed error above \(10^{12}\).

## Interfaces

Single POSCAR:

```bash
python elemental_reference_heat_capacity.py POSCAR --format poscar
```

Frozen JSONL structures:

```bash
python elemental_reference_heat_capacity.py structures.jsonl \
  --format jsonl --output descriptor.jsonl
```

Every JSONL input line produces one output line with `status: "ok"` or `status: "unavailable"`. A JSONL command returns exit code 1 if any row is unavailable, while preserving all rows.

Python:

```python
from elemental_reference_heat_capacity import descriptor_from_record

result = descriptor_from_record({
    "lattice": [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
    "species": ["Li"],
    "fractional_coordinates": [[0, 0, 0]],
})
```

## Cost and invariants

The implementation is deterministic and has no neighbor search. Cost is \(O(N_\mathrm{sites}+N_\mathrm{elements}^2)\), with at most 118 element types. Features are invariant to site order, translation, rigid rotation of the lattice, and integer supercell replication. The volume proxy changes under a genuine volume change, as intended.

## Source and license audit

- Upstream project: matminer 0.10.1
- Pinned upstream commit: `25cd8f2778d73e54f80b38ee1983a28643368118`
- Table: `src/matminer/utils/data_files/magpie_elementdata/HeatCapacityMolar.table`
- Upstream documentation: `src/matminer/utils/data_files/magpie_elementdata/README.txt`
- Documented property/condition: molar heat capacity at STP
- Documented units: J/mol-K
- Documented underlying source: CRC Handbook
- Upstream software/data distribution license: BSD 3-Clause

The embedded rows are redistributed with the applicable matminer copyright and BSD 3-Clause notice in `THIRD_PARTY_ELEMENTAL_CP_NOTICE.md`.
