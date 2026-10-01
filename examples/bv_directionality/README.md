# Bond-valence directionality and radius-contact descriptors

This example accompanies `descriptors/bv_directionality.py`. The implementation
accepts an ordered VASP 5 POSCAR and emits a fixed set of structure-level
scalars plus auditable site records. It deliberately keeps two scientific
layers separate.

## 1. General radius-contact layer (default)

For a center `i` and a periodic neighbour `j`, let

```text
q_ij = d_ij / (r_i + r_j)
w_ij = exp[-max(q_ij - 1, 0) / tau]
I_i  = |sum_j w_ij u_ij| / sum_j w_ij
```

where `d_ij` is the distance, `r_i` and `r_j` are fixed single-bond covalent
radii, `u_ij` points from the center to the neighbour, and `tau` is the
reported `contact_softness_ratio`. `I_i` is named
`contact_vector_imbalance`. It is bounded by zero and one and is exactly zero
for an exactly balanced shell. The opposite unit vector defines a geometric
`open_axis_cartesian` when the moment is non-zero.

The embedded radius lookup follows the covalent-radius set of Cordero *et al.*,
*Dalton Transactions* (2008), DOI `10.1039/B801115J`, as commonly distributed
for atomistic geometry tools. A radius is an empirical size convention, not an
oxidation-state-specific ionic radius. Values for poorly characterized very
heavy elements are estimates. The lookup is fixed so that results are
reproducible, and every output states the radius protocol.

This layer is an original engineering descriptor. It must **not** be called a
bond-valence sum (BVS), bond-valence-sum mismatch (BVSM), bond-valence site
energy (BVSE), electron density, a lone-pair calculation, or a migration
barrier.

Additional site quantities are:

- `effective_coordination = (sum w)^2 / sum(w^2)`;
- weighted mean and dispersion of `q_ij`;
- `cavity_radius_angstrom`, the weighted mean of `d_ij-r_j`;
- `radius_fit_mismatch`, the normalized difference between that cavity radius
  and `r_i`;
- a point one ångström opposite the contact moment, summarized only by nearest
  atomic-surface clearance; and
- `opposed_void_support`, the product of imbalance and capped positive
  clearance.

The virtual point is a geometric probe. It is not a localized electron pair or
an atom.

## 2. Explicit bond-valence layer (opt-in)

The opt-in protocol implements only the local equations

```text
s_ij = exp[(R0_ij - d_ij) / B_ij]
BVS_i = sum_j s_ij
BVSM_i = |BVS_i - |V_i||
Psi_i = sum_j s_ij u_ij
```

The raw `Psi_i` is paper-aligned. The additional dimensionless
`bond_valence_vector_imbalance = |Psi_i|/BVS_i` is a newly introduced
normalization for comparing sites with different total bond-valence weight.
The equations are evaluated with oxidation states and pair-specific `R0`/`B`
supplied by the caller. No oxidation state is inferred from the formula,
element, or geometry. Every pair row must carry a non-empty source string. The
bundled JSON is explicitly synthetic and exists only to demonstrate the file
contract.

The preferred general source for literature parameters is the IUCr
`bvparmxxxx.cif` collection maintained for the Commission on Inorganic and
Mineral Structures. That collection warns that `R0`, `B`, oxidation state,
and the distance convention are coupled, and that unchecked entries require
care. A scientific run should archive the exact parameter file/version and
selection rule used.

If a site is explicitly listed in `candidate_site_indices` and
`|Psi_i| > vector_cutoff`, the program reports a point at
`dummy_offset_angstrom` opposite `Psi_i`. This follows the geometric direction
used by Shaw *et al.*, *Journal of Applied Crystallography* (2025), DOI
`10.1107/S1600576725009574`, but the present implementation reports only point
clearance. It does not add the paper's empirical BVSM penalty, construct a
voxel map, calculate the Morse/screened-Coulomb BVSE field, or perform
percolation analysis.

## Prior implementations and names

| Resource | What it implements | License/provenance handling here |
| --- | --- | --- |
| **Lone Pair BV** (`robbie6shaw/lone-pair-BV`, inspected at commit `a4bdeb78f334b6e71450863662d79fb343c5581f`) | Author implementation of BVSM/BVSE maps and empirical lone-pair corrections used by the 2025 paper. | No repository license file was observed during the audit. No source code or bundled parameter database was copied. |
| **softBV** | Bond-valence site-energy fields, migration networks, and screening. The academic command-line license permits research/education use under its stated conditions and requires citation. | Not vendored or called. This descriptor does not claim softBV-compatible energies or barriers. |
| **BVlain** | Python BVSE/BVSM volumetric maps and percolation barriers. | MIT-licensed mature alternative; not vendored or called. |
| **IUCr bond-valence parameter files** | Literature `R0`/`B` values and source metadata. | No table is bundled. Users provide selected values explicitly. |

This separation prevents a radius-decay analogue from being silently relabelled
as BVS and prevents a static directional moment from being interpreted as a
real lone pair or kinetic pathway.

## Usage

Analyze the geometric layer for the center atom in the open-cage example:

```bash
python3 descriptors/bv_directionality.py analyze \
  examples/bv_directionality/POSCAR_open_cage \
  --site-indices 0 \
  --output out/open-cage.json
```

Add the synthetic explicit protocol:

```bash
python3 descriptors/bv_directionality.py analyze \
  examples/bv_directionality/POSCAR_open_cage \
  --site-indices 0 \
  --explicit-bv-protocol \
    examples/bv_directionality/explicit_bv_protocol.synthetic.json \
  --output out/open-cage-with-explicit-bv.json
```

Run the fixed-seed 1,000-structure qualification (five families × 200):

```bash
python3 descriptors/bv_directionality.py benchmark \
  --count-per-family 200 \
  --seed 20251001 \
  --output-dir out/bv-directionality-1000
```

The benchmark is synthetic and target-blind. It varies coordination family,
element sizes, shell orientation, a lengthened contact, and a small center
shift. It checks determinism, finite output, directional response, fallback
frequency, and outliers. It does not validate conductivity or a barrier.

## Input and limitations

- ordered VASP 5 POSCAR only;
- Direct or Cartesian coordinates; triclinic cells are supported;
- no partial occupancies, site splitting, disorder resolution, or symmetry
  expansion;
- periodic neighbours are generated explicitly up to the reported cutoff;
- default contacts use covalent radii and are not chemical bonds;
- explicit BVS results are only as meaningful as the supplied oxidation states,
  parameter provenance, and cutoff protocol;
- one static structure cannot determine carrier concentration, correlated
  hopping, framework relaxation, polarizability, temperature dependence,
  conductivity, or an activation barrier.
