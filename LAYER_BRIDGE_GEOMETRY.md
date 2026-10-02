# Static layer, coordination, and bridge geometry

`layer_bridge_geometry.py` turns one fully occupied VASP 5-style POSCAR into a
fixed vector of 21 inexpensive, interpretable scalars.  It is intended for
screening and diagnostics when the only trusted input is the supplied periodic
structure and a small, explicit element table.

The implementation is deliberately dependency-free and does **not** estimate
pressure response, magnetic order, exchange constants, elasticity, energetic
stability, transport, kinetics, synthesis history, or an experimentally
realized phase.  Different POSCARs are different input phases; the code neither
relaxes nor standardizes them.

## Scientific provenance and naming

The immediate structural inspiration is Geers *et al.*, “High-pressure
behavior of the magnetic van der Waals molecular framework Ni(NCS)2,”
*Physical Review B* **108**, 144439 (2023), DOI
[10.1103/PhysRevB.108.144439](https://doi.org/10.1103/PhysRevB.108.144439).
That article discusses stacked two-dimensional metal–ligand layers,
metal–ligand–metal pathways, an interlayer pathway, and a metal-centered hinge
quantity involving `cos(omega/2)`.  This module uses those ideas only to motivate
generic **static geometry** summaries.  It does not reproduce the paper's
pressure derivatives, equation-of-state fits, magnetic model, DFT results, or
material-specific pathway labels.

Related mature terminology and implementations were checked before naming the
new quantities:

| Prior art / alias | What it provides | Relation to this module | Source / license |
| --- | --- | --- | --- |
| Connected-component dimensionality; Larsen component method | Bonded periodic components, dimensionality, and orientation | Same broad problem class. This module independently computes the rank of periodic graph-cycle translations; it does not compute the Larsen dimensionality score. | Larsen *et al.*, DOI [10.1103/PhysRevMaterials.3.034003](https://doi.org/10.1103/PhysRevMaterials.3.034003). `pymatgen.analysis.dimensionality.get_structure_components` is a mature implementation; pymatgen is MIT licensed. |
| Rank-determination algorithm (RDA); `ase dimensionality` | Dimensionality analysis over bonding intervals | “RDA” is a useful search alias. No ASE code is imported or copied. | ASE documentation; ASE is GNU LGPL. |
| Topology-scaling algorithm | Cluster-size scaling under supercells | An alternative dimensionality family; not implemented here. | Ashton *et al.*, DOI [10.1103/PhysRevLett.118.106101](https://doi.org/10.1103/PhysRevLett.118.106101). |
| ChemEnv; continuous symmetry measure (CSM) | Reference-polyhedron coordination labels and distortion measures | More complete local-environment classification. This module instead exposes low-cost CN, angle, bond-CV, and direction-moment summaries. | Waroquiers *et al.*, DOI [10.1107/S2052520620007994](https://doi.org/10.1107/S2052520620007994); pymatgen implementation under MIT. |
| `NCS` / `SCN` | Chemical linkage notation for thiocyanate/isothiocyanate connectivity | It is **not** treated as a generic descriptor acronym. The bridge path search is element-agnostic and can represent M–N–C–S–M, M–S–C–N–M, or unrelated ligand paths when those bonds are present. | Standard coordination-chemistry linkage notation; no material-specific label is inferred. |

No third-party source code is included.  The repository had no root `LICENSE`
file at the implementation baseline, so this addition does not assert or alter
a project license.  The numerical covalent radii are factual table values based
on Cordero *et al.*, “Covalent radii revisited,” DOI
[10.1039/B801115J](https://doi.org/10.1039/B801115J); values are embedded through
curium (Z=96).  Later elements require an explicit override.

## Input contract

The parser accepts a VASP 5-style POSCAR with:

- one universal positive scale or one negative target-volume scale;
- three row lattice vectors;
- explicit element symbols and positive counts;
- optional `Selective dynamics`;
- `Direct`, `Cartesian`, or `Kpoints`-style Cartesian mode;
- full occupancy only.

A conventional POSCAR does not encode oxidation state, spin state, bond order,
partial occupancy, molecular identity, pressure, temperature, or sample
history.  The implementation does not guess any of them.  VASP 4 files without
an element-symbol line and three-component scale-factor syntax are rejected.

## Periodic bond multigraph

For sites `i` and `j` and lattice-image translation `t`, an undirected edge is
added when

```text
minimum_distance < |r_j + t - r_i|
                 <= bond_scale * (R_i + R_j)
```

where `R` is the tabulated covalent radius.  The default `bond_scale` is 1.18
and the default minimum distance is 0.35 Å.  All periodic images inside a bound
derived from the three reciprocal cell heights are considered, so skew cells
are supported.  Multiple image edges between the same quotient-cell sites are
retained because their translations carry dimensionality information.

This bond rule is a transparent heuristic, not a valence or energy model.  Use
`--cutoff-ensemble` to rerun fixed scales 1.10, 1.18, and 1.26 and inspect every
descriptor's min/max/span.

## Component and layer descriptors

For each connected quotient-graph component, a traversal assigns an integer
lattice offset to every reached site.  Every graph closure produces an integer
translation mismatch.  The real-vector rank of the independent closure
translations is reported as component dimensionality: 0, 1, 2, or 3.

For a 2D component, two independent translation vectors define a primitive
integer normal `h = (h,k,l)`.  With row lattice matrix `A`, the crystallographic
normal repeat is

```text
d_hkl = 1 / || A^(-T) h ||.
```

Every atom in the supplied cell is projected onto the resulting circle of
length `d_hkl` and expanded by its covalent radius.  The largest uncovered
circular interval is the reported clearance.  Thus explicit guests reduce the
clearance.  It is a one-dimensional hard-sphere projection, **not** an exact
pore diameter, van der Waals surface, exfoliation energy, or mechanical gap.

| Descriptor | Definition |
| --- | --- |
| `framework_dimensionality_max` | Maximum periodic translation rank among components. |
| `framework_dimensionality_mean` | Atom-weighted mean component rank. |
| `layered_atom_fraction` | Fraction of atoms belonging to rank-2 components. |
| `mixed_dimensionality_entropy` | `-sum(f_d ln f_d)/ln(4)` over atom fractions in ranks 0–3. |
| `layer_repeat_angstrom` | `d_hkl` for the largest rank-2 component; zero if absent. |
| `interlayer_clearance_angstrom` | Largest uncovered circular projected interval. |
| `interlayer_clearance_fraction` | Clearance divided by `d_hkl`. |
| `bond_direction_anisotropy` | Rotation-invariant second-moment anisotropy over all inferred bond directions. |

For unit bond vectors `u`, the direction tensor is `Q = mean(u outer u)` and
anisotropy is `sqrt(3/2 * (Tr(Q^2) - 1/3))`, ranging from zero for an isotropic
set to one for a collinear set; an isotropic plane gives 0.5.

## Metal–ligand coordination descriptors

A fixed element-role table classifies metalloids/nonmetals as ligand-like and
other supported elements as metal-like.  This is only a structural role default.
Every role and radius can be overridden in JSON.  A metal site's coordination
number counts inferred bonds to ligand-like neighbors; direct metal–metal bonds
are not included in the coordination summaries.

| Descriptor | Definition |
| --- | --- |
| `metal_fraction` | Metal-like site fraction. |
| `metal_coordination_mean`, `metal_coordination_std` | Mean and population standard deviation of ligand CN. |
| `metal_ligand_bond_cv_mean` | Mean sitewise coefficient of variation of metal–ligand lengths. |
| `metal_coordination_anisotropy_mean` | Mean second-moment direction anisotropy of each metal's ligand vectors. |
| `metal_ligand_hinge_angle_mean_deg` | Mean, over metal sites, of all unordered ligand–metal–ligand pair angles. |
| `metal_ligand_hinge_cos_half_mean` | Corresponding mean of `cos(theta/2)`.  This is an inspiration-driven generic extension, not the paper's selected hinge coordinate. |
| `metal_ligand_radius_mismatch_mean` | Mean `abs(R_m-R_l)/(R_m+R_l)` over metal–ligand bonds. |
| `metal_ligand_atomic_number_contrast_mean` | Mean `abs(Z_m-Z_l)` over metal–ligand bonds. |

## Bridge-path descriptors

The code enumerates unique periodic paths

```text
metal – (one to max_bridge_atoms ligand-like sites) – metal
```

with no intervening metal-like site.  Paths are unique up to reversal and a
uniform lattice translation.  The default permits one through four internal
atoms, so both one-atom M–L–M bridges and longer M–N–C–S–M-style paths are in
scope.  Search stops at 10,000 unique paths by default and reports a truncation
flag; this limit is configurable.

| Descriptor | Definition |
| --- | --- |
| `bridge_path_incidence_per_metal` | `2 * unique_paths / metal_sites`, the mean endpoint incidence of unique periodic bridge paths. |
| `bridge_internal_atoms_mean` | Mean number of ligand-like internal sites. |
| `bridge_straightness_mean` | Mean endpoint chord length divided by full path length, in `[0,1]`. |
| `single_atom_bridge_angle_mean_deg` | Mean M–L–M angle for paths with one internal atom. |

These are connectivity and geometry counts only.  A short or straight path is
not automatically a strong interaction, superexchange pathway, or bond.

## Element-table override

An override file is a JSON object keyed by element symbol.  Fields may replace
the built-in values:

```json
{
  "Bk": {"atomic_number": 97, "covalent_radius": 1.70, "is_metal": true},
  "Si": {"is_metal": true}
}
```

Use it with `--element-table override.json`.  Unsupported elements never receive
a silently guessed radius.

## CLI

```bash
python3 layer_bridge_geometry.py examples/POSCAR.layer-bridge
python3 layer_bridge_geometry.py examples/POSCAR.layer-bridge --cutoff-ensemble
```

The output contains the fixed descriptor map, graph/component diagnostics,
parameters, a path-truncation flag, and an explicit scope warning.

## Validation

`tests/test_layer_bridge_geometry.py` checks:

- constructed 0D, 1D, 2D, and 3D periodic graphs;
- atom-order, wrapped-translation, rigid-rotation, and in-plane-supercell invariance;
- layer clearance and its reduction by an explicit gap guest;
- metal coordination, bridge incidence, path straightness, and bridge angle;
- POSCAR modes, negative target-volume scale, explicit unsupported-element overrides, cutoff sensitivity, deterministic edge ordering, CLI behavior, and concise errors;
- a frozen 1,000-case synthetic set with 250 known examples of each rank, 21,000 finite descriptor values, and a checked SHA-256 descriptor digest.

The frozen set is a deterministic algorithmic regression, not evidence for a
materials-property correlation.  Large-dataset validation and comparison with
alternative neighbor definitions remain separate acceptance work.
