# Methods, definitions, aliases, and limits

## 1. Input contract

The descriptor accepts a fully occupied periodic structure with lattice matrix \(L\), fractional coordinates \(f_i\), and element symbols. Coordinates are wrapped into \([0,1)^3\). No charge, oxidation state, magnetic state, defect label, surface termination, or experimental metadata is accepted.

A fixed static lookup provides atomic number, covalent radius, van der Waals radius, outer-electron count, metal flag, f-block flag, and a conservative ligand-priority flag. Radius arrays were generated from RDKit 2025.09.4 `Chem.GetPeriodicTable()`; they are lookup inputs, not learned values.

## 2. Radius bond graph

For sites \(i,j\) and periodic image \(t\in\mathbb{Z}^3\), an undirected quotient-graph edge is added when

\[
0.25\ \text{Å} \leq \left\|(f_j+t-f_i)L\right\| \leq
s\,(r_i^{\rm cov}+r_j^{\rm cov}),
\]

with default \(s=1.25\). The image search range is derived from reciprocal plane spacings and capped by configuration. Self-image edges are retained once per \(\pm t\) pair; this is necessary to represent periodic one-site networks.

Reported graph summaries include coordination statistics, edge density, normalized bond length \(d/(r_i+r_j)\), component density, cyclomatic number per atom, and quotient-graph translational rank. Translational rank is obtained by assigning integer image potentials during graph traversal and taking the matrix rank of non-zero cycle translations. Rank 0–3 is a geometrical periodic-network dimensionality, not a transport dimensionality.

`component_dominance = largest_component_size × n_components / n_nodes` is used instead of raw largest-component fraction because it remains unchanged when identical zero-dimensional components are repeated in an integer supercell.

## 3. Local polyhedron candidates

A site is a polyhedron candidate when it has coordination 3–12 and at least 60% of its neighbors are ligand-like. A neighbor is ligand-like when it is in a conservative ligand-priority element set or has covalent radius no more than 92% of the center radius. Candidate centers are divided into:

- low-CN centers: CN 3–5;
- high-CN centers: CN 6–12;
- exact CN4 and CN6 subsets for tetrahedral and octahedral shape scores.

This is a generic geometry/lookup rule. It does not assert formal anions, cations, valences, or phosphate identity.

For each candidate, the implementation computes:

- radial coefficient of variation \(\sigma(d)/\bar d\);
- RMS normalized bond strain \(\sqrt{\langle(d/(r_i+r_j)-1)^2\rangle}\);
- directional second-moment anisotropy
  \[
  A=\sqrt{\frac{3}{2}\sum_{k=1}^3(\lambda_k-1/3)^2},
  \]
  where \(\lambda_k\) are eigenvalues of the bond-unit-vector second moment;
- for CN4, the rescaled tetrahedral order parameter
  \[
  q=1-\frac{3}{8}\sum_{j<k}(\cos\theta_{jk}+1/3)^2,
  \]
  plus angular RMSE from 109.4712°;
- for CN6, RMSE of sorted pair cosines from the ideal octahedral multiset \(\{-1,-1,-1,0\times12\}\), plus angular RMSE from twelve 90° and three 180° angles.

The tetrahedral `q` name follows the Chau–Hardwick / Errington–Debenedetti convention. Radial CV, normalized strain, and angle RMSE are **not** claimed to be Robinson quadratic elongation or continuous-symmetry measures; their names state their actual formulas.

## 4. Ligand-mediated framework coupling

An explicit coordinating atom is any non-center bonded to a polyhedron candidate. For each such ligand, low- and high-CN neighboring centers are collected. A low/high center edge is created through each shared ligand, carrying the composed periodic image shift. The implementation reports:

- fraction and multiplicity of bridge ligands;
- center–ligand saturation fractions;
- edge density, cyclomatic number, component dominance, axis span, and translational rank of the low/high bridge graph;
- pair fractions sharing one, two, or at least three ligands, named corner-, edge-, and face-sharing **topological counts**;
- low/low, high/high, and low/high pair fractions.

The sharing labels describe common ligand indices in the supplied graph. They are not energetic bond classifications.

## 5. Periodic atomistic void grid

A deterministic fractional grid is attached to a canonically selected atom, making the sampling origin invariant to atom ordering and uniform fractional translation. At grid point \(x\), surface clearance is

\[
c(x)=\min_i\left[d_{\rm PBC}(x,f_i)-r_i^{\rm vdW}\right].
\]

The nearest-image search checks the local 27-image neighborhood, supporting skewed cells without an orthorhombic assumption. Reported values include clearance distribution statistics and fractions satisfying \(c(x)\ge r_p\) for fixed probe radii 0, 0.5, 1.0, 1.2, 1.4, and 1.8 Å.

To estimate channel bottlenecks, grid points are activated from largest to smallest clearance. An integer-potential disjoint-set structure detects the first non-contractible cycle along each lattice axis on the periodic three-torus. The associated activation clearances are the three `axial_bottleneck_*_radius` values.

Aliases and non-equivalence:

- `grid_largest_included_sphere_diameter = 2 max(c)` is a grid analogue of a largest included sphere (`Di`).
- `grid_largest_free_sphere_diameter = 2 max(axial bottleneck radius)` is a grid analogue of a largest free sphere (`Df`).
- These values are **not exact Zeo++ Voronoi-network `Di`, `Df`, or `Dif`**, and convergence depends on grid spacing.
- They characterize periodic atomistic void space, not the paper's disordered ~130 nm monolith pores or experimental surface area/porosity.

## 6. Determinism and representation tests

The test suite checks:

- finite output and exactly 103 ordered features;
- ideal tetrahedral and octahedral limits;
- atom-order invariance;
- uniform fractional-translation invariance;
- aligned integer-supercell invariance on a reference structure;
- quotient-graph periodic rank on a one-site periodic network;
- monotonic atomistic void response to cell expansion;
- explicit refusal to infer VASP-4 chemistry without supplied symbols.

Grid descriptors are numerical approximations. Non-aligned supercell grids can differ at finite resolution; convergence should be assessed by decreasing `grid_spacing` for quantitative use.
