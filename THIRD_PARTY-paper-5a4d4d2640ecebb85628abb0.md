# Third-party methods, aliases, data, and licenses

This note separates the source paper, mature software/methods reviewed for aliases, and this repository's new engineering definitions.

## Source paper — inspiration only

- W. Li, L. Ye, C. Tu, and K. Xie, *Molecules* 2025, 30, 331. DOI: 10.3390/molecules30020331.
- License: CC BY 4.0.
- The paper contributes the scientific motivation: ordered rare-earth-phosphate lattices, local coordination, lattice reconstruction, disordered meso/macropores, and defect/catalysis experiments.
- It does **not** publish a fixed POSCAR descriptor algorithm. The 103-feature implementation here is newly defined and must not be attributed to the paper as an original method.

## Mature methods and software checked

### Zeo++

- T. F. Willems, C. H. Rycroft, M. Kazi, J. C. Meza, and M. Haranczyk, “Algorithms and tools for high-throughput geometry-based analysis of crystalline porous materials,” *Microporous and Mesoporous Materials* 149 (2012) 134–141. DOI: 10.1016/j.micromeso.2011.08.020.
- Mature aliases reviewed: largest included sphere (`Di`), largest free sphere (`Df`), included sphere along a free path (`Dif`), accessible volume, and channel dimensionality on a Voronoi network.
- Treatment here: no Zeo++ source is copied. Grid clearance and toroidal winding are independent approximations and are explicitly named `grid_*` / `axial_bottleneck_*`, not exact Zeo++ outputs.

### pymatgen and matminer

- S. P. Ong et al., “Python Materials Genomics (pymatgen): A Robust, Open-Source Python Library for Materials Analysis,” *Computational Materials Science* 68 (2013) 314–319. DOI: 10.1016/j.commatsci.2012.10.028. License: MIT.
- Mature implementations reviewed: `CrystalNN`, `VoronoiNN`, `StructureGraph`, and `LocalStructOrderParams`; matminer site/statistics fingerprints built on these ideas.
- Treatment here: deliberately not imported, keeping the task implementation NumPy-only and making the neighbor rule/formula explicit. No source is copied. Users needing chemically adaptive neighbor weights should compare against these mature implementations.

### Tetrahedral order and polyhedral distortion terminology

- P.-L. Chau and A. J. Hardwick, “A new order parameter for tetrahedral configurations,” *Molecular Physics* 93 (1998) 511–518. DOI: 10.1080/002689798169195.
- J. R. Errington and P. G. Debenedetti, “Relationship between structural order and the anomalies of liquid water,” *Nature* 409 (2001) 318–321. DOI: 10.1038/35053024.
- K. Robinson, G. V. Gibbs, and P. H. Ribbe, “Quadratic Elongation: A Quantitative Measure of Distortion in Coordination Polyhedra,” *Science* 172 (1971) 567–570. DOI: 10.1126/science.172.3983.567.
- Treatment here: tetrahedral `q` uses the conventional rescaled formula. The radial CV, normalized strain RMS, and angular RMSE are new explicit engineering summaries; they are not relabeled as quadratic elongation.

## Static element lookup

- `phosphate_framework_descriptor/elements.py` was generated from RDKit 2025.09.4 `Chem.GetPeriodicTable()` for symbols, covalent radii, van der Waals radii, and outer-electron counts.
- RDKit source: https://github.com/rdkit/rdkit
- License: BSD-3-Clause. The upstream notice is reproduced in `LICENSES/RDKit-BSD-3-Clause.txt`.
- Scientific context for common radius tables includes B. Cordero et al., “Covalent radii revisited,” DOI: 10.1039/B801115J, and S. Alvarez, “A cartography of the van der Waals territories,” DOI: 10.1039/C3DT50599E. The exact serialized values used here are attributed to RDKit's API rather than silently claiming a single paper table.

## Runtime dependency

- NumPy, BSD-3-Clause, is used for linear algebra and deterministic array operations. It is installed as a dependency; no NumPy source is vendored.

## New definitions in this task

The following are task-specific engineering definitions, not paper originals or claimed mature standards:

- radius quotient-graph translational rank and axis-span aggregate;
- supercell-stable component dominance;
- low/high-CN ligand bridge graph and its periodic rank;
- sorted-cosine octahedral RMSE;
- atom-anchored periodic clearance grid;
- integer-potential union-find activation threshold for grid winding;
- the exact 103-feature aggregation and zero policy.
