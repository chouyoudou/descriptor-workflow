# Periodic framework–polyhedron–void descriptors

**Paper inspiration:** W. Li, L. Ye, C. Tu, and K. Xie, “Porous Single-Crystalline Rare Earth Phosphates Monolith to Enhance Catalytic Activity and Durability,” *Molecules* **30** (2025) 331, DOI: [10.3390/molecules30020331](https://doi.org/10.3390/molecules30020331), CC BY 4.0.

This implementation turns only the paper's **structural vocabulary**—phosphate connectivity, local coordination polyhedra, and pore geometry—into a generic, fixed-length descriptor for a fully occupied periodic structure. It is not a reproduction of a descriptor proposed by the paper: the paper reports synthesis, characterization, and catalytic tests rather than a POSCAR descriptor algorithm.

## What is implemented

`phosphate_framework_descriptor` returns **103 deterministic scalars** in five blocks:

1. composition and representation-normalized cell density;
2. radius-bond quotient-graph coordination and periodic winding;
3. local coordination-polyhedron shape and distortion, including tetrahedral `q` and octahedral ideality residuals;
4. ligand-mediated coupling between low- and high-coordination centers, including corner/edge/face sharing summaries and periodic bridge-graph rank;
5. periodic atomistic void clearance, fixed-probe void fractions, and grid bottleneck radii along the three lattice directions.

The exact ordered feature list is in [`feature_names.json`](feature_names.json). Formulae and alias boundaries are in [`METHODS.md`](METHODS.md).

## Hard scope boundary

The input is an ordinary fully occupied POSCAR plus static element lookup data. Therefore the implementation **does not infer**:

- the paper's approximately 130 nm pore morphology or mercury-intrusion porosity;
- centimeter-scale monolith shape, surface area, facet exposure, or processing state;
- oxygen-vacancy concentration, reduction history, atmosphere, temperature, or reaction history;
- catalytic conversion, selectivity, yield, coking resistance, or durability;
- oxidation states, charges, spin states, or electronic structure.

The grid-void block describes only geometrical free space in the supplied periodic atomic cell. Its `grid_largest_*_sphere` names intentionally distinguish it from exact Voronoi-network Zeo++ values and from experimental porosity.

## Minimal use

```bash
python -m phosphate_framework_descriptor.cli POSCAR \
  --output descriptor.json
```

Python API:

```python
from phosphate_framework_descriptor import read_poscar, featurize_structure

structure = read_poscar("POSCAR")
result = featurize_structure(structure)
print(result.feature_names)
print(result.values)
```

VASP-4 POSCAR files do not contain element symbols. The parser refuses to guess chemistry; pass symbols explicitly with `--species` or `species_override`.

## Verification

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python scripts/run_phosphate_framework_smoke.py \
  --n 1000 \
  --output-dir artifacts/paper-5a4d4d2640ecebb85628abb0
```

The smoke suite covers 1000 deterministic synthetic structures across five families: phosphate-like low/high-CN bridge motifs, tetrahedral units, octahedral units, dense binaries, and open channels. It checks finite fixed-length output plus atom-order and uniform fractional-translation invariance. It is structural software verification, **not** a property benchmark.

## License and provenance

New implementation code is BSD-3-Clause; see `LICENSE-paper-5a4d4d2640ecebb85628abb0.txt`. Static element lookup arrays were generated from RDKit's `PeriodicTable` API and retain the upstream BSD-3-Clause notice in `LICENSES/RDKit-BSD-3-Clause.txt`. No Zeo++, pymatgen, or matminer source code is copied. See `THIRD_PARTY-paper-5a4d4d2640ecebb85628abb0.md` for the distinction between mature aliases, reviewed alternatives, and newly defined engineering features.
