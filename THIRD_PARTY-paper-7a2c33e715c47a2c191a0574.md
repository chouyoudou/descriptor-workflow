# Third-party provenance for paper-7a2c33e715c47a2c191a0574

## Vendored factual element data

`site_environment_descriptor/elements.py` contains element symbols, neutral
covalent radii (`GetRcovalent`), and neutral outer-electron counts
(`GetNOuterElecs`) generated from the RDKit 2025.09.4 periodic-table API.

- Upstream: https://github.com/rdkit/rdkit
- Upstream license: BSD-3-Clause
- Exact notice: `LICENSES/RDKit-BSD-3-Clause.txt`
- Runtime relationship: none; RDKit is not imported or installed by this task.
- Nature of use: factual lookup arrays plus independent wrapper code.

## Runtime dependency

NumPy is installed from the pinned task requirement range and used through its
public Python API. No NumPy source is vendored.

- Upstream: https://github.com/numpy/numpy
- License: BSD-3-Clause

## Consulted mature implementations; no source copied

- pymatgen-core `EconNN`, `CrystalNN`, and local-environment infrastructure:
  https://github.com/materialsproject/pymatgen-core (MIT).
- matminer `CrystalNNFingerprint`, `SiteStatsFingerprint`, and
  `LocalPropertyDifference`: https://github.com/hackingmaterials/matminer
  (BSD-3-Clause).
- VASP POSCAR format specification:
  https://vasp.at/wiki/POSCAR. The parser is independently written.

These projects were consulted to identify mature aliases, expected distinctions,
and input semantics. They are not runtime dependencies and no implementation
source was copied.

## Scientific references

- Rai et al., DOI 10.1039/d2ra06962h — material-specific structural inspiration.
- Hoppe effective coordination, DOI 10.1524/zkri.1979.150.1-4.23 — mature alias
  check; the implementation here is not ECoN.
- Baur distortion index, DOI 10.1107/S0567740874004560 — named radial baseline.
- Cordero et al. covalent radii, DOI 10.1039/B801115J — radius provenance context.
- Ward et al., DOI 10.1103/PhysRevB.96.024104; Zimmermann et al.,
  DOI 10.3389/fmats.2017.00034 and 10.1039/C9RA07755C; matminer,
  DOI 10.1016/j.commatsci.2018.05.018 — mature site/structure feature context.

No article prose, figures, tables, spectra, or experimental data are copied into
the implementation.
