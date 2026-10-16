# Periodic site-environment and frozen-cage geometry

This task adds a dependency-light, target-blind descriptor family inspired by the
structural discussion in Rai *et al.*, RSC Advances 13 (2023) 4182–4194,
DOI [`10.1039/d2ra06962h`](https://doi.org/10.1039/d2ra06962h).

The paper's material-specific refinement and optical/thermal observations are not
treated as universal labels. The implementation reads only an explicit, fully
occupied periodic structure and a pinned neutral-element table.

## Public interfaces

### Fixed core descriptor

`featurize_structure(structure)` returns `periodic-site-environment/1`, a fixed
94-entry vector:

- 17 composition, cell, coverage, and contact-weight scalars;
- 14 explicit-site channels, each summarized by mean, population standard
  deviation, and replication-invariant empirical q10/q50/q90;
- 7 species-separation (`eta²`) channels.

Undefined domains are JSON `null`, not physical zero. Coverage and missing-reason
metadata accompany the vector.

### Explicit probe extension

`evaluate_probe_elements(structure, ["Cr", "Eu"])` is separate from the fixed
vector and runs only on caller-named elements. It compares the named neutral
covalent radius with each unchanged explicit cage. It does **not** identify a
dopant, choose a site, assign a valence, relax a structure, or estimate a
substitution energy.

## Use

Only NumPy is required.

```bash
python -m pip install -r requirements-paper-7a2c33e715c47a2c191a0574.txt
python -m site_environment_descriptor.cli POSCAR \
  --probes Cr,Eu \
  --output site_environment.json
```

For VASP-4 POSCAR files without element names, chemistry is never guessed:

```bash
python -m site_environment_descriptor.cli POSCAR.vasp4 \
  --vasp4-species La,V,O
```

Python API:

```python
from site_environment_descriptor import (
    evaluate_probe_elements,
    featurize_structure,
    read_poscar,
)

structure = read_poscar("POSCAR")
fixed = featurize_structure(structure)
explicit = evaluate_probe_elements(structure, ["Cr", "Eu"])
```

## Scientific scope

The values are static geometric and neutral-lookup summaries. They are not:

- oxidation states, charges, bond orders, spin states, or electronic levels;
- inferred dopant identities, occupancies, concentrations, or substitution sites;
- defect or substitution energies, relaxed strain, solubility, or stability;
- band gaps, photoluminescence, energy transfer, lifetimes, optical heating, or
  any other optical/thermal response;
- phase-formation, particle-size, processing, or synthesis-history predictions.

Ordinary POSCAR input is treated as fully occupied. Partial occupancy and disorder
are not invented.

See [METHODS.md](METHODS.md), [FEATURES.md](FEATURES.md), and the task-specific
third-party notice at the repository root.
