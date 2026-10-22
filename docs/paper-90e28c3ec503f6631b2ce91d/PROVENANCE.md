# Provenance and validation record

## Source separation

### Paper-derived motivation

Hinterstein *et al.* study in-situ neutron diffraction of coarse-grained
functional materials and use barium titanate to analyze coexisting phases,
grain-size dependence, orientation-dependent response, and frequency-dependent
strain mechanisms. That experimental framing motivates measuring intrinsic
local geometry and directional organization while explicitly refusing to infer
polycrystalline or loading-state information from one POSCAR.

The paper's STRAP/Rietveld/orientation-series workflow, phase fractions, domain
switching strains, grain-size statistics, electric-field conditions, and
frequency response are **not reproduced** by this code. They require external
measurements and state variables unavailable in a single static structure.

### Mature concepts and implementations checked

- Baur distortion index: G. M. Baur, *Acta Crystallographica B* **30** (1974),
  1195-1215, DOI `10.1107/S0567740874004560`.
- Orientational order lineage: P. J. Steinhardt, D. R. Nelson and M. Ronchetti,
  *Physical Review B* **28** (1983), 784-805,
  DOI `10.1103/PhysRevB.28.784`.
- Centrosymmetry parameter as a related, distinct local-symmetry diagnostic:
  C. L. Kelchner, S. J. Plimpton and J. C. Hamilton,
  *Physical Review B* **58** (1998), 11085-11088,
  DOI `10.1103/PhysRevB.58.11085`.
- Pymatgen periodic-neighbor infrastructure and coordination-analysis ecosystem:
  S. P. Ong *et al.*, *Computational Materials Science* **68** (2013), 314-319,
  DOI `10.1016/j.commatsci.2012.10.028`.
- CrystalNN/LoStOP alternatives: N. E. R. Zimmermann and A. Jain,
  *RSC Advances* **10** (2020), 6063-6081,
  DOI `10.1039/C9RA07755C`.
- Matminer feature ecosystem: L. Ward *et al.*, *Computational Materials
  Science* **152** (2018), 60-69,
  DOI `10.1016/j.commatsci.2018.05.018`.

No source code was copied from those projects or publications. The present
implementation independently composes a periodic adaptive shell, a rank-2
fabric tensor, fixed scalar aggregation, typed null semantics, and an optional
atomic-number contrast weighting.

### Project-defined processing and new heuristics

- Fixed shell cutoff `1.25 * nearest periodic distance`.
- Site mean/RMS/max aggregation for first-shell distortion and off-centering.
- Linear/planar/isotropic barycentric decomposition of the rank-2 fabric.
- Bond-count-weighted local-to-global `q2` cancellation.
- Atomic-number-contrast fabric and its `q2` shift.
- Conditional paired-state logarithmic lattice change with caller-asserted
  lattice correspondence.

The final three bullets are heuristic definitions introduced for this workflow;
they should be tested for utility on independent downstream datasets rather
than treated as established physical observables.

## Formal frozen-1000 execution

- Private bundle: `bt-paper-90e28c3ec503f6631b2ce91d-v1`
- Public Actions run: `37014829265`, conclusion `success`
- Private materialized output:
  `transport/bundle-executions/bt-paper-90e28c3ec503f6631b2ce91d-v1/37014829265-1/`
- Frozen input: 1000 structures, target properties not read
- Materialized input SHA-256:
  `98ed32e4c0a62ea718bf884be8dbd8937bdd87ca23c00ce11c2b3b8506418406`
- Runtime: Python 3.12.14, NumPy 2.5.3, pymatgen 2026.5.4, four workers
- Wall time: 1.851472917 s
- Row status: 1000 `ok`, zero failed/unavailable rows
- Result SHA-256:
  `aa97f4bd0f9e5287634191ab235bf1193ecddd9f8386f0aff8a2c9458f2909cf`
- Invariance/synthetic diagnostics: passed at tolerance `2e-8`; maximum absolute
  discrepancy `6.180766216856824e-11`

Coverage of typed feature values:

- 20 of 23 features were defined for all 1000 structures.
- `first_shell_offcentering_alignment`: 945 defined; 55 typed nulls because all
  local off-centering vectors had zero norm.
- `atomic_number_contrast_fabric_q2` and
  `atomic_number_contrast_q2_shift`: 889 defined each; 111 typed nulls because
  every selected pair had zero atomic-number contrast.

Synthetic reference checks recovered exact ideal limits for a two-direction
axis, a four-direction plane, and a six-direction cubic shell. Rotation,
translation, site permutation, simple-supercell, and paired rigid-rotation
checks passed on three frozen development structures with no null mismatch.

## Reproducibility identities

Formal private source snapshot:

- private ref: `3698a297660915e9f662836283605659b4b7fc82`
- `intrinsic_directionality.py` blob:
  `ce008e5f411bfdb0592c50eb2b4676a8ea763707`
- `task_runner_impl.py` blob:
  `a2c7c565e2c67aebedcf05aa5ae9ba5282cf3e91`
- frozen input blob: `b031d4c8621a7591b301113d28b369a953f845ef`
- runtime science-source SHA-256:
  `e0440921be41c774abd95ca7a7ce91d8c0dfccf8efb5d3dcc9f2414213523387`
- runtime runner SHA-256:
  `27e7115e9c392eebbec3e2776b6433e5829f7cf61e97c1f464e6905c52743b2c`

The public module is byte-identical to the formal private science source. The
private runner and raw 1000-row results remain outside the public branch so Main
can independently inspect and validate the delivered record.
