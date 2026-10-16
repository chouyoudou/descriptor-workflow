# Methods, provenance, and distinctions

## 1. Paper-derived inspiration

Rai *et al.* report monoclinic `P2₁/n` LaVO4 samples and use an assumed La-site
substitution model during Rietveld refinement. The article also discusses an
explicit nine-coordinate rare-earth environment and small refined cell-volume
changes across nominal compositions. These observations motivate generic
questions about explicit site environments and size fit.

The implementation does not make the paper's nominal `Eu3+`/`Cr3+` labels,
preparation-dependent phase assignment, optical spectra, energy transfer,
lifetime, or heating measurements into inputs or targets. Element symbols in a
POSCAR do not determine oxidation state or dopant role.

## 2. Periodic smooth contact construction

For center `i`, periodic image `j,n`, row lattice `L`, and fractional positions
`f`,

```text
d_ijn = || (f_j + n - f_i) L ||
s_ijn = d_ijn / (r_i + r_j)
```

where `r` is a pinned neutral covalent radius.

The independently defined compact gate is

```text
w(s) = 1                                                s <= s_inner
       0.5 [1 + cos(pi (s-s_inner)/(s_outer-s_inner))]  s_inner < s < s_outer
       0                                                s >= s_outer
```

with defaults `s_inner=0.95` and `s_outer=1.35`. This gate is a deterministic
geometric treatment, not bond order or interaction energy.

The exact translation search bound follows reciprocal-plane spacings. If the
required image range or pair-evaluation count exceeds configured guards, the code
raises `ResourceLimitError`; it does not silently truncate neighbors.

## 3. Explicit-site channels

With normalized local weights `p_k = w_k / sum(w)`, direction `u_k`, and distance
`d_k`:

- effective coordination: `sum(w_k)`;
- mean normalized distance: `sum p_k s_k`;
- weighted Baur-type radial distortion:
  `sum p_k |d_k-d_bar| / d_bar`;
- off-centering: `||sum p_k u_k||`;
- orientation tensor `M = sum p_k u_k u_k^T`, with descending eigenvalues
  `lambda1 >= lambda2 >= lambda3`;
- anisotropy: `1.5 sum_a (lambda_a-1/3)^2`;
- planarity: `lambda2-lambda3`;
- linearity: `lambda1-lambda2`;
- frozen-cage fitted radius: `sum p_k (d_k-r_neighbor,k)`;
- resident radius residual:
  `(r_center-r_fit)/max(r_center,r_floor)`;
- contact-strain RMS: `sqrt(sum p_k (s_k-1)^2)`;
- signed atomic-number contrast:
  `(Z_center-sum p_k Z_neighbor,k)/118`;
- signed RDKit outer-electron contrast:
  `(v_center-sum p_k v_neighbor,k)/15`;
- heteroelement contact fraction;
- effective neighbor-element count `exp(-sum_e q_e log(q_e))`.

The Baur name is retained as a mature radial-distortion baseline, while smooth
weights and all frozen-cage/aggregation treatments are independently defined
here.

## 4. Aggregation and typed missingness

Each local channel is summarized by mean, population standard deviation, and
empirical q10/q50/q90 using NumPy's `inverted_cdf` quantile method. That
inverse-CDF definition is unchanged by integer supercell replication.

No valid local contact shell means the local statistics are `null`. It does not
mean that distortion, fit radius, or chemistry contrast is physically zero.
`environment_valid_fraction`, `environment_missing_fraction`, and
`missing_reasons` preserve applicability.

For seven channels, species separation is

```text
eta² = sum_e n_e (mean_e-mean)^2 / sum_i (x_i-mean)^2.
```

It is `null` when fewer than two represented species make the ratio inapplicable,
and zero when at least two species are present but total variance is exactly zero.

## 5. Explicit probe extension

For a caller-named probe radius `r_probe`, each valid existing cage receives

```text
probe_residual = (r_probe-r_fit)/max(r_probe,r_floor).
```

Only aggregate signed/absolute residual summaries are returned. Coordinates,
neighbors, and the cage are frozen. The extension does not rank individual
sites, infer occupancy, or estimate a relaxed defect energy.

## 6. Mature aliases checked

The following established methods informed naming and overlap review, but their
source is neither copied nor imported:

- Hoppe effective coordination and pymatgen `EconNN`
  ([source](https://github.com/materialsproject/pymatgen-core/blob/main/src/pymatgen/core/local_env.py),
  DOI [`10.1524/zkri.1979.150.1-4.23`](https://doi.org/10.1524/zkri.1979.150.1-4.23)).
  The cosine radius gate here is not ECoN.
- matminer `CrystalNNFingerprint`, a local-order fingerprint, and
  `SiteStatsFingerprint`, a general site-statistics wrapper
  ([matminer](https://github.com/hackingmaterials/matminer);
  DOI [`10.1016/j.commatsci.2018.05.018`](https://doi.org/10.1016/j.commatsci.2018.05.018)).
  This implementation is not CrystalNN/Voronoi/ChemEnv.
- matminer `LocalPropertyDifference`, a mature neighbor-property-difference
  featurizer. The signed fixed contrasts here are independently coded and use
  the task's own smooth gate.
- Zimmermann local order parameters and site fingerprints
  (DOIs [`10.3389/fmats.2017.00034`](https://doi.org/10.3389/fmats.2017.00034)
  and [`10.1039/C9RA07755C`](https://doi.org/10.1039/C9RA07755C)).
- Baur distortion index
  (DOI [`10.1107/S0567740874004560`](https://doi.org/10.1107/S0567740874004560)).
- Cordero *et al.* covalent radii
  (DOI [`10.1039/B801115J`](https://doi.org/10.1039/B801115J)).

## 7. Lookup provenance and license

Neutral covalent radii and `GetNOuterElecs` values were generated from the RDKit
2025.09.4 periodic-table API, pinned as factual arrays, and attributed under
RDKit's BSD-3-Clause license. RDKit is not a runtime dependency.

The new task files are covered by
`LICENSE-paper-7a2c33e715c47a2c191a0574.txt`; the license does not relicense
unrelated repository content.

## 8. Invariances and limitations

The fixed vector is designed to be invariant to atom ordering, global fractional
translation, rigid Cartesian rotation, and diagonal integer supercell
replication. Tests and the deterministic thousand-structure smoke check those
properties.

Limitations include neutral-radius heuristics, full-occupancy input, no bond
orders, finite configured image/evaluation guards, and no relaxed or energetic
substitution model.
