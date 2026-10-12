# Element-conditioned local geometry descriptor

Task: `paper-68ccd2058009969716b8879c`  
Paper inspiration: Salmenjoki et al., *Scientific Reports* 13, 11114 (2023), DOI `10.1038/s41598-023-37633-9`.

## Scope

`element-conditioned-local-geometry/1` is a fixed vector of 30 deterministic scalars
from one explicit, fully occupied periodic structure. The only elemental lookup is
neutral-element atomic number. No experimental target, fitted parameter, EBSD map,
grain label, dislocation density, strain history, force, energy, or learned
representation is read.

The paper motivates only the broad question of separating explicit chemical identity
from local structural heterogeneity. This implementation does not reproduce the
paper's grain-level analysis or prediction problem. A POSCAR is not a polycrystalline
EBSD map, and this descriptor is not a dislocation-density, deformation-history,
strength, or ductility predictor.

The public module is split for auditability. `element_conditioned_local_geometry_base.py`
contains the fixed science implementation. `element_conditioned_local_geometry.py`
is the public facade; it installs the documented numerical-stability rules below and
re-exports the fixed API. Both files are required and are part of the source identity.

## Periodic local shell

For every crystallographic site, the code finds the 12 nearest periodic neighbours.
If the 12th distance is degenerate, every neighbour within

```
d_12 + max(1e-5 angstrom, 2e-8 * abs(d_12))
```

is retained. The absolute term is a representation-stability tolerance: it prevents
micro-angstrom round-off in nearly symmetry-equivalent periodic images from selecting
a different shell after rigid rotation, origin translation, site reordering, or
integer-supercell replication. It is not a thermal broadening, bond uncertainty, or
physical interaction range. The tie-complete shell is geometric rather than a
bond-order assignment.

The neighbour search starts from a density-scaled cutoff and expands to at most
48 Å. It rejects overlapping periodic sites below 0.10 Å, singular cells, partial
occupancy, non-neutral species labels, more than 4096 sites, more than 2,000,000
periodic pairs, or more than 512 tied neighbours per site. These cases return typed
unavailability rather than imputed values.

For site `s`, neighbour distances are `d_sj` and the geometric shell scale is

```
mu_s = mean_j(log(d_sj)).
```

The scale-free radial residual is

```
r_sj = log(d_sj) - mu_s,
```

and radial disorder is `sqrt(mean_j(r_sj**2))`. Structure-level shell-scale
heterogeneity uses `mu_s - mean_s(mu_s)` and is therefore unchanged by uniform
isotropic scaling. `knn_distance_mean_angstrom` is the deliberately retained absolute
length channel and scales with the lattice.

## Independent elemental contrast

For central site `s` and neighbour `j`,

```
c_sj = abs(Z_j - Z_s) / 117,
```

where `Z` is neutral-element atomic number. The denominator is the largest possible
atomic-number difference among elements 1--118. This is a categorical chemical
contrast, not oxidation state, charge transfer, valence, electronegativity, bond
order, electronic structure, or a material response.

Two site channels are formed: mean central-to-neighbour atomic-number contrast and
contrast--geometry coupling `mean_j(c_sj * abs(r_sj))`.

## Element-conditioned decomposition

Four site quantities are analysed together:

1. centred log shell scale;
2. radial log-RMS disorder;
3. mean atomic-number contrast;
4. contrast--radial coupling.

For each quantity `y_s`, the code computes the mean for each central element and the
usual between-element and total site variances. `*_element_between_fraction` reports
`Var_between / Var_total` when the total variance exceeds `1e-24`. At or below that
floor the ratio is set to zero because both numerator and denominator are numerical
round-off around an exactly uniform site channel; interpreting their quotient would
be unstable. The floor is not a physical threshold and does not zero the underlying
site features.

These fractions state how much explicit site-to-site heterogeneity is aligned with
the central element label. They do not identify a host, solute, defect, phase, or
causal mechanism.

The main compact candidate,
`rarity_weighted_element_environment_contrast`, summarizes the Euclidean shift of
element-conditioned means from the structure-wide mean. Each explicit site is
weighted by central-element surprisal,

```
w_s proportional to -log(x_element(s)),
```

where `x` is the global atomic fraction. The four componentwise
`rarity_weighted_*_shift` fields expose the same quantity without vector collapse.
For a monatomic structure all shifts are zero and rarity weighting is marked inactive.
This weighting is minority-sensitive but deliberately is not called a solute score:
stoichiometric minority sublattices receive the same treatment.

A complementary family weights pairwise differences between element-conditioned
means by

```
x_a * x_b * abs(Z_a - Z_b) / 117.
```

It reports four `z_weighted_*_separation` components and one Euclidean aggregate.
The final two scalars are the RMS residual heterogeneity within central-element
classes and the total RMS site-environment heterogeneity.

## Fixed output

The ordered feature vector is:

1. `composition_element_count`
2. `composition_entropy_norm`
3. `composition_atomic_number_std_norm`
4. `composition_pairwise_atomic_number_contrast`
5. `knn_shell_size_mean`
6. `knn_tie_excess_fraction`
7. `knn_distance_mean_angstrom`
8. `site_log_shell_scale_std`
9. `site_radial_log_rms_mean`
10. `site_radial_log_rms_std`
11. `site_atomic_number_contrast_mean`
12. `site_atomic_number_contrast_std`
13. `site_contrast_radial_coupling_mean`
14. `site_contrast_radial_coupling_std`
15. `shell_scale_element_between_fraction`
16. `radial_disorder_element_between_fraction`
17. `atomic_number_contrast_element_between_fraction`
18. `contrast_geometry_element_between_fraction`
19. `rarity_weighted_shell_scale_shift`
20. `rarity_weighted_radial_disorder_shift`
21. `rarity_weighted_atomic_number_contrast_shift`
22. `rarity_weighted_contrast_geometry_shift`
23. `rarity_weighted_element_environment_contrast`
24. `z_weighted_shell_scale_separation`
25. `z_weighted_radial_disorder_separation`
26. `z_weighted_atomic_number_contrast_separation`
27. `z_weighted_contrast_geometry_separation`
28. `z_weighted_element_environment_separation`
29. `within_element_environment_rms`
30. `total_environment_heterogeneity_rms`

Every fixed feature is finite when row status is `ok`. Definition-domain and resource
failures are returned separately; failed structures are never zero-filled and there
is no target-based imputation.

## Interfaces

Single POSCAR:

```bash
python element_conditioned_local_geometry.py POSCAR --format poscar
```

JSONL records:

```bash
python element_conditioned_local_geometry.py structures.jsonl \
  --format jsonl --output descriptor.jsonl
```

Every nonempty JSONL input line produces exactly one `ok` or `unavailable` row.

## Cost and invariants

For `N` sites and `E` element types, post-neighbour work is linear in retained
directed shell edges plus `O(E^2)` for element-pair means. Search cost is bounded by
explicit atom, pair, cutoff, and tied-shell limits. The vector is invariant to site
order, origin translation, rigid rotation, and integer-supercell replication within
the declared numerical tolerance. Uniform isotropic scaling changes only the absolute
distance channel.

The frozen-development regression `dft_3d_JVASP-100493` is retained because it
contains micro-angstrom near-degenerate shell distances and site-channel variances
at round-off scale. It directly tests all four representation transformations rather
than weakening the global invariance tolerance.

## Explicit non-claims

This static descriptor is not:

- a grain-size, texture, boundary-misorientation, or EBSD descriptor;
- a geometrically necessary or statistically stored dislocation density;
- a deformation, loading, strain, slip, twinning, work-hardening, strength, or
  ductility model;
- a relaxed substitution, segregation, defect, formation-energy, or diffusion model;
- a learned network, property predictor, alloy selector, or process/component
  optimization method.

Its scientific use is limited to chemistry-conditioned local geometry already
explicit in the supplied periodic structure.
