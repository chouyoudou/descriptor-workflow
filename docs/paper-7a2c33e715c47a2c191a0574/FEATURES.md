# Fixed feature contract (`periodic-site-environment/1`)

The order is frozen in `site_environment_descriptor.descriptor.FEATURE_NAMES`
and mirrored in [`feature_names.json`](feature_names.json).

## Global block: 17 scalars

| Name | Meaning |
|---|---|
| `structure_valid` | Parsed explicit periodic structure flag (`1`). |
| `volume_per_atom_ang3` | Cell volume divided by explicit site count. |
| `number_density_per_ang3` | Explicit site count divided by cell volume. |
| `unique_element_count` | Number of explicit element symbols. |
| `covalent_radius_mean_ang`, `covalent_radius_std_ang` | Composition-weighted neutral-radius moments. |
| `atomic_number_mean`, `atomic_number_std` | Composition-weighted atomic-number moments. |
| `outer_electrons_mean`, `outer_electrons_std` | RDKit neutral outer-electron lookup moments. |
| `environment_valid_fraction`, `environment_missing_fraction` | Local-shell coverage. |
| `contact_weight_per_site_mean`, `contact_weight_per_site_std` | Moments of `sum(w)` including true zero-contact sites. |
| `heteroelement_contact_weight_fraction`, `same_element_contact_weight_fraction` | Directed smooth-contact weight fractions; `null` when total weight is zero. |
| `multi_species_structure` | `1` for at least two explicit species, otherwise `0`. |

## Local block: 70 scalars

Each of the following 14 site channels is aggregated by suffixes
`_mean`, `_std`, `_q10`, `_q50`, `_q90`:

| Channel | Unit / interpretation |
|---|---|
| `effective_coordination` | Smooth contact-weight count. |
| `mean_normalized_distance` | Dimensionless `d/(r_i+r_j)`. |
| `radial_distortion_index` | Dimensionless weighted Baur-type absolute radial deviation. |
| `offcentering` | Dimensionless norm of weighted direction first moment, nominally `[0,1]`. |
| `orientation_anisotropy` | Dimensionless second-moment anisotropy, isotropic `0`, linear `1`. |
| `orientation_planarity` | `lambda2-lambda3`. |
| `orientation_linearity` | `lambda1-lambda2`. |
| `frozen_cage_fit_radius_ang` | Å; weighted `d-r_neighbor`. |
| `resident_radius_residual_rel` | Signed relative neutral-radius mismatch. |
| `contact_strain_rms` | RMS deviation of normalized contacts from `1`. |
| `atomic_number_contrast` | Signed center-minus-neighbor contrast divided by `118`. |
| `outer_electron_contrast` | Signed center-minus-neighbor contrast divided by `15`. |
| `heteroelement_fraction` | Local smooth-contact fraction to a different symbol. |
| `local_element_effective_count` | Exponential Shannon effective count of neighbor symbols. |

All five summaries are `null` when no site has a valid weighted shell.

## Species-separation block: 7 scalars

`species_eta2_...` is supplied for:

- `effective_coordination`
- `mean_normalized_distance`
- `offcentering`
- `orientation_anisotropy`
- `frozen_cage_fit_radius_ang`
- `resident_radius_residual_rel`
- `heteroelement_fraction`

These values summarize how much of the explicit-site variance is separated by
the element symbol already present in the POSCAR. They do not infer chemical
roles, charge states, or dopant labels. The channel is `null` for fewer than two
represented explicit species.

## Optional probes are not columns

Caller-supplied probe results intentionally have a separate variable-length
schema (`explicit-frozen-cage-probes/1`). They are not appended to the 94 fixed
columns and cannot silently make a material-specific element list part of the
general descriptor.
