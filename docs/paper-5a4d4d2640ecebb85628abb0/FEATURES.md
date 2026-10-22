# Fixed feature vector

The canonical machine-readable order is [`feature_names.json`](feature_names.json). The vector contains 103 finite floats:

| Index range | Block | Meaning |
|---:|---|---|
| 1–14 | Composition/cell | volume and number density per atom, unique elements, metal/f-block fractions, atomic-number/radius/outer-electron statistics |
| 15–32 | Radius bond graph | coordination, bond lengths, hetero-edge fraction, normalized component/cycle summaries, quotient-graph winding rank |
| 33–58 | Local polyhedra and role chemistry | candidate fractions, radial/angle/directional distortions, tetrahedral `q`, octahedral residuals, low/high-center lookup contrasts |
| 59–76 | Ligand-mediated framework | bridge-ligand fractions, low/high graph topology and periodicity, shared-ligand corner/edge/face counts |
| 77–103 | Periodic atomistic void | clearance distribution, fixed-probe fractions, axial bottleneck radii, grid `Di`/`Df` analogues, channel-axis fractions |

Every missing subset is represented by zero-valued aggregates rather than `NaN`, while the presence fractions (for example `cn4_center_fraction`) preserve whether a statistic had supporting sites.
