# Site–void coupling examples

These two deliberately synthetic POSCARs isolate the geometry tested by
`descriptors.site_void_coupling`:

- `POSCAR_symmetric_cage` places Ca at the centroid of a regular CaO6 cage.
  The off-centering and site–void coupling are zero within floating-point
  precision.
- `POSCAR_open_cage` shifts Ca by 0.2 Å toward +x and moves the +x oxygen to
  4.8 Å from the cage center.  The adaptive first shell contains the five
  nearer oxygen atoms; the least-blocked axis and the off-centering vector
  both point toward +x, so the signed coupling is positive.

Run from the repository root:

```bash
python3 -m descriptors.site_void_coupling analyze \
  examples/site_void_coupling/POSCAR_open_cage \
  --center-element Ca --candidate-element Al --candidate-element Ga
```

The output is a fixed JSON schema with site-level diagnostics and aggregated
scalars.  Candidate-element rows replace only the tabulated central radius in
the unchanged cage.  They are geometric projections, not relaxed substitution
predictions.
