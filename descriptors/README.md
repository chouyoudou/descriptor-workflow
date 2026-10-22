# Periodic host--guest geometry

`periodic_host_guest_geometry.py` computes a fixed 18-scalar, static geometry
vector from a VASP 5/6 POSCAR using only the Python standard library.

```bash
python3 -m descriptors.periodic_host_guest_geometry \
  --poscar examples/POSCAR_phgg_host_guest \
  --params '{"max_grid_points":512}' \
  --pretty
```

The descriptor builds a radius-gated periodic multigraph, assigns the
translation rank of each connected component, treats every periodic component
as host, estimates host-only clearance on a canonical periodic grid, and
summarizes explicit finite guest contacts. Guest-only channels are zero when no
finite guest exists, with applicability recorded separately.

The pore metric is a grid approximation, not Zeo++ LCD/PLD. Contact fractions
are radius-normalized atom fractions, not Hirshfeld-surface fractions. No
charge, oxidation state, energetic performance, synthesis condition, missing
atom, or material property is inferred.

See [`../docs/periodic_host_guest_geometry_provenance.md`](../docs/periodic_host_guest_geometry_provenance.md)
for equations, aliases, source/license distinctions, and limitations.
