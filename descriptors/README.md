# Interpretable structure descriptors

## Octahedral trans-pair deformation (OTPD)

`octahedral_trans_pair.py` implements `octahedral-trans-pair-deformation/1`, a static geometry descriptor for a six-neighbour octahedral-like local shell. The six donor directions are partitioned into the three most nearly trans pairs by exhaustive deterministic matching. The three pair mean lengths are then decomposed into a non-negative pair-range amplitude, a signed one-short-pair / one-long-pair polarity, log channels, and optional donor-role couplings.

For sorted pair means `m1 <= m2 <= m3`, the signed polarity is `(2*m2-m1-m3)/(2*mean(m1,m2,m3))`. Positive means one isolated short geometrical trans pair; negative means one isolated long pair. This is not an oxidation-state, ligand-field, magnetic-anisotropy, or energy descriptor.

```bash
python3 -m descriptors.octahedral_trans_pair \
  examples/POSCAR_otpd_compressed --center-element Ni --sites
```

The formal workflow uses the project-reviewed element table with Cordero covalent radii as primary and Pyykkö single-bond radii only as an explicit fallback. Mature Baur bond-length distortion and Robinson-style cis-angle variance are reported only as baselines. The implementation is independent of SHAPE/continuous-shape-measure and pymatgen ChemEnv code.
