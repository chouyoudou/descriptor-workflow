# Illustrative POSCAR

`POSCAR_LaO6` is a deliberately idealized, fully occupied periodic structure used
to demonstrate the command-line interface. It is not the paper's refined
LaVO4 structure, not an experimental doped model, and not a claim that La is
six-coordinate in that article.

```bash
python -m site_environment_descriptor.cli \
  examples/site_environment_descriptor/POSCAR_LaO6 \
  --probes Cr,Eu
```

The fixed descriptor reads the explicit La and O sites. The optional probes are
caller-supplied neutral-radius counterfactuals in the unchanged cages; they do
not assign `Cr3+`, `Eu3+`, occupancy, or a preferred substitution site.
