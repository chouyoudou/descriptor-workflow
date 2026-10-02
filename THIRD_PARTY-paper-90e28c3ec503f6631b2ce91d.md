# Third-party and literature record for paper-90e28c3ec503f6631b2ce91d

This contribution is an independent implementation. It calls public APIs from
NumPy and pymatgen but does not vendor or copy their source code.

| Item | Role | License / terms | Source |
|---|---|---|---|
| Attached J. Appl. Cryst. paper | Scientific motivation and terminology only | CC BY 4.0, as marked by the publisher | DOI `10.1107/S1600576723005940` |
| pymatgen | `Structure`, POSCAR parsing, and periodic neighbor search | MIT | <https://github.com/materialsproject/pymatgen> |
| NumPy | Array algebra and symmetric eigensolver | BSD-3-Clause | <https://github.com/numpy/numpy> |
| Baur (1974) | Historical distortion-index definition | Citation; no code copied | DOI `10.1107/S0567740874004560` |
| Steinhardt-Nelson-Ronchetti (1983) | Orientational-order lineage and alias check | Citation; no code copied | DOI `10.1103/PhysRevB.28.784` |
| Kelchner-Plimpton-Hamilton (1998) | Related centrosymmetry diagnostic checked for overlap | Citation; no code copied | DOI `10.1103/PhysRevB.58.11085` |
| CrystalNN/LoStOP paper | Mature coordination/local-order alternative checked | Citation; no code copied | DOI `10.1039/C9RA07755C` |
| matminer paper | Mature descriptor ecosystem checked | Citation; no code copied | DOI `10.1016/j.commatsci.2018.05.018` |

The implementation deliberately does not claim that its rank-2 fabric is the
full Steinhardt `Q_l` family, that its adaptive shell is CrystalNN, or that its
off-centering scalar is a polarization. Those names are retained only to make
method boundaries and prior-art checks explicit.
