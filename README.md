# descriptor-workflow

Python utilities for geometric descriptors of crystal structures.

## Features

- Unit-cell volume from lattice vectors
- Volume per atom
- Reference checks for cell scaling and equivalent lattice representations

## Usage

Requires Python 3; no additional packages are needed.

```bash
python3 crystal_geometry.py examples/cells.json
```

Input lattice vectors use angstroms. The output is JSON containing the cell volume
and volume per atom, in cubic angstroms and cubic angstroms per atom respectively.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
