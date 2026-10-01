# descriptor-workflow

A small Python toolkit for unit-cell geometry and reproducible scientific jobs.

The repository brings together basic crystal-geometry calculations, interpretable
periodic-structure descriptors, a resource-limited execution helper, and GitHub
Actions workflows with reusable Python environments. The tools are intended as
building blocks for research scripts, rather than a complete materials-modelling
package.

## Quick start

The geometry utility uses only the Python standard library:

```bash
python3 crystal_geometry.py examples/cells.json
```

It reports the cell volume and volume per atom for each input record. For the
cubic example, the output includes:

```json
{
  "id": "cubic",
  "volume_angstrom3": 8,
  "volume_per_atom_angstrom3": 2.0
}
```

Input is a JSON array of cells. Each cell supplies an identifier, three lattice
vectors in ångströms (one vector per row), and a positive integer atom count:

```json
[
  {
    "id": "cubic",
    "lattice": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    "atom_count": 4
  }
]
```

The calculation uses the supplied cell; it does not standardize the structure
or determine a primitive cell. See [the examples](examples/cells.json) for
orthorhombic and skew cells.

The site–void coupling descriptor also uses only the standard library. It reads
an ordinary VASP 5 POSCAR and reports fixed site-level geometry plus aggregate
scalars:

```bash
python3 -m descriptors.site_void_coupling analyze \
  examples/site_void_coupling/POSCAR_open_cage \
  --center-element Ca --candidate-element Al --candidate-element Ga
```

Its off-centering reference is the centroid of an adaptive local shell. The
least-blocked local axis comes from the shell-direction second moment, and a
capped ray–sphere calculation measures forward/backward open-space contrast.
Optional candidate-element rows change only a tabulated covalent radius in the
unchanged cage. They are static geometric projections, not relaxed substitutions.

## Components

| Component | Purpose |
| --- | --- |
| [`crystal_geometry.py`](crystal_geometry.py) | Cell volume and volume per atom. |
| [`descriptors/site_void_coupling.py`](descriptors/site_void_coupling.py) | Static off-centering, local open-space, radius-fit, and site–void coupling scalars for periodic structures. |
| [`restricted_exec.py`](restricted_exec.py) | Subprocess time limits and bounded output capture. |
| [`public_env/`](public_env/) | Pinned dependencies and reusable Docker runtime caching. |
| [`request_contract.py`](request_contract.py) | Validation of job requests and execution slots. |
| [`private_io.py`](private_io.py) | Optional authenticated repository I/O for separately managed job inputs and results. |
| [`bundle_control.py`](bundle_control.py) | Immutable source-bundle materialization and result publication. |

The site–void descriptor is deliberately limited to one static periodic
structure. Its values are not pressure derivatives, force constants, bond
stiffnesses, polarization, piezoelectric tensors, migration barriers, or measured
response coefficients. The embedded covalent-radius table can be overridden by
callers that need a different declared radius protocol.

The Actions workflows cover public tests, runtime preparation, and configured
integration jobs. Docker is required for container-based execution; the
standalone geometry and site–void examples do not need it.

The integration bridge accepts both pre-hashed `private-task-bundle/2` requests
and text-first `private-task-bundle/3` requests. For text-first requests, the
workflow derives source checksums from the received UTF-8 text, records them in
the immutable source manifest and verifies them again before execution. Any
client-supplied checksums are still checked. This avoids requiring callers to
maintain a second copy of source identity while preserving exact-byte replay.
Existing input pinning, resource limits and publication rules are unchanged.

## Tests

From the repository root:

```bash
python3 -m unittest discover -s tests -v
```

The site–void tests include translation and rigid-rotation invariance, skew-cell
periodic images, POSCAR parsing, open/symmetric cage limits, and a deterministic
1,200-structure synthetic qualification with a fixed row digest. The examples
and tests are target-blind and independent of any particular research dataset.

## Project status

This is an experimental toolkit. Interfaces may change as the execution
workflow develops. Application-specific models, datasets, and research results
are maintained separately; they are not part of this distribution.
