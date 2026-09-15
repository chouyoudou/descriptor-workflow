# descriptor-workflow

A small Python toolkit for unit-cell geometry and reproducible scientific jobs.

The repository brings together basic crystal-geometry calculations, a
resource-limited execution helper, and GitHub Actions workflows with reusable
Python environments. The tools are intended as building blocks for research
scripts, rather than a complete materials-modelling package.

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

## Components

| Component | Purpose |
| --- | --- |
| [`crystal_geometry.py`](crystal_geometry.py) | Cell volume and volume per atom. |
| [`restricted_exec.py`](restricted_exec.py) | Subprocess time limits and bounded output capture. |
| [`public_env/`](public_env/) | Pinned dependencies and reusable Docker runtime caching. |
| [`request_contract.py`](request_contract.py) | Validation of job requests and execution slots. |
| [`private_io.py`](private_io.py) | Optional authenticated repository I/O for separately managed job inputs and results. |
| [`bundle_control.py`](bundle_control.py) | Immutable source-bundle materialization and result publication. |

The Actions workflows cover public tests, runtime preparation, and configured
integration jobs. Docker is required for container-based execution; the
standalone geometry example does not need it.

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

The examples and tests exercise the utilities independently of any particular
research dataset.

## Project status

This is an experimental toolkit. Interfaces may change as the execution
workflow develops. Application-specific models, datasets, and research results
are maintained separately; they are not part of this distribution.
