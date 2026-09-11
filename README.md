# descriptor-workflow

Public, reusable utilities for crystal-geometry calculation and bounded GitHub
Actions execution.

## Public software

- `crystal_geometry.py`: validated unit-cell volume and volume-per-atom CLI.
- `restricted_exec.py`: bounded subprocess execution with timeout and captured
  output limits.
- `public_env/`: exact public Python dependency lock, wheel manifest, and
  content-addressed wheelhouse verification.
- `request_contract.py`: validates a deliberately non-sensitive trigger envelope
  and derives the corresponding private task descriptor path.
- `private_io.py`: scoped transport used by an optional repository-owner
  integration test. Scientific source, input data, detailed logs, and results
  are not stored in this public repository or public artifacts.

The Actions workflows test and exercise these actual public utilities. This
repository is not a facade for unrelated computation, and its existence does
not create an exemption from GitHub's terms or security boundaries.

## Trigger contracts

Legacy serial requests use:

```json
{"id": "direct-example-01"}
```

They resolve to the historical private `transport/active_task.json` pointer.

New task preparation should use the version-2 envelope:

```json
{"version": 2, "id": "agent-a-0001"}
```

This derives a create-only private task descriptor at
`transport/task_requests/agent-a-0001.json`. The public request contains no
private source path, data, result, or credential. Execution remains serialized
until the version-2 path has been qualified under concurrent runs.

## Dependency cache

The integration workflow verifies the exact lock and every wheel hash before
installation. Its current cache key includes OS, architecture, Python ABI, and
the content of both the resolved lock and wheel manifest. A one-way exact
legacy-cache fallback avoids an unnecessary redownload during migration; no
fuzzy restore key is used.

Only public dependency wheels may enter this cache. Private task files,
candidate code, structures, element tables, captured output, and results must
never be cached.

## Geometry CLI

```bash
python3 crystal_geometry.py examples/cells.json
```

Input lattice vectors use angstroms. Output volume is in cubic angstroms and
volume per atom in cubic angstroms per atom.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
