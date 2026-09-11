# descriptor-workflow

Public, reusable infrastructure for crystal-geometry utilities and bounded
scientific execution.

## Public components

- `crystal_geometry.py`: deterministic unit-cell geometry CLI.
- `restricted_exec.py`: bounded subprocess execution and output capture.
- `request_contract.py`: non-sensitive task-trigger validation, including
  descriptor, ML, and infrastructure scheduling lanes.
- `private_io.py`: scoped fetch/execute/persist transport around an optional
  private integration task. Scientific definitions and private data are not
  stored here.
- `public_env/`: pinned public Python wheels and a content-addressed Docker
  runtime cache assembled only from public packages.

## Parallel scheduling contract

A trigger commit changes exactly one public slot file:

```json
{"version":3,"id":"opaque-task-id","lane":"descriptor","slot":7}
```

The path must match its lane and slot, for example
`private_job/slots/descriptor-07.json`. Available slots are:

- `descriptor-00` through `descriptor-19`;
- `ml-00` through `ml-07`;
- `infrastructure-00` and `infrastructure-01`.

Each slot is serial and has a bounded GitHub Actions pending queue; distinct
slots may execute concurrently. A stable agent should own one descriptor slot.
The slot file contains no private path, source, data, result, or credential.

## Runtime cache

The local-environment CPU runtime is identified by the immutable base-image
digest plus the exact public lock, wheel manifest, and runtime Dockerfile.
Actions first restore and verify the complete public runtime image. On a miss it
falls back to the verified public wheelhouse, builds once, verifies installed
versions and a synthetic fingerprint smoke, then saves the public-only runtime
cache before any private source is fetched.

The cache is an optimization, not scientific evidence. Private candidate code,
inputs, outputs, logs, and credentials must never be cached.

## Local tests

```bash
python3 -m unittest discover -s tests -v
```

## Scope and policy boundary

This repository must remain a real public software project: public geometry,
execution isolation, request validation, cache construction, tests, and
documentation belong here. Private scientific implementations, structure
datasets, element tables, detailed results, and research reports do not.

Adding unrelated or cosmetic files to make a private workload appear compliant
is not an accepted strategy and provides no GitHub policy or account-safety
guarantee. Public Actions runs should genuinely test or operate this public
software.
