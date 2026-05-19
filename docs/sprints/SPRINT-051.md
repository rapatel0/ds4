# Sprint 051: Gate Aggregate Matrix Profiles

## Status

Complete.

## Overview

Sprint 051 adds explicit aggregate-throughput matrix profiles to the full
appliance gate so broad Level-6 envelope coverage can be requested from CLI
without patching scripts between runs.

## Goals

1. Preserve existing default gate behavior for fast closure runs.
2. Add a first-class `full` matrix profile for broader slot/context/policy
   coverage.
3. Allow explicit CLI overrides for matrix axes and request shape.
4. Document the profile contract in the operator runbook.

## Scope

- `tools/ds4-v100-gate.sh`:
  - add `--aggregate-profile fast|full`;
  - add CLI overrides:
    `--aggregate-ctx-tiers`, `--aggregate-slot-tiers`,
    `--aggregate-queue-policies`, `--aggregate-requests`,
    `--aggregate-tokens`, `--aggregate-host`, `--aggregate-port-base`;
  - emit resolved aggregate profile/matrix as a gate line.
- `docs/operations/DS4-V100-APPLIANCE.md`:
  - add fast/full profile defaults and an example full-profile gate command.

## Out of Scope

- New CUDA kernels.
- Runtime decode algorithm changes.
- Cluster execution of the full profile matrix.

## Definition of Done

- `tools/ds4-v100-gate.sh --help` documents the new profile/override options.
- `bash -n tools/ds4-v100-gate.sh` passes.
- Existing default aggregate configuration remains unchanged under `fast`.
- Runbook documents both profiles and the full-profile invocation path.
