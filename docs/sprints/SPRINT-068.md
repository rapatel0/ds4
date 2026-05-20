# Sprint 068: Appliance Async Serving Profile

## Status

Complete.

## Overview

Sprint 067 proved that per-step async is the preferred opt-in path, but the
deployment launcher still starts the appliance without an async serving profile.
Sprint 068 wires that measured path into the operator-facing appliance config
so practical multi-slot serving does not depend on remembering benchmark-only
flags.

## Outcome

`SHIP`.

The appliance launcher now owns an async serving profile. The example and
Kubernetes configs default to a practical 4-slot sequential profile with
`DS4_V100_ASYNC_PIPELINE_MODE=auto`, which resolves to the measured faster
`per-step` async path when `DS4_V100_ACTIVE_MICROBATCH > 1`. A V100 loopback
smoke launched through `tools/ds4-v100-run-appliance.sh` reported
`async_pipeline_mode=per-step` in `/v100/status` and returned token hex `3136`.

## Goals

1. Add `DS4_V100_ASYNC_PIPELINE_MODE` to the appliance environment contract.
2. Support `off`, `auto`, `per-step`, and `persistent` in
   `tools/ds4-v100-run-appliance.sh`.
3. Resolve `auto` to `per-step` when `DS4_V100_ACTIVE_MICROBATCH > 1`, otherwise
   `off`.
4. Update the example and Kubernetes deployment config with practical
   multi-slot defaults.
5. Validate the launcher through `--check`, `--print-command`, and a V100
   loopback smoke that proves status reports `async_pipeline_mode=per-step`.

## Non-Goals

- Making MTP commit default.
- Externally exposing the appliance.
- Adding a new API endpoint.
- Changing the replay scheduler beyond mode selection.

## Definition of Done

- [x] `bash -n tools/ds4-v100-run-appliance.sh` passes.
- [x] Local `--allow-missing --check` and `--allow-missing --print-command` cover
  the new async mode.
- [x] V100 launcher check passes against the real model and pack index.
- [x] V100 loopback launcher smoke returns token hex `3136`.
- [x] `/v100/status` reports `async_pipeline_mode=per-step` for the practical
  multi-slot config.
- [x] Sprint report and vision are updated with evidence and the next blocker.
