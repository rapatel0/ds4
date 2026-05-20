# Sprint 063: Stage Wavefront Scheduler Proof

## Status

Complete.

## Overview

Sprint 062 showed that the practical serving path is dominated by serialized
stage execution. Four active slots increase serialized stage time but do not
raise aggregate tok/s. The next sprint should prove the runtime can address
independent slot lanes and overlap stage work across GPUs without changing
model math.

This sprint adds the scheduler primitives needed for stage wavefronting and a
bounded V100 smoke that compares wavefront slot-lane execution against the
existing serial path.

## Goals

1. Add slot-addressable scheduler entrypoints for decode-token, decode-HC, and
   handoff batch operations.
2. Preserve all existing zero-based batch APIs and default serving behavior.
3. Add a slot-addressable HC readback helper for parity smokes.
4. Add a bounded V100 stage-wavefront smoke that exercises two independent
   slot lanes through at least two stages and compares against the serial path.
5. Use the smoke result to decide whether Sprint 064 should wire wavefronting
   into the served sustained benchmark.

## Out of Scope

- Production HTTP wavefront serving by default.
- MTP draft commit.
- Kernel rewrites.
- Changing default queue/admission semantics.

## Definition of Done

- Local syntax/object builds pass for touched files.
- `git diff --check` passes.
- Existing V100 scheduler and selected-token smokes still pass.
- New V100 wavefront smoke passes on the cluster.
- Sprint report records whether the stage-lane mechanics are correct enough to
  proceed to an opt-in served wavefront path.

## Outcome

`SHIP`. The scheduler now exposes slot-addressable decode-token, decode-HC,
handoff, and HC-read entrypoints while preserving the existing zero-based APIs.
The CUDA backend's temporary scratch buffer is now per device, which removes a
global scratch hazard for cross-device staged work.

The new V100 smoke proves two independent slot lanes can be advanced through
two stages in wavefront order and match the serial reference exactly:
`max_abs_slot0=0`, `max_abs_slot1=0`.

Sprint 064 should wire these primitives into an opt-in served wavefront path for
same-length, non-MTP sustained decode, then compare against the Sprint 062
profiled baseline.
