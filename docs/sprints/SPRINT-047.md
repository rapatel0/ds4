# Sprint 047: Active-Microbatch Scheduler Core

## Status

Complete.

## Overview

Sprint 046 established slot/context admission and queue policy. Execution was
still effectively single-slot in the scheduler state model. Sprint 047 ships
the core runtime primitives for active microbatch scheduling inside
`ds4_v100_stage_scheduler`:

- per-slot KV/cache ownership;
- per-slot HC cursor ownership;
- batch decode and handoff APIs.

This sprint does not claim request-loop multi-prompt batching in HTTP yet. It
ships the scheduler substrate and validation surface needed for that next step.

## Goals

1. Make stage scheduler KV and HC state slot-aware for up to 8 active slots.
2. Add batch scheduler APIs for token-seed decode, HC decode, and stage handoff.
3. Validate batch scheduler execution in CUDA smokes.
4. Surface scheduler capability in service status/metrics without overstating
   tensor-batched kernel coverage.

## Scope

- `ds4_v100_scheduler`:
  - slot-aware cache layout (`[layer][slot]`);
  - slot-aware HC buffers/cursors;
  - `decode_token_batch`, `decode_hc_batch`, `handoff_batch`;
  - single-slot wrappers preserved.
- CUDA smokes:
  - `tests/cuda_v100_stage_scheduler_smoke.c` gets `--slots`;
  - `tests/cuda_v100_full_scheduler_smoke.c` gets `--slots`.
- Gate:
  - add `active_microbatch_scheduler` rung using 2-slot full scheduler smoke.
- Service surface:
  - report `scheduler_slots_ready=true`;
  - keep `tensor_batched_slots=false`.

## Out of Scope

- Concurrent multi-prompt request-loop batching in `tools/ds4-v100-replay --serve`.
- Tensor-batched fused kernels (this sprint uses slot-loop execution per layer).
- Throughput claims for 2/4/8 active slots.

## Definition of Done

- `ds4_v100_scheduler` compiles with new batch APIs and slot-aware state.
- Stage/full scheduler CUDA smoke sources compile with `--slots` support.
- Gate script includes an `active_microbatch_scheduler` rung.
- Status/metrics scheduler readiness reflects implemented scheduler batching.
