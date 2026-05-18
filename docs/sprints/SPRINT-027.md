---
sprint: 027
title: V100 Selected-Token Correctness And HC Checkpoints
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
report: SPRINT-027-REPORT.md
followups: SPRINT-027-FOLLOWUPS.md
verdict: SHIP
---

# SPRINT-027: V100 Selected-Token Correctness And HC Checkpoints

## Overview

Sprint 027 converts the Sprint 026 output-head finding into a layer-body
diagnostic and fixes the first correctness issues it exposes. The sprint adds
CPU-vs-V100 HC checkpoint capture, after-attention checkpoints, route
comparison, and gate coverage for the selected-token oracle.

The shipped outcome is larger than a diagnostic-only sprint: the V100
scheduler now selects the expected official token bytes `3136` for
`short_reasoning_plain`.

## Outcome Contract

- `SHIP`: selected-token oracle passes on V100 and checkpoint diagnostics can
  localize remaining body drift.
- `EXTEND`: selected-token still fails but the first divergent layer is
  identified with useful checkpoint evidence.
- `STOP`: checkpoint instrumentation cannot run on the real model or corrupts
  the scheduler path.

## Non-Goals

- No public server.
- No MTP.
- No throughput benchmark.
- No multi-slot scheduler.
- No claim that all CPU-vs-V100 layer math is numerically identical.

## Implementation

### Phase 1: CPU Checkpoint Oracle

**Files:**
- `ds4.h`
- `ds4.c`

**Tasks:**
- [x] Add CPU source-layout HC checkpoint replay for seed, layer-final, and
  after-attention checkpoints.
- [x] Add CPU route checkpoint replay for selected expert ids and route
  weights.
- [x] Replay only through the maximum requested layer so bisection stays
  usable.

### Phase 2: Scheduler Checkpoint Plumbing

**Files:**
- `ds4_v100_scheduler.h`
- `ds4_v100_scheduler.c`
- `ds4_v100_layer_execute.h`
- `ds4_v100_layer_execute.c`

**Tasks:**
- [x] Add seed, after-attention, and layer-final checkpoint callbacks.
- [x] Surface route reports at layer-final checkpoints.
- [x] Preserve the existing non-checkpoint decode API as the default wrapper.

### Phase 3: Correctness Fixes

**Files:**
- `ds4_cuda.cu`
- `ds4_gpu.h`
- `ds4_v100_layer_execute.c`
- `ds4_v100_scheduler.c`

**Tasks:**
- [x] Decode native BF16 token embeddings correctly on V100 instead of
  treating them as F16.
- [x] Restore the default F16 KV/cache contract for V100 decode.
- [x] Keep FP8 KV as an explicit scheduler option rather than the default
  correctness path.
- [x] Round emitted compressed KV rows to the source-layout F16 contract.

### Phase 4: Cluster Gate And Evidence

**Files:**
- `tests/cuda_v100_scheduler_checkpoint_parity_smoke.c`
- `Makefile`
- `tools/ds4-v100-gate.sh`

**Tasks:**
- [x] Add a checkpoint parity smoke with layer specs such as `-1`, `N`, and
  `aN`.
- [x] Add the checkpoint smoke to the CUDA build and appliance gate.
- [x] Require `scheduler_output_head` to match expected token hex `3136`.
- [x] Remove `real_model_selected_token` from missing readiness items when the
  oracle passes.

## Outcome

`SHIP`.

The V100 selected-token path now passes the official short prompt check. The
full appliance gate remains `ready=false` only because public serving, MTP, and
throughput benchmarking are not implemented.

Checkpoint diagnostics also show that seed, early layers, and layer-4
after-attention match the CPU source-layout reference, while layer-4 final HC
still has measurable FFN numeric drift. That drift is now a performance and
precision follow-up, not the blocker for the official selected-token gate.

## Definition Of Done

- CPU checkpoint APIs build locally.
- V100 scheduler checkpoint APIs build on `sm_70`.
- Checkpoint parity smoke can compare seed, after-attention, and layer-final
  HC.
- Real-model selected-token smoke passes with expected token bytes `3136`.
- Full 8-GPU appliance gate passes with readiness missing only public serving,
  MTP, and throughput.
