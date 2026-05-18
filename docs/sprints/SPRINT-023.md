---
sprint: 023
title: Cross-GPU Two-Stage Scheduler Handoff
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
report: SPRINT-023-REPORT.md
followups: SPRINT-023-FOLLOWUPS.md
verdict: SHIP
---

# SPRINT-023: Cross-GPU Two-Stage Scheduler Handoff

## Overview

Sprint 023 extends the resident scheduler from one GPU to the first real
cross-GPU hop. The target is stage 0 on gpu0 followed by stage 1 on gpu1:
layers 0-5 execute from token embedding, HC is handed off peer-to-peer, and
layers 6-11 execute from the incoming HC.

This sprint also fixes a CUDA backend issue exposed by multi-device execution:
model-range/control-tensor caches were global by source offset and could reuse
gpu0 device pointers while running kernels on gpu1.

## Outcome Contract

- `SHIP`: stage 0 and stage 1 can both keep resident shard arenas, the
  scheduler can peer-copy HC between them, and the two-stage V100 smoke passes
  in the full appliance gate.
- `EXTEND`: the resident stage scheduler works independently, but cross-GPU
  handoff requires a relay-buffer redesign or a concrete CUDA topology fix.
- `STOP`: stage 1 cannot execute safely with the current CUDA backend without
  replacing model/cache ownership.

## Non-Goals

- No full 43-layer selected-token claim.
- No output-head selected-token gate.
- No public server.
- No MTP.
- No throughput benchmark.
- No multi-slot wavefront.

## Workstreams

| Lane | Responsibility | Write Scope | Validation |
|---|---|---|---|
| A: device context | Add explicit GPU selection before scheduler tensor work. | `ds4_gpu.*`, scheduler | local compile, V100 smoke |
| B: CUDA cache ownership | Make model-range and model-arena caches device-aware. | `ds4_cuda.cu` | two-stage smoke |
| C: scheduler handoff | Add HC handoff and decode-from-HC scheduler APIs. | `ds4_v100_scheduler.*` | gpu0 -> gpu1 walk |
| D: gate evidence | Add two-stage scheduler to the V100 gate. | `tools/ds4-v100-gate.sh`, docs | full gate |

## Implementation

### Phase 1: Device-Aware GPU Runtime

**Files:**
- `ds4_gpu.h`
- `ds4_cuda.cu`
- `ds4_metal.m`
- `ds4_gpu_arena_stub.c`

**Tasks:**
- [x] Add `ds4_gpu_set_device`.
- [x] Track CUDA tensor allocation device.
- [x] Set the tensor device before tensor fill/read/write/free.
- [x] Use `cudaMemcpyPeer` for tensor copies across devices.
- [x] Make CUDA model-range cache entries device-aware.
- [x] Make CUDA model arena allocations device-aware.

### Phase 2: Cross-Stage Scheduler API

**Files:**
- `ds4_v100_scheduler.h`
- `ds4_v100_scheduler.c`

**Tasks:**
- [x] Allow non-token stages to open resident arenas.
- [x] Keep `decode_token` restricted to the token-embedding stage.
- [x] Add `ds4_v100_stage_scheduler_handoff`.
- [x] Add `ds4_v100_stage_scheduler_decode_hc`.
- [x] Set the scheduler device before embedding and stage decode.

### Phase 3: Two-Stage Smoke And Gate

**Files:**
- `tests/cuda_v100_two_stage_scheduler_smoke.c`
- `tools/ds4-v100-gate.sh`

**Tasks:**
- [x] Open stage 0 and stage 1 schedulers.
- [x] Decode stage 0 from token embedding.
- [x] Peer-copy HC from stage 0 to stage 1.
- [x] Decode stage 1 from incoming HC.
- [x] Read final HC and validate finite nonzero output.
- [x] Add the two-stage scheduler smoke to the full V100 gate.

## Outcome

`SHIP`.

The scheduler now executes layers 0-11 across gpu0 and gpu1 with resident
stage arenas and a peer HC handoff. The full V100 gate passes with the new
two-stage scheduler check.

## Definition Of Done

- Stage 0 scheduler regression still passes.
- Two-stage scheduler executes 12 layers across two V100s.
- CUDA model/control tensor caching is device-aware.
- Full V100 gate passes and remains `ready=false` only for the known remaining
  product milestones.
