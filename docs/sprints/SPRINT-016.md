---
sprint: 016
title: V100 Descriptor-Bound Router FFN Gate
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-016-INTENT.md
deferred: SPRINT-016-DEFERRED.md
verdict: SHIP
---

# SPRINT-016: V100 Descriptor-Bound Router FFN Gate

## Overview

Sprint 015 proved descriptor-bound FFN compute from real model bytes, but it
used a fixed expert. Sprint 016 upgrades that gate to model-selected experts by
adding source-F32 router projection from the resident arena and using the real
layer-2 hash-router table.

This still is not serving. It is the router+FFN part of a future
scheduler-owned layer slice.

## Outcome Contract

- `SHIP`: source-F32 arena matmul exists, descriptor-bound FFN smoke computes
  router logits from real `ffn_gate_inp.weight`, selects experts from real
  `ffn_gate_tid2eid`, executes all selected routed experts plus shared expert,
  compares against CPU references, and appliance gate passes.
- `EXTEND`: source-F32 matmul and router selection pass, but all-routes FFN
  composition needs documented follow-up.
- `STOP`: real router descriptor semantics conflict with existing hash-router
  code or V100 memory/runtime limits block the all-routes smoke.

## Non-Goals

- No attention, residual, norm, HC transform, or full layer output.
- No output-head logits or selected-token gate.
- No public serving, MTP, or throughput benchmark.
- No production grouped expert kernel.

## Implementation

### Phase 1: Source-F32 Arena Matmul

**Files:**
- `ds4_gpu.h`
- `ds4_cuda.cu`
- `ds4_gpu_arena_stub.c`

**Tasks:**
- [x] Add `ds4_gpu_arena_f32_matmul_f32`.
- [x] Validate arena range, row count, column count, stride, input size, and
      output size.
- [x] Implement CUDA row-reduction kernel for V100 diagnostics.
- [x] Fail closed in the non-CUDA stub.

### Phase 2: Router CPU/GPU Reference Path

**Files:**
- `tests/cuda_v100_descriptor_bound_ffn_smoke.c`

**Tasks:**
- [x] Bind and load `ffn_gate_inp.weight`.
- [x] Bind and read `ffn_gate_tid2eid`.
- [x] Compute CPU router logits and hash-router selected experts/weights.
- [x] Compute GPU router logits with source-F32 arena matmul and select
      experts/weights with `ds4_gpu_router_select_tensor`.

### Phase 3: Router-Selected FFN Composition

**Files:**
- `tests/cuda_v100_descriptor_bound_ffn_smoke.c`

**Tasks:**
- [x] Upload selected routed expert spans for all six routed experts.
- [x] Execute routed gate/up/down for all selected experts with router weights.
- [x] Preserve shared F8 expert computation.
- [x] Compare summed CPU/GPU FFN output.

### Phase 4: Gate, Validation, Closeout

**Files:**
- `tools/ds4-v100-gate.sh`
- `docs/sprints/drafts/SPRINT-016-*.log`
- `docs/sprints/SPRINT-016-REPORT.md`
- `docs/sprints/SPRINT-016-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

**Tasks:**
- [x] Run local validation.
- [x] Run cluster smoke and full appliance gate.
- [x] Archive logs.
- [x] Update sprint docs and vision.

## Definition Of Done

- Source-F32 arena matmul is implemented and linked into the CUDA build.
- Descriptor-bound FFN smoke uses real router projection and hash-router
  selection by default.
- Cluster full gate passes and reports `ready=false`.
- Sprint report, follow-ups, logs, and vision update are committed.

## Risks

- Hash-router table shape semantics must match `ds4_gpu_router_select_tensor`
  expectations: token-major rows of 6 selected experts.
- Running six routed experts increases smoke runtime, but it remains bounded to
  one token and one layer.
