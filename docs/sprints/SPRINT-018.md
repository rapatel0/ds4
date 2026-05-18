---
sprint: 018
title: V100 Descriptor-Bound Attention Projection Residual Norm Gate
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-018-INTENT.md
deferred: SPRINT-018-DEFERRED.md
verdict: SHIP
---

# SPRINT-018: V100 Descriptor-Bound Attention Projection Residual Norm Gate

## Overview

Sprint 017 created a scheduler-owned layer-state surface for descriptor-bound
router/FFN execution. Sprint 018 extends that state to attention/control
descriptors and adds a real-byte attention projection/residual/norm gate.

This is not full layer output. It is the next bounded bridge from existing
synthetic attention kernels to real descriptor-bound attention source bytes.

## Outcome Contract

- `SHIP`: layer state owns attention descriptors, the new CUDA smoke runs real
  layer-2 attention projection/residual/norm work from source bytes, compares
  against CPU references, and the appliance gate passes.
- `EXTEND`: attention descriptor state lands but the CUDA composition needs a
  documented follow-up.
- `STOP`: real attention descriptor dimensions or dtype semantics conflict with
  the existing source-F8/RMSNorm kernels.

## Non-Goals

- No full attention softmax/compressed-KV layer output.
- No selected-token real-model decode.
- No public serving, MTP, or throughput benchmark.
- No production persistent arena reuse.

## Implementation

### Phase 1: Attention State

**Files:**
- `ds4_v100_layer_state.h`
- `ds4_v100_layer_state.c`
- `tests/v100_layer_state_smoke.c`

**Tasks:**
- [x] Add attention bound matrices and control bindings to layer state.
- [x] Validate q/kv/output projection dimensions.
- [x] Add attention arena-span helper.
- [x] Extend the layer-state smoke to validate attention state.

### Phase 2: Descriptor-Bound Attention Smoke

**Files:**
- `tests/cuda_v100_descriptor_bound_attention_smoke.c`
- `Makefile`

**Tasks:**
- [x] Map the real source model and layer state.
- [x] Upload real attention FP8 source matrices to a V100 arena.
- [x] Run CPU/GPU attn RMSNorm, q_a, q_a_norm, q_b, kv_latent, output_a,
      output_b, residual add, and ffn_norm references.
- [x] Compare GPU outputs against CPU source-format references.

### Phase 3: Gate, Validation, Closeout

**Files:**
- `tools/ds4-v100-gate.sh`
- `docs/sprints/drafts/SPRINT-018-*.log`
- `docs/sprints/SPRINT-018-REPORT.md`
- `docs/sprints/SPRINT-018-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

**Tasks:**
- [x] Add `descriptor_bound_attention` to the appliance gate when
      `--pack-index` and model are supplied.
- [x] Run local validation.
- [x] Run cluster attention smoke and full appliance gate.
- [x] Archive logs.
- [x] Update sprint docs and vision.

## Definition Of Done

- Attention descriptors are part of `ds4_v100_layer_state`.
- `tests/cuda_v100_descriptor_bound_attention_smoke` passes on the V100 pod.
- Full appliance gate passes and reports `ready=false`.
- Sprint report, follow-ups, logs, and vision update are committed.

## Risks

- The smoke could be mistaken for full attention correctness. Keep the report
  explicit that softmax/compressed-KV visibility remains deferred.
- Large q/output projection surfaces can make CPU references slow. Keep the
  smoke single-token and bounded.
