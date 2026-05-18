---
sprint: 017
title: V100 Scheduler-Owned Layer State Gate
status: planned
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-017-INTENT.md
deferred: SPRINT-017-DEFERRED.md
---

# SPRINT-017: V100 Scheduler-Owned Layer State Gate

## Overview

Sprint 016 proved real descriptor-bound router-selected FFN compute, but the
binding, route view, and arena-span logic still lives in one CUDA smoke. Sprint
017 turns that into a reusable V100 layer-state surface that a future scheduler
can own and call.

This is still not serving. It is the runtime contract needed before attention,
residual/norm, real selected-token decode, and server exposure can be built
without another test-local rewrite.

## Outcome Contract

- `SHIP`: reusable layer-state API exists, validates layer-2 router/FFN
  descriptors, descriptor-bound FFN smoke uses it, and the V100 appliance gate
  passes with a new `layer_state` check.
- `EXTEND`: layer-state binding lands but descriptor-bound FFN still has local
  route logic that needs follow-up.
- `STOP`: the proposed state API cannot represent the real descriptor layout
  without duplicating context internals or breaking the Sprint 016 gate.

## Non-Goals

- No full attention output.
- No residual, RMSNorm, or HC transform execution beyond descriptor ownership.
- No selected-token real-model decode.
- No public serving, MTP, or throughput optimization.
- No production grouped expert kernel or persistent arena reuse.

## Implementation

### Phase 1: Layer-State API

**Files:**
- `ds4_v100_layer_state.h`
- `ds4_v100_layer_state.c`
- `Makefile`

**Tasks:**
- [ ] Add public bound-matrix/source-row-view helpers.
- [ ] Add `ds4_v100_layer_state` with layer id, stage id, owner GPU, layer
      class, router kind, FFN descriptor bindings, and KV view snapshot.
- [ ] Validate router/FFN dimensions and expected source dtypes/layouts.
- [ ] Add route-expert helper for selected MXFP4 gate/up/down views.
- [ ] Add FFN arena-span helper for production/smoke arena sizing.

### Phase 2: Local State Smoke

**Files:**
- `tests/v100_layer_state_smoke.c`
- `Makefile`

**Tasks:**
- [ ] Open the real pack index.
- [ ] Build layer-2 state.
- [ ] Validate hash router dimensions and selected expert route views.
- [ ] Validate FFN arena span stays inside the owning stage arena.

### Phase 3: Descriptor FFN Refactor

**Files:**
- `tests/cuda_v100_descriptor_bound_ffn_smoke.c`
- `Makefile`

**Tasks:**
- [ ] Replace local bound-matrix construction with layer-state helpers.
- [ ] Keep CPU and GPU router-selected FFN comparisons unchanged in behavior.
- [ ] Keep source-model byte reads and arena uploads source-faithful.

### Phase 4: Gate, Validation, Closeout

**Files:**
- `tools/ds4-v100-gate.sh`
- `docs/sprints/drafts/SPRINT-017-*.log`
- `docs/sprints/SPRINT-017-REPORT.md`
- `docs/sprints/SPRINT-017-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

**Tasks:**
- [ ] Add `layer_state` to the appliance gate when `--pack-index` is supplied.
- [ ] Run local validation.
- [ ] Run cluster state smoke, router FFN smoke, and full appliance gate.
- [ ] Archive logs.
- [ ] Update sprint docs and vision.

## Definition Of Done

- `ds4_v100_layer_state.*` is committed and used by at least one CUDA smoke.
- `tests/v100_layer_state_smoke` passes locally and on the V100 pod.
- `tests/cuda_v100_descriptor_bound_ffn_smoke` still passes on the V100 pod.
- Full appliance gate passes and reports `ready=false`.
- Sprint report, follow-ups, logs, and vision update are committed.

## Risks

- The layer-state API could become too narrow if it only mirrors FFN. Keep the
  state ready for attention by carrying layer/stage/KV metadata, but defer
  attention execution.
- The API could become too broad if it tries to own scratch allocation,
  residency upload, and scheduler dispatch in this sprint. Keep those as
  follow-ups.
