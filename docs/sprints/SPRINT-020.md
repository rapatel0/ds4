---
sprint: 020
title: V100 Compressor/Indexer And HC Scheduler Bridge
status: planned
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-020-INTENT.md
deferred: SPRINT-020-DEFERRED.md
verdict: pending
---

# SPRINT-020: V100 Compressor/Indexer And HC Scheduler Bridge

## Overview

Sprint 019 shipped a real hidden-vector layer executor. Sprint 020 makes that
slice more faithful to DS4 by moving compressed-row production and HC state
handling into the scheduler-owned path.

The sprint still targets one representative ratio-4 layer first. The point is
to replace test-provided compressed KV with real descriptor-bound
compressor/indexer work and to prove `[4 x 4096]` HC state can flow through the
layer surface.

## Outcome Contract

- `SHIP`: compressor/indexer descriptors are bound, compressed-row generation
  and indexer visibility are executor-owned for layer 2, an HC-state layer
  smoke passes on V100, and the full appliance gate passes.
- `EXTEND`: descriptors and one of compressor/indexer or HC scheduling ship,
  with a concrete hardware-backed blocker for the missing half.
- `STOP`: existing CUDA compressor/indexer or HC APIs cannot represent the DS4
  reference semantics without a runtime-layout redesign.

## Non-Goals

- No public server.
- No MTP.
- No throughput benchmark or slot wavefront.
- No tensor-parallel implementation.
- No full 43-layer selected-token claim.
- No persistent dequantized weights.

## Parallel Workstreams

| Lane | Responsibility | Write Scope | Validation |
|---|---|---|---|
| A: descriptor state | Bind real compressor/indexer tensors and validate shapes. | `ds4_v100_layer_state.*`, layer-state smoke | local and V100 layer-state checks |
| B: compressor/indexer execution | Generate compressed attention/indexer rows and top-k or masks inside executor. | `ds4_v100_layer_execute.*`, CUDA call glue | CPU/GPU bounded layer-2 comparison |
| C: HC scheduler wrapper | Add `[4 x 4096]` HC pre/post entrypoint around the hidden-vector executor. | `ds4_v100_layer_execute.*`, HC smoke | V100 HC-state output comparison |
| D: gate/evidence | Build Sprint 020 smoke and add it to the appliance gate. | `tests/*`, `Makefile`, `tools/ds4-v100-gate.sh`, docs | one-card smoke and full gate logs |

## Implementation

### Phase 1: Compressor And Indexer State

**Files:**
- `ds4_v100_layer_state.h`
- `ds4_v100_layer_state.c`
- `tests/v100_layer_state_smoke.c`

**Tasks:**
- [ ] Bind `attn_compressor_kv`, `attn_compressor_gate`,
      `attn_compressor_ape`, and `attn_compressor_norm`.
- [ ] Bind ratio-4 `indexer_attn_q_b`, `indexer_proj`,
      `indexer_compressor_kv`, `indexer_compressor_gate`,
      `indexer_compressor_ape`, and `indexer_compressor_norm`.
- [ ] Validate ratio-4 dimensions from the source layout.
- [ ] Add arena-span coverage for compressor/indexer matrices.

### Phase 2: Executor-Owned Compressed Rows

**Files:**
- `ds4_v100_layer_execute.h`
- `ds4_v100_layer_execute.c`
- `tests/cuda_v100_integrated_layer_smoke.c`

**Tasks:**
- [ ] Add executor config for raw KV, compressed KV, indexer KV, and compressor
      state tensors.
- [ ] Generate current attention compressor KV/score projections from
      `attn_norm`.
- [ ] Update compressor state and write emitted compressed attention rows.
- [ ] Generate/update ratio-4 indexer rows.
- [ ] Run indexer scoring/top-k or build an equivalent compressed-row mask.
- [ ] Feed executor-owned compressed visibility into semantic attention.

### Phase 3: HC State Wrapper

**Files:**
- `ds4_v100_layer_execute.h`
- `ds4_v100_layer_execute.c`
- `tests/cuda_v100_hc_layer_smoke.c` or extended integrated smoke

**Tasks:**
- [ ] Add an HC entrypoint that accepts `[4 x 4096]` input state.
- [ ] Use HC attention controls to produce the attention hidden vector and
      retain split state for post expansion.
- [ ] Use HC FFN controls around the FFN hidden-vector body.
- [ ] Compare HC output against CPU references for a bounded layer-2 fixture.

### Phase 4: Gate And Hardware Evidence

**Files:**
- `Makefile`
- `tools/ds4-v100-gate.sh`
- `docs/sprints/drafts/SPRINT-020-*.log`
- `docs/sprints/SPRINT-020-REPORT.md`
- `docs/sprints/SPRINT-020-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

**Tasks:**
- [ ] Add Sprint 020 smoke target(s) to the build.
- [ ] Add Sprint 020 smoke target(s) to the appliance gate.
- [ ] Run one-card V100 iteration with `CUDA_VISIBLE_DEVICES=0`.
- [ ] Run full V100 gate.
- [ ] Update readiness reasons honestly.
- [ ] Commit report, follow-ups, logs, and vision update.

## Definition Of Done

- Compressor and indexer descriptors are in `ds4_v100_layer_state`.
- The executor can produce compressed attention/indexer rows from real source
  bytes for layer 2.
- The executor no longer needs test-provided compressed KV for the Sprint 020
  layer-2 path.
- An HC-state layer smoke passes on V100.
- Full appliance gate passes and remains `ready=false` for selected-token,
  serving, MTP, and throughput.

## Risks

- Compressor recurrence and indexer top-k semantics may expose a mismatch
  between existing CUDA helpers and the CPU reference path.
- HC pre/post kernels may not cover the exact one-token path without wrapper
  glue.
- Scratch pressure can rise quickly if compressor/indexer and HC tensors are
  allocated naively.
- A layer-2 ratio-4 proof still does not cover SWA-only layers 0-1 or
  ratio-128 odd layers.

