---
sprint: 019
title: V100 Integrated Single-Layer Runtime Slice
status: planned
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-019-INTENT.md
deferred: SPRINT-019-DEFERRED.md
verdict: pending
---

# SPRINT-019: V100 Integrated Single-Layer Runtime Slice

## Overview

Sprint 018 proved real descriptor-bound attention projection, residual, and
norm surfaces. Sprint 019 turns the isolated attention and FFN smokes into a
callable single-layer runtime slice for a representative ratio-4 layer.

The sprint uses `docs/architecture/DS4-V100-LAYOUT.md` as the layout and
topology contract: layer-sharded baseline, source FP8 dense weights, source
MXFP4 routed experts, F16 KV first, FP16 activations on V100, and no persistent
dequantized weight copies.

## Outcome Contract

- `SHIP`: a reusable layer execution surface produces a coherent next hidden
  state for layer 2 by composing semantic attention and real router-selected
  FFN, and the integrated smoke passes on V100 hardware.
- `EXTEND`: the execution surface lands and one half of the layer is semantic,
  but the other half requires a documented follow-up with a blocking technical
  reason.
- `STOP`: raw/compressed KV visibility, sink handling, or output projection
  semantics conflict with the current kernels in a way that prevents progress
  without redesigning the runtime layout.

## Non-Goals

- No public server exposure.
- No MTP speculative decoding.
- No multi-slot throughput optimization.
- No tensor-parallel implementation.
- No persistent dequantized weight buffers.
- No claim that a one-layer slice proves full 43-layer selected-token decode.

## Parallel Workstreams

| Lane | Responsibility | Write Scope | Validation |
|---|---|---|---|
| A: semantic attention | Turn descriptor-bound q/kv surfaces into raw plus compressed KV softmax output, including sinks and ratio-4 row visibility. | `ds4_cuda.cu`, attention test helpers | GPU output compared to CPU/source-format bounded reference |
| B: layer executor | Add a reusable scheduler-owned layer execution API around `ds4_v100_layer_state`. | `ds4_v100_layer_execute.h`, `ds4_v100_layer_execute.c`, minor state additions | local build and integrated smoke calls the API |
| C: integrated smoke/gate | Compose attention, residual, FFN pre-norm, router-selected FFN, and final residual into next hidden. | `tests/cuda_v100_integrated_layer_smoke.c`, `Makefile`, `tools/ds4-v100-gate.sh` | one-GPU V100 smoke and full gate |
| D: direct hardware evidence | Run fast direct-host validation on one or two cards before the full 8-GPU gate. | logs under `docs/sprints/drafts/` | captured command output and gate summary |

The lanes are designed so subagents can work independently: A owns kernel
semantics, B owns API boundaries, C owns test/gate integration, and D owns
hardware execution/logs.

## Implementation

### Phase 1: Attention Semantics

**Files:**
- `ds4_cuda.cu`
- `ds4_gpu.h`
- `tests/cuda_v100_projection_attention_smoke.c`
- `tests/cuda_v100_descriptor_bound_attention_smoke.c`

**Tasks:**
- [ ] Extract or share the CPU bounded attention reference for raw plus
      compressed rows.
- [ ] Drive the existing GPU attention/compressor kernels from real layer-state
      descriptors instead of synthetic rows.
- [ ] Include attention sink contribution and ratio-4 compressed-row visibility
      in the bounded comparison.
- [ ] Emit a semantic attention vector suitable for the output A/B projection,
      not the Sprint 018 proxy.

### Phase 2: Layer Execution Surface

**Files:**
- `ds4_v100_layer_execute.h`
- `ds4_v100_layer_execute.c`
- `ds4_v100_layer_state.h`
- `ds4_v100_layer_state.c`

**Tasks:**
- [ ] Add a small execution config for layer id, token row, context rows,
      active slots, and scratch ownership.
- [ ] Add a callable single-layer decode/prefill-slice function that consumes
      `ds4_v100_layer_state`.
- [ ] Keep API boundaries explicit: source descriptors in state, KV views in
      config, scratch in caller-owned arenas, no hidden persistent dequant.
- [ ] Return per-phase timing/counter fields for later throughput work.

### Phase 3: Integrated Layer Smoke

**Files:**
- `tests/cuda_v100_integrated_layer_smoke.c`
- `Makefile`

**Tasks:**
- [ ] Map the real model and pack index.
- [ ] Upload only the required layer-2 source bytes into bounded V100 arenas.
- [ ] Execute attention, residual, FFN pre-norm, real router selection,
      routed/shared FFN, and final residual through the new layer executor.
- [ ] Compare the bounded next-hidden output to CPU/source-format references.
- [ ] Print memory, layer, token, context, selected experts, arena span, and
      max-difference diagnostics.

### Phase 4: Gate And Hardware Validation

**Files:**
- `tools/ds4-v100-gate.sh`
- `docs/sprints/drafts/SPRINT-019-*.log`
- `docs/sprints/SPRINT-019-REPORT.md`
- `docs/sprints/SPRINT-019-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

**Tasks:**
- [ ] Add the integrated layer smoke to the appliance gate when model and pack
      index are supplied.
- [ ] Run one-card direct-host validation using `CUDA_VISIBLE_DEVICES=0`.
- [ ] Run the full appliance gate on the V100 host.
- [ ] Update readiness reasons honestly after the new evidence.
- [ ] Archive logs and update sprint report, follow-ups, and vision.

## Direct V100 Validation Plan

Use direct SSH for fast iteration:

```sh
ssh ubuntu@192.168.102.5
cd /path/to/ds4
CUDA_VISIBLE_DEVICES=0 ./tests/cuda_v100_integrated_layer_smoke \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --layer 2 \
  --router-token 16
```

Use one card for kernel correctness. Use two visible cards only if boundary or
stage behavior is being tested. Save the full 8-GPU run for the final gate.

## Definition Of Done

- `ds4_v100_layer_execute.*` exists and is used by an integrated CUDA smoke.
- The integrated layer smoke validates a real descriptor-bound ratio-4 layer on
  V100 hardware.
- The smoke produces a next-hidden vector after attention plus FFN, not only an
  intermediate projection.
- Local build checks pass.
- Full appliance gate includes the integrated smoke and passes with an honest
  readiness summary.
- Sprint report, follow-ups, logs, and vision update are committed.

## Risks

- The DS4 attention reference may require more exact compressed-row/indexer
  behavior than the current bounded kernels expose.
- A one-layer slice can still hide issues in layer 0/1 SWA-only and later
  ratio-128 layers; those must remain future gate targets.
- Scratch pressure may be higher than previous smokes because attention and FFN
  are composed in one process.
- If the integrated smoke uses bounded partial arenas, production resident
  arena reuse remains a follow-up.

## Open Questions

1. Should `full_layer_scheduler` remain in the readiness missing list until all
   43 layers can be scheduled, or can it become `full_43_layer_scheduler` after
   this one-layer executor lands?
2. Is the first integrated path decode-only, or should it also cover a tiny
   prefill microbatch while the attention reference is already loaded?
3. Is a checked-in direct-host helper script worth adding, or are captured
   commands in the sprint report enough?

