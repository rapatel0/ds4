---
sprint: 010
title: V100 Single-Slot Decode Integration
status: completed
date: 2026-05-18
verdict: SHIP
---

# SPRINT-010 Report: V100 Single-Slot Decode Integration

## Verdict

`SHIP`

Sprint 010 shipped the bounded integration gate between Sprint 009's standalone
KV smoke and a real layer-owned V100 runtime. The CUDA context now exposes
deterministic per-layer subviews inside each stage-owned `kv_arena`, can write
diagnostic raw SWA, compressed attention KV, ratio-4 indexer KV, and split
KV/score state through that arena, and has a V100 compressor recurrence smoke
that compares ratio-128, ratio-4 attention, and ratio-4 indexer-shaped outputs
against a CPU reference.

Normal source-layout generation remains fail-closed.

## What Shipped

- Added `ds4_v100_layer_kv_view` metadata for raw SWA, compressed attention KV,
  indexer KV, attention KV state, attention score state, indexer KV state, and
  indexer score state.
- Derived deterministic per-layer views inside the existing per-stage KV arena
  without changing Sprint 009 arena totals or reserve accounting.
- Added CUDA context accessors for stage-owned layer KV views and arena
  readback diagnostics.
- Added `ds4_v100_cuda_context_prefill_kv_update_f16`, a bounded diagnostic
  writer that targets the real stage-owned `kv_arena` instead of standalone
  tensors.
- Extended `tests/cuda_v100_context_smoke.c` to verify stage-owned writes and
  readbacks for layer 2 ratio-4 and layer 3 ratio-128 at 1M context on 8 V100s.
- Added `tests/cuda_v100_compressor_bridge_smoke.c`, covering existing DS4
  compressor recurrence for ratio-128 attention, ratio-4 attention, and
  ratio-4 indexer-shaped state with CPU softmax-pool/RMSNorm references.
- Documented the V100 precision policy in Sprint 010: no broad BF16 runtime
  path and no broad FP32 GEMM fallback; BF16 source tensors convert to FP16
  runtime storage/scratch, while FP32 is reserved for small reductions,
  control, state, logits selection, debug, and oracle paths.

## Evidence

Local validation:

- `docs/sprints/drafts/SPRINT-010-PHASE1-LOCAL.log`
  - Builds and runs context/subview validation.
- `docs/sprints/drafts/SPRINT-010-PHASE2-LOCAL.log`
  - Builds model-less context targets and CUDA context smoke object.
- `docs/sprints/drafts/SPRINT-010-PHASE3-LOCAL.log`
  - Builds the compressor bridge smoke object and runs `git diff --check`.
- `docs/sprints/drafts/SPRINT-010-PHASE4-LOCAL.log`
  - Builds final local objects, runs `v100_context_smoke: ok`, confirms the
    1M context report returns `context_smoke_result OK`, and runs
    `git diff --check`.

Cluster validation:

- `docs/sprints/drafts/SPRINT-010-PHASE1-CLUSTER-CUDA-CONTEXT.log`
  - V100 context allocation and layer KV view smoke pass at 1M context.
- `docs/sprints/drafts/SPRINT-010-PHASE2-CLUSTER-CUDA-CONTEXT.log`
  - Stage-owned KV arena update passes for ratio-4 and ratio-128 on
    8x V100-SXM2-32GB.
- `docs/sprints/drafts/SPRINT-010-PHASE3-CLUSTER-COMPRESSOR-BRIDGE.log`
  - Real compressor recurrence smoke passes on `sm_70`.
- `docs/sprints/drafts/SPRINT-010-PHASE4-CLUSTER-GUARDS.log`
  - `tools/ds4-source-oracle-vector --guards-only` passes against
    `/models/DSv4-Flash-256e-fixed.gguf`.
- `docs/sprints/drafts/SPRINT-010-PHASE4-CLUSTER-CUDA-SMOKES.log`
  - Final integrated V100 context smoke and compressor bridge smoke pass.

## Deviations

- Sprint 010 did not unlock public serving, selected-token generation, or full
  43-layer logits.
- The stage-owned KV writer is still diagnostic and host-row driven. The
  compressor recurrence smoke uses real CUDA compressor kernels and device
  tensors, but it is not yet connected to the full source-model projection,
  attention, router, MoE, or output-head path.
- The bounded reference for compressor recurrence is a CPU helper in the CUDA
  smoke, not a new source-oracle intermediate dump.

## Handoff

Sprint 011 should not be public deployment yet. The next gate is a
logits-producing V100 source-layout slice: real source-format dense projections,
attention output, router/shared/routed expert path, and output-head/top-k for a
bounded single-slot prompt, compared against the source oracle before serving
is exposed.
