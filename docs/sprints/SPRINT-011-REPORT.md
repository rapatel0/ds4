---
sprint: 011
title: V100 Source Projection And Attention Slice
status: completed
date: 2026-05-18
verdict: SHIP
---

# SPRINT-011 Report: V100 Source Projection And Attention Slice

## Verdict

`SHIP`

Sprint 011 shipped the next bounded correctness gate for the V100 appliance:
source F8_E4M3_B128 projection math can now execute from a device-resident
source arena, compare against CPU source-format references, feed ratio-128 and
ratio-4 attention/compressor slices, and write projection-derived rows into
stage-owned KV arena views without host-row staging.

Normal source-layout generation remains fail-closed.

## What Shipped

- Added `ds4_gpu_arena_f8_e4m3_b128_matmul_f32`, a bounded diagnostic CUDA
  source-F8 row-matmul path from immutable `ds4_gpu_arena` bytes into a device
  F32 output tensor.
- Tightened V100 BF16/F32 policy so BF16 declares an explicit
  `bf16_source_to_fp16_or_f32_boundary`, native BF16 V100 execution remains
  diagnostic-only, and F32 GEMM/matmul labels are rejected as broad model
  fallback paths.
- Added a stage-owned KV update API that accepts device-resident F32 attention
  and ratio-4 indexer rows, preserving device residency from projection output
  into the layer-owned KV arena.
- Added `tests/cuda_v100_projection_attention_smoke.c`, covering source-F8
  projection into Q/KV/score/indexer-shaped tensors, ratio-128 and ratio-4
  compressor recurrence, mixed attention, and stage-owned KV writes.
- Preserved Sprint 010 context, relay, KV, and compressor recurrence smokes on
  V100 `sm_70`.

## Evidence

Local validation:

- `docs/sprints/drafts/SPRINT-011-PHASE1-LOCAL.log`
  - Builds the source dtype smoke object after adding F8 matmul diagnostics.
- `docs/sprints/drafts/SPRINT-011-PHASE2-LOCAL.log`
  - Builds and runs V100 context policy checks; `v100_context_smoke: ok`.
- `docs/sprints/drafts/SPRINT-011-PHASE3-LOCAL.log`
  - Builds the projection/attention smoke object and runs `git diff --check`.
- `docs/sprints/drafts/SPRINT-011-PHASE4-LOCAL.log`
  - Builds final local model-less targets, runs `v100_context_smoke: ok`,
    runs `source_dtypes_smoke: ok`, and runs `git diff --check`.

Cluster validation:

- `docs/sprints/drafts/SPRINT-011-PHASE1-CLUSTER-F8-MATMUL.log`
  - `tests/cuda_source_dtypes_smoke` passes on V100 `sm_70`.
- `docs/sprints/drafts/SPRINT-011-PHASE2-CLUSTER-BF16-POLICY.log`
  - `tests/cuda_bf16_probe` and production V100 context smoke pass at 1M
    single-slot KV sizing.
- `docs/sprints/drafts/SPRINT-011-PHASE3-CLUSTER-PROJECTION-ATTENTION.log`
  - The new projection/attention smoke passes on V100 `sm_70`.
- `docs/sprints/drafts/SPRINT-011-PHASE4-CLUSTER-SOURCE-GUARDS.log`
  - `tools/ds4-source-oracle-vector --guards-only` passes against
    `/models/DSv4-Flash-256e-fixed.gguf`.
- `docs/sprints/drafts/SPRINT-011-PHASE4-CLUSTER-CUDA-SMOKES.log`
  - Final CUDA regression run passes source dtype, BF16 probe, production
    context/KV, compressor bridge, prefill KV, HC relay, and projection
    attention smokes.

## Deviations

- The F8 projection matmul is diagnostic and scalar-decoding, not the final
  production HMMA/low-bit kernel. It proves source-format device residency and
  correctness boundaries.
- Attention/compressor inputs are synthetic source-F8 matrices shaped like the
  runtime path, not full model tensor descriptors from the real pack.
- Sprint 011 still does not produce full layer HC output, router/expert
  outputs, output-head logits, selected-token decode, public serving, MTP, or
  throughput numbers.

## Handoff

Sprint 012 should be the logits-producing V100 gate. It needs to connect the
source projection slice to a bounded source-layout layer path that includes
attention output, residual/HC movement, router, shared expert, routed experts,
and output-head/top-k comparison against the guarded source oracle.
