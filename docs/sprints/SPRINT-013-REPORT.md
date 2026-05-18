---
sprint: 013
title: V100 Source MXFP4 MoE And Selected-Token Gate
status: completed
date: 2026-05-18
verdict: SHIP
---

# SPRINT-013 Report: V100 Source MXFP4 MoE And Selected-Token Gate

## Verdict

`SHIP`

Sprint 013 shipped the first source-MXFP4 expert execution surface and a
bounded V100 MoE/logits selected-token smoke. The new path reads MXFP4 routed
expert rows from a resident arena, decodes them in-kernel, composes router
selection, gate/up/down expert matmuls, SwiGLU, route accumulation, BF16
output-head logits, and selected-token comparison against CPU source-format
references.

The appliance is still not ready for real serving. The gate now proves a
bounded synthetic MoE selected-token path, but real pack-index layer descriptors
and the full layer scheduler are not wired.

## What Shipped

- Added `ds4_gpu_arena_mxfp4_matmul_f32`, a bounded source-MXFP4 matrix-vector
  diagnostic path from resident arena bytes to device F32 outputs.
- Added V100 CUDA decode/reduction for MXFP4 using GGML low-half/high-half
  nibble ordering and E8M0 block scales.
- Added `tests/cuda_v100_mxfp4_moe_smoke.c`, covering:
  - 256-way router selection and route weights;
  - six selected MXFP4 routed gate/up experts;
  - SwiGLU and weighted route accumulation;
  - MXFP4 routed down projection;
  - BF16 bounded output-head logits and selected-token comparison.
- Extended `tools/ds4-v100-gate.sh` to build/run the new MoE selected-token
  smoke.

## Evidence

Local validation:

- `make ds4_gpu_arena_stub.o tests/cuda_v100_mxfp4_moe_smoke.o tests/cuda_v100_bounded_logits_smoke.o`
- `bash -n tools/ds4-v100-gate.sh`
- `git diff --check`

Cluster validation:

- `docs/sprints/drafts/SPRINT-013-GATE-CLUSTER/gate-summary.log`
  - Source guards passed against `/models/DSv4-Flash-256e-fixed.gguf`.
  - Source dtype, BF16 probe, 1M single-slot context/KV, compressor bridge,
    prefill KV, HC relay, projection/attention, bounded logits, and MXFP4 MoE
    smokes passed on the 8x V100 pod.
  - Gate summary: `PASS`, `failures=0`, `ready=false`.
- `docs/sprints/drafts/SPRINT-013-GATE-CLUSTER/mxfp4_moe.log`
  - `cuda_v100_mxfp4_moe_smoke: ok`.

## Deviations

- The MXFP4 primitive is diagnostic and scalar-reduction oriented, not the
  final grouped production expert kernel.
- The selected-token smoke is synthetic and bounded. It does not yet consume
  real pack-index layer descriptors.
- Shared-expert F8 composition is not included in the bounded MoE smoke yet.

## Handoff

Sprint 014 should wire the bounded source primitives into real pack-index
layer descriptors and produce a real single-layer or short selected-token path.
The next readiness transition should remove the gap between synthetic MoE
composition and real model layer scheduling.
