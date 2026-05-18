---
sprint: 012
title: V100 Appliance Gate And Bounded Output-Head Logits
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-012-INTENT.md
deferred: SPRINT-012-DEFERRED.md
verdict: SHIP
---

# SPRINT-012: V100 Appliance Gate And Bounded Output-Head Logits

## Overview

Sprint 011 proved source-F8 projection diagnostics and projection-fed
attention/compressor slices on V100. Sprint 012 adds the next concrete
logits-facing primitive: a bounded source-BF16 output-head matmul from resident
arena bytes, plus an appliance readiness gate that can be run on the V100 pod.

This sprint is intentionally not the full public serving unlock. It should
produce real logits/top-k evidence for the output-head surface and make
readiness status machine-visible, while continuing to fail closed for normal
source-layout generation.

## Outcome Contract

- `SHIP`: a bounded CUDA source-BF16 output-head/logits primitive passes V100
  and CPU reference checks; a V100 gate command runs real-model source guards,
  existing CUDA smokes, the new logits smoke, and emits fail-closed readiness
  status; Sprint 011 regressions still pass.
- `EXTEND`: the BF16 output-head/logits primitive passes but the readiness gate
  is incomplete, or the gate passes but one non-blocking regression needs a
  documented fix.
- `STOP`: implementation requires public source-layout generation unlock,
  full 43-layer MoE execution, broad FP32 production GEMMs, persistent
  dequantized full-model copies, MTP, throughput scheduling, or tensor
  parallelism.

## Non-Goals

- No normal source-layout generation/server unlock.
- No full 43-layer selected-token decode.
- No production MXFP4 routed expert kernel selection.
- No MTP/speculative decoding.
- No multi-slot throughput scheduling or benchmark claim.
- No vocab-parallel output head; this sprint uses a bounded local output-head
  fixture.
- No persistent dequantized copies of large source weights.

## Precision Policy

V100 does not execute BF16, FP8, or FP4 natively on tensor cores. The Sprint 012
BF16 output-head primitive is a bounded diagnostic conversion/reduction surface:
source BF16 bytes are expanded explicitly and accumulated for correctness
evidence. It is not a claim that V100 has native BF16 compute and it is not the
final throughput kernel.

Production dense math remains expected to move toward FP16 HMMA with FP32
accumulation, or toward explicitly validated low-bit/integer kernels where
quality permits.

## Planning Inputs

| File | Role |
|---|---|
| `docs/sprints/VISION.md` | Sprint sequence and deployment gate |
| `docs/architecture/DS4-V100-LAYOUT.md` | Source dtype, runtime dtype, sharding, and topology anchor |
| `docs/sprints/SPRINT-011-REPORT.md` | Projection/attention evidence and handoff |
| `docs/sprints/SPRINT-011-FOLLOWUPS.md` | Critical Sprint 012 blockers |
| `ds4_gpu.h`, `ds4_cuda.cu` | CUDA arena/tensor APIs and source dtype kernels |
| `ds4_source_formats.[ch]` | CPU source-format references |
| `tests/cuda_v100_projection_attention_smoke.c` | Current V100 projection/attention regression |
| `tools/ds4-source-oracle-vector.c` | Real-model guard/oracle entry point |

## Implementation

### Phase 1: Source BF16 Output-Head Matmul Primitive

**Files:**
- `ds4_gpu.h`
- `ds4_cuda.cu`
- `ds4_gpu_arena_stub.c`
- CUDA/source dtype tests

**Tasks:**
- [x] Add a bounded `ds4_gpu_arena_bf16_matmul_f32` API that reads a
      `ds4_gpu_bf16_matrix_view` from a resident arena and multiplies it by a
      device F32 hidden vector.
- [x] Validate rows, cols, row stride, byte ranges, tensor dtype, and output
      capacity.
- [x] Accumulate to F32 for diagnostic correctness.
- [x] Keep the API explicitly labeled as BF16 source conversion/diagnostic, not
      native BF16 V100 compute.
- [x] Add host-stub behavior or a fail-closed stub consistent with existing
      arena APIs.

### Phase 2: Bounded Logits And Top-K Smoke

**Files:**
- `tests/cuda_v100_bounded_logits_smoke.c` or equivalent focused test
- `Makefile`
- `ds4_source_formats.[ch]` if a reusable CPU helper is needed

**Tasks:**
- [x] Build a deterministic synthetic source-BF16 output-head matrix and hidden
      vector.
- [x] Run the V100 BF16 matmul primitive to produce bounded logits.
- [x] Compare logits against a CPU BF16 reference within a tight deterministic
      tolerance.
- [x] Compare top-1 or top-k selection against the CPU reference.
- [x] Add invalid-view/shape checks for undersized source and output buffers.

### Phase 3: Appliance Readiness Gate

**Files:**
- `tools/ds4-v100-gate.sh` or `tools/ds4-v100-gate.c`
- `Makefile` if needed
- sprint draft logs

**Tasks:**
- [x] Add a runnable gate that can execute on the V100 pod.
- [x] Run real-model source-layout guards.
- [x] Run current V100 regression smokes from Sprints 009-011.
- [x] Run the new bounded logits smoke.
- [x] Emit a concise report that distinguishes test failure from fail-closed
      missing readiness gates.
- [x] Keep final readiness false until MoE/full selected-token serving is
      validated.

### Phase 4: Validation And Closeout

**Files:**
- `docs/sprints/drafts/SPRINT-012-*.log`
- `docs/sprints/SPRINT-012-REPORT.md`
- `docs/sprints/SPRINT-012-FOLLOWUPS.md` if needed
- `docs/sprints/VISION.md`

**Tasks:**
- [x] Run local build/test checks and `git diff --check`.
- [x] Build and run V100 `sm_70` cluster validation.
- [x] Archive logs under `docs/sprints/drafts/`.
- [x] Write the sprint report and any follow-ups discovered during execution.
- [x] Update the vision with the Sprint 012 outcome.

## Definition Of Done

- `ds4_gpu_arena_bf16_matmul_f32` or equivalent source-BF16 output-head
  primitive exists and is tested.
- A bounded V100 logits/top-k smoke passes against a CPU BF16 reference.
- Existing source dtype, BF16 policy, context/KV, compressor, prefill, relay,
  and projection/attention smokes still pass on V100.
- Real-model source-layout guards still fail closed for normal generation.
- A V100 appliance gate command is committed and emits readiness status.
- No broad FP32 production GEMM, native BF16 V100 claim, public serving unlock,
  or persistent dequantized full-model copy is introduced.
- Sprint report, logs, deferred/follow-up notes, and vision update are
  committed.

## Risks

- The bounded BF16 output-head matmul will not be the final fast output-head
  implementation. It is a correctness gate and integration substrate.
- A synthetic bounded logits fixture does not prove full model quality. It
  proves the output-head source dtype and top-k surface before MoE integration.
- If the gate is too permissive, users may mistake partial readiness for a
  deployable runtime. It must fail closed until a full selected-token path is
  validated.
