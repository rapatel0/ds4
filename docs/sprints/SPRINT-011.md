---
sprint: 011
title: V100 Source Projection And Attention Slice
status: planned
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-011-INTENT.md
deferred: SPRINT-011-DEFERRED.md
---

# SPRINT-011: V100 Source Projection And Attention Slice

## Overview

Sprint 010 proved stage-owned KV/state ownership and real compressor recurrence
on V100. Sprint 011 targets the next missing correctness surface: real
source-format projection math feeding a bounded attention/compressor slice.

This sprint deliberately stops before full logits, public serving, MTP,
multi-slot throughput, or tensor parallelism.

## Outcome Contract

- `SHIP`: a bounded V100 source F8_E4M3_B128 projection/tile path compares
  against a CPU source-format reference; at least one ratio-4 and one ratio-128
  attention/compressor slice consume device projection-equivalent tensors and
  compare against CPU/source references; Sprint 010 KV/context smokes and
  real-model source guards still pass.
- `EXTEND`: source F8 projection ships and one ratio class passes, with a
  diagnosed blocker for the other ratio class or attention/compressor slice.
- `STOP`: implementation requires broad FP32 GEMMs, persistent dequantized
  large source weights, public serving unlock, full logits, MTP, throughput
  scheduling, tensor parallelism, or unbounded host/SSD offload.

## Non-Goals

- No public CLI/server deployment.
- No normal source-layout generation unlock.
- No full selected-token generation or full 43-layer logits.
- No MTP/speculative decoding.
- No multi-slot batching, wavefront scheduling, or throughput benchmark.
- No tensor-parallel exceptions.
- No persistent dequantized copies of large source weights.

## Precision Policy

V100 does not execute BF16, FP8, or FP4 natively on tensor cores. Source F8 and
BF16 tensors must feed explicit decode, low-bit kernels, or FP16 HMMA tile paths
with FP32 accumulation where appropriate. FP32 remains acceptable for small
control/reduction/oracle paths and bounded diagnostic references, not for broad
model GEMMs.

## Planning Inputs

| File | Role |
|---|---|
| `docs/sprints/VISION.md` | Sprint sequence and deployment gate |
| `docs/architecture/DS4-V100-LAYOUT.md` | Runtime dtype, topology, and kernel-selection anchor |
| `docs/sprints/SPRINT-010-REPORT.md` | KV/state and compressor recurrence evidence |
| `docs/sprints/SPRINT-010-FOLLOWUPS.md` | Dense projection/logits blockers |
| `ds4_source_formats.[ch]` | CPU source F8/BF16 reference helpers |
| `ds4_gpu.h`, `ds4_cuda.cu` | CUDA projection, attention, compressor, and indexer APIs |
| `ds4_v100_context.[ch]`, `ds4_v100_context_cuda.cu` | Stage ownership and KV arena views |
| `tools/ds4-source-oracle-vector.c` | Source-layout guard/oracle entry point |

## Implementation

### Phase 1: Source F8 Dense Projection Diagnostic

**Files:**
- `ds4_gpu.h`
- `ds4_cuda.cu`
- `tests/cuda_source_dtypes_smoke.c` or new focused CUDA smoke
- `ds4_source_formats.[ch]`

**Tasks:**
- [x] Add a bounded V100 source F8_E4M3_B128 dense projection/tile diagnostic.
- [x] Compare against CPU source-format decode plus reference matmul.
- [x] Keep decoded data bounded to output rows or scratch tiles.
- [x] Reject unsupported shapes, invalid source views, and undersized tensors.

### Phase 2: BF16 Runtime Boundary Check

**Files:**
- `ds4_gpu.h`
- `ds4_cuda.cu`
- BF16/source-format tests

**Tasks:**
- [ ] Verify BF16 source tensors only enter V100 math through FP16/F32
      diagnostic conversion surfaces.
- [ ] Add or extend tests so no code path claims native BF16 V100 compute.
- [ ] Preserve the no-broad-FP32-GEMM policy in reports/tests.

### Phase 3: Projection-To-Attention/Compressor Slice

**Files:**
- `ds4_cuda.cu`
- `ds4_v100_context_cuda.cu`
- V100 CUDA smoke(s)

**Tasks:**
- [ ] Feed device projection-equivalent outputs into ratio-128 attention and
      compressor recurrence.
- [ ] Feed device projection-equivalent outputs into ratio-4 attention,
      compressor recurrence, and indexer-shaped recurrence.
- [ ] Write resulting rows through stage-owned KV arena views where applicable.
- [ ] Compare bounded outputs against CPU/source references.

### Phase 4: Guard Validation And Closeout

**Files:**
- `docs/sprints/drafts/SPRINT-011-*.log`
- `docs/sprints/SPRINT-011-REPORT.md`
- `docs/sprints/SPRINT-011-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

**Tasks:**
- [ ] Run local model-less validation and `git diff --check`.
- [ ] Run source-layout `--guards-only` on the real model.
- [ ] Run Sprint 010 regression smokes on V100 `sm_70`.
- [ ] Run new Sprint 011 CUDA smokes on V100 `sm_70`.
- [ ] Archive logs, write report/follow-ups, and update the vision.

## Definition Of Done

- Bounded source F8 projection/tile math passes V100 and CPU reference checks.
- Ratio-4 and ratio-128 bounded attention/compressor slices consume device
  projection-equivalent tensors and pass references.
- No broad FP32 GEMM or native BF16 V100 path is introduced.
- Source-layout guards remain fail-closed on the real model.
- Sprint 010 context/KV/compressor smokes still pass.
- Sprint report, follow-ups, logs, and vision update are committed.

## Risks

- A correct but simple F8 diagnostic projection may be too slow for production;
  performance kernels should remain a later optimization until correctness is
  established.
- The source oracle may need a new intermediate output mode for clean
  comparison against V100 slice outputs.
- Layer 2/3 attention surfaces may reveal missing shape/layout details in the
  current architecture sketch.
