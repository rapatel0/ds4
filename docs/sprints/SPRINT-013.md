---
sprint: 013
title: V100 Source MXFP4 MoE And Selected-Token Gate
status: planned
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-013-INTENT.md
deferred: SPRINT-013-DEFERRED.md
---

# SPRINT-013: V100 Source MXFP4 MoE And Selected-Token Gate

## Overview

Sprint 012 shipped the bounded BF16 output-head/logits primitive and a V100
appliance gate. The gate still reports `ready=false` because routed MoE and
full selected-token execution are missing.

Sprint 013 targets that blocker with a bounded source-MXFP4 expert primitive
and a single-token MoE/logits fixture that produces a selected token against a
CPU source-format reference.

## Outcome Contract

- `SHIP`: source-MXFP4 arena matmul passes V100/CPU reference checks; a bounded
  router + MXFP4 routed expert + BF16 output-head selected-token smoke passes;
  the appliance gate runs it and still fails closed for real serving readiness.
- `EXTEND`: MXFP4 matmul passes and is committed, but bounded MoE composition
  exposes a diagnosed blocker.
- `STOP`: implementation requires full real-model scheduler unlock, public
  serving, broad FP32 production GEMMs, persistent dequantized expert copies,
  MTP, throughput tuning, or tensor-parallel integration.

## Non-Goals

- No normal source-layout generation/server unlock.
- No full 43-layer real-model selected-token run.
- No final production TurboMind/tc-grid expert kernel selection.
- No throughput benchmark claim.
- No MTP/speculative decoding.
- No persistent dequantized full expert weights.

## Precision Policy

MXFP4 on V100 is not native Blackwell FP4 tensor-core execution. Sprint 013's
MXFP4 primitive is a bounded source-format decode/reduction path that preserves
resident packed bytes and correctness semantics. It is a correctness gate and
integration substrate, not the final performance kernel.

## Implementation

### Phase 1: Source MXFP4 Arena Matmul Primitive

**Files:**
- `ds4_gpu.h`
- `ds4_cuda.cu`
- `ds4_gpu_arena_stub.c`
- source dtype tests

**Tasks:**
- [ ] Add `ds4_gpu_arena_mxfp4_matmul_f32` using `ds4_gpu_source_row_view`.
- [ ] Validate MXFP4 rows, cols, row stride, byte ranges, input tensor, and
      output tensor.
- [ ] Decode low-half/high-half nibble ordering to match
      `ds4_src_mxfp4_row_dot`.
- [ ] Add focused CUDA coverage and invalid-view checks.

### Phase 2: Bounded Router/MoE Selected-Token Smoke

**Files:**
- `tests/cuda_v100_mxfp4_moe_smoke.c`
- `Makefile`

**Tasks:**
- [ ] Build a deterministic single-token fixture with 256 router logits and
      six selected routes.
- [ ] Run router selection on device and compare selected expert ids/weights
      against CPU reference.
- [ ] Run source-MXFP4 gate/up/down expert matmuls for selected routes.
- [ ] Run SwiGLU and route accumulation on device.
- [ ] Run BF16 bounded output-head logits and selected-token comparison.

### Phase 3: Appliance Gate Extension

**Files:**
- `tools/ds4-v100-gate.sh`
- sprint logs

**Tasks:**
- [ ] Add the new MoE selected-token smoke to the gate target list.
- [ ] Preserve `ready=false` until real layer descriptors and full selected
      token are wired.
- [ ] Run the gate on the V100 pod and archive logs.

### Phase 4: Closeout

**Files:**
- `docs/sprints/drafts/SPRINT-013-*.log`
- `docs/sprints/SPRINT-013-REPORT.md`
- `docs/sprints/SPRINT-013-FOLLOWUPS.md` if needed
- `docs/sprints/VISION.md`

**Tasks:**
- [ ] Run local validation and `git diff --check`.
- [ ] Run V100 `sm_70` validation.
- [ ] Archive cluster logs.
- [ ] Write report/follow-ups and update the vision.

## Definition Of Done

- Source-MXFP4 arena matmul exists and passes CPU reference checks on V100.
- Bounded MoE selected-token smoke passes on V100.
- Appliance gate includes and passes the new smoke.
- Source-layout real-model guards remain fail-closed.
- No broad FP32 production GEMM, public serving unlock, MTP, or persistent
  dequantized expert copy is introduced.
- Report, logs, follow-ups/deferred notes, and vision update are committed.

## Risks

- The MXFP4 primitive will be diagnostic and may be too slow for production.
- Router/MoE composition may expose assumptions in existing router APIs, such
  as fixed 256 experts and six selected routes.
- Passing a synthetic bounded selected-token fixture still does not prove full
  model quality; it proves the MoE/logits kernel surfaces are composable.
