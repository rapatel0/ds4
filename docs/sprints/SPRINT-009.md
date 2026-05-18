---
sprint: 009
title: V100 Prefill And Compressed KV Execution
status: planned
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-009-INTENT.md
deferred: SPRINT-009-DEFERRED.md
---

# SPRINT-009: V100 Prefill And Compressed KV Execution

## Overview

Sprint 008 shipped the contracts needed to start real V100 runtime work:
official-vector oracle automation, fail-closed source-layout guards, exact F16
KV admission by layer/stage/context/slot, source dtype parity hardening, and a
bounded CUDA F8_E4M3_B128 source-format anchor.

Sprint 009 consumes those contracts to build the first bounded V100 prefill/KV
execution surface. The sprint should not try to expose normal serving or full
decode. Its job is to allocate and update DS4 F16 KV state on the owning V100
stage, prove raw SWA plus compressed KV/indexer state updates for representative
ratio classes, and compare the bounded device slice against CPU/source
references.

Normal source-layout generation remains fail-closed.

## Outcome Contract

- `SHIP`: a diagnostic V100 prefill/KV path allocates F16 KV from the derived
  stage budget, executes bounded raw SWA plus compressed KV updates for at least
  one ratio-4 layer and one ratio-128 layer on `sm_70`, validates outputs
  against CPU/source references, preserves all source-layout guards, and
  archives local and cluster evidence.
- `EXTEND`: the context/KV allocation and one ratio class ship, but the second
  ratio class or indexer state update is blocked by a diagnosed kernel mismatch
  or source-layout gap. Guards must remain intact.
- `STOP`: implementation requires normal source-layout serving unlock,
  persistent dequantized large source tensors, host/SSD/offload as the success
  path, MTP/speculative decode, multi-slot throughput scheduling, tensor
  parallelism, or broad production FP8/MXFP4 GEMMs.

## Non-Goals

- No normal source-layout generation unlock.
- No full 43-layer logits-producing decode.
- No public server/API deployment.
- No MTP or speculative decoding.
- No throughput benchmarking, slot batching, or wavefront scheduling.
- No tensor-parallel exceptions.
- No persistent dequantized F16/F32 copies of large FP8/MXFP4 weights.
- No F8 KV baseline; F16 KV remains the correctness baseline.

## Planning Inputs

| File | Role |
|---|---|
| `docs/sprints/VISION.md` | North Star and Sprint 009 sequencing |
| `docs/architecture/DS4-V100-LAYOUT.md` | V100 topology, dtype, memory, and scheduling contract |
| `docs/sprints/SPRINT-008-REPORT.md` | Oracle, guard, KV admission, and F8 CUDA anchor evidence |
| `docs/sprints/SPRINT-008-DEFERRED.md` | Prefill/KV and production-kernel deferred scope |
| `docs/sprints/SPRINT-008-FOLLOWUPS.md` | Sprint 009 validation and source-format follow-ups |
| `ds4_v100_context.[ch]` | Stage map, descriptor policy, and F16 KV admission |
| `ds4_v100_context_cuda.cu` | CUDA context/resource wrapper |
| `ds4_gpu.h`, `ds4_cuda.cu` | CUDA arena, source-format, attention, compressor, and indexer primitives |
| `ds4_source_formats.[ch]` | CPU source-format reference helpers |
| `tools/ds4-source-oracle-vector.c` | Source oracle and guard validation |

## Use Cases

1. **KV admission to allocation**: a developer can request `ctx` and `slots` and
   see V100 F16 KV buffers allocated according to the Sprint 008 stage budget.
2. **Ratio-class KV update proof**: a CUDA smoke can update raw SWA and
   compressed KV for representative ratio-4 and ratio-128 layers.
3. **Indexer state proof**: a ratio-4 CUDA smoke can allocate and update the
   indexer KV/state surface without touching full decode.
4. **Source-format bridge**: F8 source packed rows can feed the bounded prefill
   slice without persistent dequantized copies.
5. **Guard continuity**: source-layout oracle guards remain executable and
   normal source serving remains blocked.

## Architecture

The sprint adds a diagnostic V100 prefill/KV harness beside the existing
context and CUDA arena code:

```text
ds4_v100_context_open(ctx, slots)
    |
    +--> derived per-stage KV budgets
    |
    v
stage-local KV arena plan
    |
    +--> raw SWA F16 rows
    +--> compressed attention KV F16 rows
    +--> ratio-4 indexer KV/state
    |
    v
bounded CUDA prefill/KV smoke
    |
    +--> source-format input tile
    +--> raw append/update
    +--> compressor/indexer update
    +--> CPU/reference comparison
```

The path is diagnostic-only. It should use explicit test harnesses and reports,
not normal `ds4` generation flags.

## Implementation

### Phase 1: KV Arena Plan And Allocation

**Files:**
- `ds4_v100_context.h`
- `ds4_v100_context.c`
- `ds4_v100_context_cuda.cu`
- `tests/v100_context_smoke.c`
- `tests/cuda_v100_context_smoke.c`

**Tasks:**
- [x] Add a stage-local KV arena descriptor derived from
      `ds4_v100_kv_budget_for_layer`.
- [x] Allocate raw SWA, compressed attention KV, indexer KV, and compression
      state buffers for explicit `ctx` and `slots`.
- [x] Fail closed if allocation would exceed the stage memory reserve.
- [x] Print stable allocation offsets/sizes in the context report.

### Phase 2: Bounded Raw SWA And Compressed KV Update

**Files:**
- `ds4_gpu.h`
- `ds4_cuda.cu`
- `tests/cuda_v100_prefill_kv_smoke.c`
- `Makefile`

**Tasks:**
- [x] Add a diagnostic CUDA API that appends a bounded F16 KV row to raw SWA.
- [x] Add a bounded compressed KV update for a ratio-128 layer.
- [x] Add a bounded compressed KV plus indexer KV/state update for a ratio-4
      layer.
- [x] Compare the resulting rows/state against CPU or source-helper references.
- [x] Reject invalid layer class, row bounds, slot bounds, and undersized
      buffers.

### Phase 3: Source-Format Prefill Input Bridge

**Files:**
- `ds4_gpu.h`
- `ds4_cuda.cu`
- `ds4_source_formats.[ch]`
- `tests/cuda_v100_prefill_kv_smoke.c`

**Tasks:**
- [x] Feed at least one F8_E4M3_B128 source-format input tile through the
      Sprint 008 row-decode pattern.
- [x] Keep dequantization bounded to output/scratch tiles only.
- [x] Document any precision or tolerance used for CUDA-vs-CPU comparisons.

**Precision note:** `tests/cuda_v100_prefill_kv_smoke.c` compares F8 source
decode bit-exactly against `ds4_source_formats`, verifies F16 KV rows by exact
half bits after host-side F32-to-F16 rounding, and uses `1e-5` absolute
tolerance for deterministic F32 diagnostic state values.

### Phase 4: Guard And Cluster Validation

**Files:**
- `docs/sprints/drafts/SPRINT-009-*.log`
- `docs/sprints/SPRINT-009-REPORT.md`
- `docs/sprints/SPRINT-009-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

**Tasks:**
- [ ] Run model-less local build/tests.
- [ ] Run `tools/ds4-source-oracle-vector --guards-only` against
      `/models/DSv4-Flash-256e-fixed.gguf` on the cluster.
- [ ] Run the new CUDA prefill/KV smoke on V100 `sm_70`.
- [ ] Run V100 context admission for at least 256K and 1M single-slot tiers.
- [ ] Archive command logs under `docs/sprints/drafts/`.
- [ ] Write the report with verdict, evidence, deviations, and Sprint 010
      handoff.
- [ ] Update `VISION.md` after the sprint verdict.

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `ds4_v100_context.[ch]` | Modify | Expose KV arena plan/offsets from derived stage budgets |
| `ds4_v100_context_cuda.cu` | Modify | Allocate/free stage-local KV buffers on V100 |
| `ds4_gpu.h` / `ds4_cuda.cu` | Modify | Bounded raw/compressed/indexer KV diagnostic APIs |
| `tests/cuda_v100_prefill_kv_smoke.c` | Create | CUDA prefill/KV ratio-class smoke |
| `tests/cuda_v100_context_smoke.c` | Modify | Exercise allocation/admission in production topology |
| `Makefile` | Modify | Build the new CUDA smoke |
| `docs/sprints/drafts/SPRINT-009-*.log` | Create | Validation evidence |
| `docs/sprints/SPRINT-009-REPORT.md` | Create | Sprint verdict |
| `docs/sprints/SPRINT-009-FOLLOWUPS.md` | Create | Follow-up/runtime handoff |
| `docs/sprints/VISION.md` | Modify | Reflect Sprint 009 outcome |

## Definition Of Done

- `ds4_v100_context` reports deterministic KV allocation offsets and sizes for
  explicit `ctx` and `slots`.
- The new CUDA prefill/KV smoke passes on V100 `sm_70`.
- The smoke covers both ratio-4 and ratio-128 layer classes.
- Source-layout guards pass on the real model through `--guards-only`.
- Local model-less validation and `git diff --check` pass.
- Normal source-layout generation remains guarded.
- Sprint report, follow-ups, and logs are committed.

## Risks

- The existing compressor/indexer kernels may expect older FP8/F32 cache shapes
  rather than the Sprint 008 F16 KV baseline.
- Full source-layout prefill may expose missing dense projection kernels; the
  sprint should stop at bounded prefill/KV execution rather than silently
  expanding to broad production GEMM work.
- It is easy to over-allocate per-stage scratch if active slots and configured
  slots are confused.

## Security

- Do not include model weights, generated secrets, or `logs/security/*` in
  commits.
- Keep source-layout oracle and prefill/KV harnesses diagnostic-only.
- Preserve fail-closed behavior for normal source-layout serving.

## Dependencies

- Access to the V100 cluster for `sm_70` CUDA validation.
- `/models/DSv4-Flash-256e-fixed.gguf` for guard validation.
- Sprint 008 source oracle, KV admission, and F8 source-format anchor commits.

## Open Questions

1. Should the first ratio-class smoke use layers 2/3 on gpu0 or a later pair on
   a nonzero stage to exercise non-gpu0 allocation?
2. Should the first source-format input tile produce F32 diagnostic output or
   F16 scratch output?
3. What numeric tolerance should gate F16 KV row/state comparisons?
