---
sprint: 010
title: V100 Single-Slot Decode Integration
status: active
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-010-INTENT.md
deferred: SPRINT-010-DEFERRED.md
---

# SPRINT-010: V100 Single-Slot Decode Integration

## Overview

Sprint 009 proved bounded V100 F16 KV allocation and row/state update surfaces.
Sprint 010 connects those surfaces to a real single-slot layer-owned
prefill/decode integration gate. The sprint should still avoid public serving
and full 43-layer logits. Its job is to replace smoke-only standalone tensors
and synthetic state values with stage-owned KV arena views, actual
projection/compressor input tensors, real compressor/indexer recurrence, and a
bounded CPU/source reference comparison.

Normal source-layout generation remains fail-closed.

## Outcome Contract

- `SHIP`: a diagnostic single-slot V100 slice writes raw SWA, compressed
  attention KV, indexer KV, and split KV/score state through the stage-owned
  `kv_arena`, uses the real compressor/indexer recurrence for at least one
  ratio-4 layer and one ratio-128 layer, compares bounded outputs against a
  CPU/source reference, passes on V100 `sm_70`, and preserves all source-layout
  guards.
- `EXTEND`: stage-owned arena views and one ratio class ship, but the second
  ratio class or oracle comparison is blocked by a diagnosed projection,
  compressor, or source-reference mismatch.
- `STOP`: implementation requires normal source-layout serving unlock, full
  logits-producing decode, routed expert production kernels, host/SSD/offload
  as the success path, MTP/speculative decode, tensor parallelism, or throughput
  scheduling.

## Non-Goals

- No public CLI/server deployment.
- No normal source-layout serving unlock.
- No full 43-layer selected-token generation.
- No MTP or speculative decoding.
- No multi-slot batching, wavefront scheduling, or performance benchmarking.
- No tensor-parallel exceptions.
- No persistent dequantized copies of large source weights.
- No F8 KV baseline; F16 KV remains the correctness baseline.

## Precision Policy

V100 has no native BF16, FP8, or FP4 tensor-core execution. Sprint 010 must not
introduce a broad BF16 runtime path or a broad FP32 GEMM fallback. Source BF16
tensors should become FP16 runtime pack data or bounded FP16 scratch tiles
before production V100 GEMMs. Main dense and projection paths should target FP16
HMMA with FP32 accumulation, while FP32 remains limited to small
control/reduction paths, compressor state/score buffers, logits selection,
debug/oracle paths, and other accuracy-sensitive non-GEMM work.

## Planning Inputs

| File | Role |
|---|---|
| `docs/sprints/VISION.md` | Sprint 010 sequencing and Sprint 011 deployment deferral |
| `docs/architecture/DS4-V100-LAYOUT.md` | Topology, dtype, memory, and scheduling contract |
| `docs/sprints/SPRINT-009-REPORT.md` | Shipped KV arena and diagnostic update evidence |
| `docs/sprints/SPRINT-009-FOLLOWUPS.md` | Runtime integration, oracle comparison, and state-view follow-ups |
| `ds4_v100_context.[ch]` | Stage map, KV arena plan, and context report |
| `ds4_v100_context_cuda.cu` | CUDA stage ownership and allocated KV arena |
| `ds4_gpu.h`, `ds4_cuda.cu` | Source-format, compressor, indexer, attention, and KV CUDA APIs |
| `tools/ds4-source-oracle-vector.c` | Source-layout oracle and guard validation |
| `tests/cuda_v100_prefill_kv_smoke.c` | Sprint 009 bounded KV smoke |

## Use Cases

1. **Stage-owned KV writes**: a diagnostic path can address raw SWA,
   compressed KV, indexer KV, and state subviews inside the owning GPU's
   `kv_arena`.
2. **Real recurrence proof**: ratio-128 and ratio-4 CUDA smokes use the actual
   compressor/indexer recurrence instead of synthetic state formulas.
3. **Oracle comparison**: a bounded CPU/source reference can validate the V100
   slice without unlocking normal serving.
4. **Deployment gate clarity**: Sprint 011 can start from a real single-slot
   correctness surface instead of isolated allocation/update smokes.

## Architecture

The sprint extends the Sprint 009 diagnostic path:

```text
ds4_v100_context_open(ctx, slots)
    |
    +--> per-stage kv_arena plan
    |
    v
stage-owned CUDA kv_arena
    |
    +--> named subviews:
         raw_swa
         compressed_attn_kv
         indexer_kv
         attn_comp_kv_state
         attn_comp_score_state
         indexer_comp_kv_state
         indexer_comp_score_state
    |
    v
single-slot V100 diagnostic slice
    |
    +--> bounded source/projection input tensors
    +--> real compressor/indexer recurrence
    +--> CPU/source reference comparison
```

The integration should remain a diagnostic test/tool surface, not a serving
flag.

## Implementation

### Phase 1: KV Arena Subviews

**Files:**
- `ds4_v100_context.h`
- `ds4_v100_context.c`
- `ds4_v100_context_cuda.cu`
- `tests/v100_context_smoke.c`

**Tasks:**
- [x] Split `compression_state` into deterministic subviews for attention
      KV state, attention score state, indexer KV state, and indexer score
      state.
- [x] Add host-side helpers or descriptors for per-stage subview offsets and
      byte lengths.
- [x] Add CUDA-side accessors for stage-owned `kv_arena` subviews.
- [x] Keep reserve accounting unchanged and fail closed on overbudget plans.
- [x] Extend local context smoke assertions and report output.

### Phase 2: Stage-Owned Diagnostic KV Update

**Files:**
- `ds4_gpu.h`
- `ds4_cuda.cu`
- `ds4_v100_context_cuda.cu`
- `tests/cuda_v100_prefill_kv_smoke.c`

**Tasks:**
- [ ] Add a wrapper that writes Sprint 009 raw/compressed/indexer rows through
      stage-owned arena subviews instead of standalone tensors.
- [ ] Preserve explicit slot, raw-row, compressed-row, ratio, and bounds
      validation.
- [ ] Add a V100 smoke proving the stage-owned path for one ratio-128 and one
      ratio-4 layer.

### Phase 3: Real Compressor/Indexer Recurrence

**Files:**
- `ds4_gpu.h`
- `ds4_cuda.cu`
- `tests/cuda_v100_prefill_kv_smoke.c`
- optional CPU helper files

**Tasks:**
- [ ] Replace the Sprint 009 synthetic F32 state formula with existing DS4
      compressor/indexer recurrence kernels where possible.
- [ ] Feed bounded device tensors that represent actual compressor inputs.
- [ ] Cover both ratio-128 and ratio-4/indexer state paths.
- [ ] Compare CUDA state/KV rows against a CPU helper or source-oracle
      reference with documented tolerance.

### Phase 4: Source Reference And Guard Validation

**Files:**
- `tools/ds4-source-oracle-vector.c`
- `ds4_source_formats.[ch]`
- `docs/sprints/drafts/SPRINT-010-*.log`
- `docs/sprints/SPRINT-010-REPORT.md`
- `docs/sprints/SPRINT-010-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

**Tasks:**
- [ ] Add a bounded source-reference mode if the existing oracle cannot expose
      the required intermediate rows/state.
- [ ] Run model-less local validation and `git diff --check`.
- [ ] Run source-layout `--guards-only` on the real model.
- [ ] Run the integrated CUDA smoke on V100 `sm_70`.
- [ ] Archive logs under `docs/sprints/drafts/`.
- [ ] Write report/follow-ups and update `VISION.md` with the sprint verdict.

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `ds4_v100_context.[ch]` | Modify | Expose named KV/state subviews |
| `ds4_v100_context_cuda.cu` | Modify | Surface stage-owned KV arena subviews to diagnostics |
| `ds4_gpu.h`, `ds4_cuda.cu` | Modify | Stage-owned update wrapper and real recurrence integration |
| `tests/cuda_v100_prefill_kv_smoke.c` | Modify | Cover stage-owned arena and real compressor/indexer recurrence |
| `tools/ds4-source-oracle-vector.c` | Modify if needed | Bounded source-reference output |
| `docs/sprints/drafts/SPRINT-010-*.log` | Create | Validation evidence |
| `docs/sprints/SPRINT-010-REPORT.md` | Create | Sprint verdict |
| `docs/sprints/SPRINT-010-FOLLOWUPS.md` | Create | Handoff items |
| `docs/sprints/VISION.md` | Modify | Reflect Sprint 010 outcome |

## Definition Of Done

- Stage-local `kv_arena` reports and exposes deterministic named subviews for
  raw SWA, compressed attention KV, indexer KV, and split KV/score state.
- A V100 diagnostic path writes through the stage-owned arena subviews.
- The CUDA smoke covers ratio-128 and ratio-4/indexer real recurrence paths.
- Bounded CUDA outputs compare against a CPU/source reference.
- Source-layout guards pass on the real model.
- Local model-less validation and `git diff --check` pass.
- Normal source-layout generation remains guarded.
- Sprint report, follow-ups, and logs are committed.

## Risks

- Existing compressor/indexer kernels may not expose exactly the intermediate
  surfaces needed for a narrow source-oracle comparison.
- Actual projection/compressor inputs may reveal missing bounded FP8 dense
  projection coverage.
- State subview offsets can be easy to get wrong because Sprint 009 budgeted
  combined state bytes while production recurrence consumes KV and score state
  separately.

## Security

- Do not commit model weights, generated secrets, or `logs/security/*`.
- Keep all new source-reference and V100 execution paths diagnostic-only.
- Preserve fail-closed normal source-layout generation.

## Dependencies

- Access to the 8x V100-SXM2-32GB cluster for `sm_70` validation.
- `/models/DSv4-Flash-256e-fixed.gguf` for guard/source-reference validation.
- Sprint 009 KV arena and diagnostic prefill/KV update commits.

## Open Questions

1. Does the first source-reference comparison come from extending the oracle or
   from a standalone CPU helper for compressor/indexer state?
2. Should Sprint 010 require a real bounded FP8 dense projection, or is a
   source-decoded projection-equivalent tensor acceptable for the first real
   recurrence proof?
3. Which stage should host the first integrated smoke: gpu0 layers 2/3 for
   simplicity, or a nonzero stage to prove arena ownership away from embedding?
