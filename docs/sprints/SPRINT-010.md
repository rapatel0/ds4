---
sprint: 010
title: V100 KV State Views And Compressor Bridge
status: planned
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-010-INTENT.md
deferred: SPRINT-010-DEFERRED.md
---

# SPRINT-010: V100 KV State Views And Compressor Bridge

## Overview

Sprint 009 proved that the V100 runtime can allocate derived F16 KV arenas and
update bounded raw/compressed/indexer rows on `sm_70`. The remaining gap before
deployment is that those writes are still diagnostic host-F32 inputs, not real
device-side layer outputs.

Sprint 010 narrows the next integration step: add explicit per-layer KV subviews
inside each stage arena, then bridge existing device F32 compressor/raw outputs
into the F16 KV cache contract without host round trips. This is still not a
public serving sprint and normal source-layout generation remains guarded.

## Outcome Contract

- `SHIP`: per-layer KV subviews are deterministic and tested; a CUDA diagnostic
  bridge stores device F32 raw/compressor outputs into F16 raw SWA, compressed
  attention KV, and ratio-4 indexer KV; V100 `sm_70` tests pass for ratio-4 and
  ratio-128; source-layout guards still pass.
- `EXTEND`: subviews and either ratio-4 or ratio-128 bridge ship, with a
  diagnosed blocker for the other ratio class.
- `STOP`: implementation requires normal serving unlock, full 43-layer logits,
  broad production FP8/MXFP4 GEMMs, persistent dequantized source weights,
  MTP/speculative decode, tensor parallelism, or throughput scheduling.

## Non-Goals

- No normal source-layout generation unlock.
- No public server/API deployment.
- No full selected-token V100 decode.
- No MTP.
- No multi-slot scheduling or throughput benchmark.
- No tensor-parallel exceptions.
- No persistent dequantized source-weight buffers.

## Planning Inputs

| File | Role |
|---|---|
| `docs/sprints/VISION.md` | Sprint 010 sequencing and deployment gate |
| `docs/architecture/DS4-V100-LAYOUT.md` | Topology, dtype, and scheduling contract |
| `docs/sprints/SPRINT-009-REPORT.md` | KV arena and bounded smoke outcome |
| `docs/sprints/SPRINT-009-FOLLOWUPS.md` | Runtime integration and subview follow-ups |
| `ds4_v100_context.[ch]` | Stage/layer topology and KV arena budget |
| `ds4_v100_context_cuda.cu` | V100 CUDA context and stage allocation |
| `ds4_gpu.h`, `ds4_cuda.cu` | CUDA tensor and compressor primitives |
| `tools/ds4-source-oracle-vector.c` | Source-layout guard validation |

## Use Cases

1. **Subview inspection**: a developer can inspect each layer's raw SWA,
   compressed attention, indexer KV, and state offsets inside its owning stage
   arena.
2. **Device-to-F16 KV bridge**: a CUDA diagnostic can take device F32 rows from
   compressor/raw outputs and store F16 cache rows without a host round trip.
3. **Ratio-class coverage**: the bridge covers ratio-128 attention compression
   and ratio-4 attention plus indexer compression.
4. **Guard continuity**: source-layout serving remains fail-closed while guard
   validation stays automated.

## Architecture

```text
stage KV arena
  |
  +-- layer N raw SWA F16 view
  +-- layer N compressed attention F16 view
  +-- layer N ratio-4 indexer F16 view
  +-- layer N attn state KV/score F32 views
  +-- layer N indexer state KV/score F32 views
        ^
        |
device F32 raw/compressor output rows
        |
        v
bounded F32 -> F16 KV bridge
```

The bridge is a diagnostic integration point, not the final fused production
kernel. It should make the data movement contract explicit before full layer
scheduler integration.

## Implementation

### Phase 1: Per-Layer KV Subviews

**Files:**
- `ds4_v100_context.h`
- `ds4_v100_context.c`
- `tests/v100_context_smoke.c`
- `tools/ds4-v100-context-smoke.c`

**Tasks:**
- [ ] Add per-layer KV view metadata for raw SWA, compressed attention,
      indexer KV, attn state KV/score, and indexer state KV/score.
- [ ] Derive layer views deterministically inside each stage arena.
- [ ] Keep stage totals identical to the Sprint 009 arena totals.
- [ ] Print layer view offsets/sizes in the context report.
- [ ] Reject invalid layer ids, ratio classes, or overflowing view spans.

### Phase 2: Device F32 To F16 KV Bridge

**Files:**
- `ds4_gpu.h`
- `ds4_cuda.cu`
- `tests/cuda_v100_prefill_kv_smoke.c`

**Tasks:**
- [ ] Add a diagnostic CUDA API that stores device F32 rows into F16 raw SWA
      and compressed attention KV views.
- [ ] Add ratio-4 indexer KV storage from device F32 rows.
- [ ] Preserve the Sprint 009 host-F32 diagnostic API as a test helper.
- [ ] Reject invalid slots, rows, dimensions, undersized tensors, and missing
      ratio-4 indexer inputs.
- [ ] Compare F16 row outputs against CPU F32-to-F16 references.

### Phase 3: Compressor Bridge Smoke

**Files:**
- `tests/cuda_v100_compressor_kv_bridge_smoke.c`
- `Makefile`
- `ds4_cuda.cu`

**Tasks:**
- [ ] Use existing CUDA compressor kernels with synthetic F32 APE/norm tensors
      to produce device F32 compressed rows for ratio-128.
- [ ] Use the ratio-4 path to produce attention and indexer compressed rows.
- [ ] Store those rows through the F16 KV bridge.
- [ ] Compare compressor/state outputs against a CPU reference for the bounded
      synthetic fixture.
- [ ] Run on V100 `sm_70`.

### Phase 4: Guards, Reports, And Vision

**Files:**
- `docs/sprints/drafts/SPRINT-010-*.log`
- `docs/sprints/SPRINT-010-REPORT.md`
- `docs/sprints/SPRINT-010-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

**Tasks:**
- [ ] Run local model-less validation.
- [ ] Run `git diff --check`.
- [ ] Run `tools/ds4-source-oracle-vector --guards-only` against the real model
      on the cluster.
- [ ] Run the new CUDA compressor/KV bridge smoke on V100 `sm_70`.
- [ ] Archive logs under `docs/sprints/drafts/`.
- [ ] Write report/follow-ups and update `VISION.md`.

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `ds4_v100_context.[ch]` | Modify | Per-layer KV subviews inside stage arenas |
| `ds4_v100_context_cuda.cu` | Modify if needed | Carry view metadata into CUDA context |
| `ds4_gpu.h`, `ds4_cuda.cu` | Modify | Device F32-to-F16 KV bridge |
| `tests/v100_context_smoke.c` | Modify | Subview math and bounds regression |
| `tests/cuda_v100_prefill_kv_smoke.c` | Modify | Device-row bridge regression |
| `tests/cuda_v100_compressor_kv_bridge_smoke.c` | Create | Existing compressor to F16 KV bridge smoke |
| `Makefile` | Modify | Build new CUDA smoke |
| `docs/sprints/drafts/SPRINT-010-*.log` | Create | Evidence |
| `docs/sprints/SPRINT-010-REPORT.md` | Create | Verdict |
| `docs/sprints/SPRINT-010-FOLLOWUPS.md` | Create | Handoff |
| `docs/sprints/VISION.md` | Modify | Reflect outcome |

## Definition Of Done

- Per-layer KV subviews are deterministic, reported, and tested.
- Stage arena totals remain unchanged from Sprint 009 for the same `ctx/slots`.
- Device F32-to-F16 KV bridge passes local compile and V100 `sm_70` tests.
- Compressor bridge smoke covers ratio-4 and ratio-128 paths.
- Source-layout guards pass on the real model.
- Normal source-layout generation remains guarded.
- Sprint report, follow-ups, logs, and vision update are committed.

## Risks

- Existing CUDA compressor kernels use F32 cache tensors; bridging to F16 KV may
  expose precision or layout assumptions.
- Ratio-4 compressor replay/state behavior is more complex than ratio-128 and
  may need a narrower `EXTEND` outcome.
- A full oracle comparison may still require a later selected-token sprint.

## Security

- Do not commit model weights, large generated artifacts, or secrets.
- Keep source-layout guard bypasses diagnostic-only.
- Preserve fail-closed behavior for normal source-layout serving.

## Dependencies

- V100 cluster access.
- `/models/DSv4-Flash-256e-fixed.gguf` for guard validation.
- Sprint 009 `SHIP` commits.

## Open Questions

1. Should the first bridge store only F16 rows or also expose F8 KV as an
   experimental option behind a guard?
2. Should the subview report be stage-first or layer-first for downstream
   scheduler consumption?
3. What source-oracle comparison becomes the deployment gate after this bridge?
