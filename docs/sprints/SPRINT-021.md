---
sprint: 021
title: Executor-Owned Compressor/Indexer Decode Rows
status: planned
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-021-INTENT.md
verdict: pending
---

# SPRINT-021: Executor-Owned Compressor/Indexer Decode Rows

## Overview

Sprint 021 removes the next major test fixture from the layer path. The layer
executor should own compressed attention and indexer row generation from real
source descriptors instead of receiving prebuilt compressed KV rows.

The target is still layer 2 first because it exercises the hardest ratio-4
path: attention compression, indexer compression, indexer top-k, and mixed
raw plus compressed attention.

## Outcome Contract

- `SHIP`: layer 2 can execute with mutable decoder-owned raw/compressed/indexer
  state, emit descriptor-bound compressed rows, use indexed compressed
  attention when needed, and pass the V100 gate.
- `EXTEND`: descriptor-bound compressed row emission ships, but indexed
  attention or full integrated use remains blocked by a concrete hardware
  finding.
- `STOP`: existing compressor/indexer CUDA APIs cannot model DS4 source
  recurrence without cache-layout redesign.

## Non-Goals

- No full 43-layer selected-token claim.
- No public server.
- No MTP.
- No throughput benchmark.
- No persistent dequantized weights.

## Parallel Workstreams

| Lane | Responsibility | Write Scope | Validation |
|---|---|---|---|
| A: executor state contract | Add mutable decode-cache struct and config plumbing. | `ds4_v100_layer_execute.*` | local compile |
| B: compressor row emission | Project BF16 compressor KV/score rows, update state, emit rows. | executor + integrated smoke | descriptor-bound row smoke |
| C: ratio-4 indexer path | Project indexer rows, score/top-k, call indexed mixed attention. | executor | long-enough synthetic cache smoke |
| D: evidence | Run one-card and 8-GPU gates, update docs. | tests/docs | V100 logs |

## Implementation

### Phase 1: Decode-Cache Contract

**Files:**
- `ds4_v100_layer_execute.h`
- `ds4_v100_layer_execute.c`

**Tasks:**
- [ ] Add a mutable decode-cache struct for raw KV, attention compressed KV,
      indexer compressed KV, compressor recurrence state, and top-k scratch.
- [ ] Keep the existing explicit compressed-KV config path for compatibility.
- [ ] Validate tensor sizes and ratio-class requirements before execution.

### Phase 2: Attention Compressor Emission

**Files:**
- `ds4_v100_layer_execute.c`
- `tests/cuda_v100_integrated_layer_smoke.c`

**Tasks:**
- [ ] Create BF16 views for `attn_compressor_kv` and
      `attn_compressor_gate`.
- [ ] Project current KV/score rows from `attn_norm`.
- [ ] Call `ds4_gpu_compressor_update_tensor` and quantize emitted attention
      compressed rows.
- [ ] Increment the mutable compressed-row count only on ratio boundaries.

### Phase 3: Ratio-4 Indexer

**Files:**
- `ds4_v100_layer_execute.c`
- `tests/cuda_v100_integrated_layer_smoke.c`

**Tasks:**
- [ ] Generate ratio-4 indexer compressed rows from `attn_norm`.
- [ ] Project `indexer_q` from `q_a_norm` and `indexer_weights` from
      `attn_norm`.
- [ ] Run score/top-k when `n_index_comp > 512`.
- [ ] Use `ds4_gpu_attention_indexed_mixed_batch_heads_tensor` for indexed
      compressed attention.

### Phase 4: Gate

**Files:**
- `docs/sprints/SPRINT-021-REPORT.md`
- `docs/sprints/SPRINT-021-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

**Tasks:**
- [ ] Run local compile and descriptor smokes.
- [ ] Run one-card V100 integrated smoke.
- [ ] Run full 8-GPU V100 gate.
- [ ] Commit code, reports, and logs.

## Definition Of Done

- Executor-owned compressed row emission is available for layer 2.
- Ratio-4 indexer state and visibility are generated from real descriptors.
- Existing explicit compressed-KV fixture path still passes.
- V100 gate passes and remains `ready=false` for selected-token, serving, MTP,
  and throughput.
