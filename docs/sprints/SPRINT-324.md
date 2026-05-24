---
sprint: 324
title: TP/EP Compressed Row Storage and Raw+Compressed Attention Read
status: planned
started: 2026-05-24
branch: claude-takeover
---

# Sprint 324 - TP/EP Compressed Row Storage and Raw+Compressed Attention Read

## Goal

Continue the Sprint 323 compressed-attention work by turning compressor/indexer
projections into resident compressed rows and making the TP/EP attention read
consume raw SWA plus compressed rows.

## Why This Sprint

Sprint 323 proved the resident tensor binding and projection side of the DS4
compressed attention path. It did not yet store emitted compressed rows,
compute indexer scores/top-k, or merge compressed rows into the attention read.
Reference parity still fails, but the live token changed from `mere` to `MARK`,
so the new path is active and worth continuing.

## Scope

TP/EP only. No PP/layer-split work. No MTP. Keep the implementation in the
dedicated TP/EP harness and supporting TP files.

## Implementation Plan

1. Add bounded resident compressed-cache buffers to the TP/EP rank/shared state:

```text
attn_comp_state_kv
attn_comp_state_score
attn_comp_rows
index_comp_state_kv
index_comp_state_score
index_comp_rows
indexer_topk
```

2. Port the DS4 compressor update sequence for decode:

```text
project kv/gate
store state row with APE
pool state on ratio boundary
RMSNorm pooled row
RoPE tail at emitted-row position
ratio-4 state shift
```

3. Add ratio-specific behavior:

- ratio `0`: raw SWA only
- ratio `128`: raw SWA plus all visible compressed rows
- ratio `4`: raw SWA plus indexer-selected compressed rows

4. Implement the first indexer scoring/top-k gate for ratio-4:

```text
indexer_q = indexer.attn_q_b(q_a_normed) + RoPE tail
indexer_w = indexer.proj(attn_normed)
scores = q dot compressed_index_rows
topk = 512 rows
```

5. Replace the raw-window-only read with a raw+compressed read that logs:

- raw rows read
- compressed rows visible
- compressed rows selected
- softmax finite stats
- output finite stats

## Definition of Done

- CUDA build passes on the V100 pod.
- 32-slot / 256K all-layer smoke passes with real compressed row storage.
- Ratio-128 layers report visible compressed rows and raw+compressed attention.
- Ratio-4 layers report indexer score/top-k and selected compressed rows.
- HTTP `short_reasoning_plain` parity is rerun.
- `VISION.md`, `TEMP_STATUS_REPORT_036.md`, and cluster artifacts are updated
  and committed.

## Risks

- The exact compressor pooling/APE/norm/RoPE sequence may need line-by-line
  comparison against `ds4_gpu_compressor_update_tensor`.
- A bounded compressed-cache first pass may not prove full 256K admission, but
  it is enough to prove layer math and row-selection semantics.
- If parity still does not move, the next diagnostic should compare one layer's
  compressed rows and attention output against the non-TP reference path.
