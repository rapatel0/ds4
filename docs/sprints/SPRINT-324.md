---
sprint: 324
title: TP/EP Compressed Row Storage and Raw+Compressed Attention Read
status: executed
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

## Outcome

Sprint 324 implemented the bounded TP/EP compressed-row path in
`tools/ds4-v100-tp-ep-full-layer-smoke.cu`.

Implemented:

- Per-rank resident attention compressor state/cache buffers:
  `attn_comp_state_kv`, `attn_comp_state_score`, and `attn_comp_rows`.
- Per-rank resident indexer compressor state/cache buffers:
  `index_comp_state_kv`, `index_comp_state_score`, `index_comp_rows`,
  `indexer_scores`, and `indexer_topk`.
- Shared gathered compressor/indexer projection buffers on GPU0.
- Decode compressor update sequence:
  - gather TP compressor projection shards
  - store the current ratio phase with APE
  - pool state on emit boundary
  - RMSNorm emitted rows
  - RoPE emitted rows
  - F16 round-trip emitted rows
  - ratio-4 state shift
- Raw+compressed attention read for emitted rows.
- Ratio-4 bounded indexer scoring for the single visible compressed row and
  top-k selection seeded from that score.

This is intentionally bounded: the first pass stores and reads row `0` on
emit boundaries. It proves the resident compressed-row lifecycle and attention
merge path, but it is not yet the full long-history compressed cache.

## Validation

Build on V100 pod:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS.

32-slot / 256K all-layer smoke:

- Shape: `32` slots, `256K`, all `43` layers, position `262143`.
- Result: PASS.
- Final scaffold:
  - `pass_invocations=43`
  - `sum_decode_ms=1670.069087`
  - `projected_slot_step_tok_s=19.160884`
  - `checksum=6655226894`
- Ratio-128 layers report `visible_compressed_rows=1` and
  `selected_compressed_rows=1`.
- Ratio-4 layers report `indexer_topk_count=512`,
  `visible_compressed_rows=1`, and `selected_compressed_rows=1`.

HTTP `short_reasoning_plain` parity:

- Result: FAIL, request completed.
- Expected: `16`
- Actual: `mere`
- Generated token: `88445`
- Wall tok/s: `20.366798`
- Decode tok/s: `21.214211`

The parity result returning to `mere` means the bounded compressed-read path is
not yet sufficient to improve top-token parity. The path is active during prompt
positions that hit ratio-4 emit boundaries, so the next sprint should compare
TP/EP compressed rows and attention outputs directly against the non-TP
reference for one ratio-4 layer.

## Artifacts

- `logs/from-cluster/sprint324-compressed-row-storage-v2/cluster/all-layer-smoke.log`
- `logs/from-cluster/sprint324-compressed-row-storage-v2/cluster/http-parity/`
