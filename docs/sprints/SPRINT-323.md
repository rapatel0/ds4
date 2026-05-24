---
sprint: 323
title: TP/EP Compressed-KV and Indexer Attention Gate
status: planned
started: 2026-05-24
branch: claude-takeover
---

# Sprint 323 - TP/EP Compressed-KV and Indexer Attention Gate

## Goal

Port the DeepSeek V4 compressed-attention cache sequence into the TP/EP
serving path far enough that attention reads raw SWA plus compressed rows, and
ratio-4 layers can select compressed rows through the indexer path.

## Why This Sprint

Sprint 322 closed the known FFN input-ordering gap: FFN norm, router, shared
FFN, and routed experts now consume the post-attention residual. The official
reference vector still fails, but the output changed again, proving the live
path is responding to the semantic gates.

The leading remaining graph-semantics gap is now attention memory:

```text
current TP/EP:
  raw SWA window only

needed DS4:
  raw SWA window
  + compressed attention rows for ratio-4 / ratio-128 layers
  + ratio-4 indexer row selection
  + raw+compressed softmax/value merge
```

The existing non-TP appliance code already documents the intended sequence in
`ds4_v100_layer_execute.c::prepare_decode_cache_attention()`. This sprint ports
that contract into the dedicated TP/EP file instead of reviving PP abstractions.

## Scope

TP/EP only. No PP/layer-split work. No MTP. No performance optimization beyond
keeping the implementation resident and avoiding obviously wasteful host
round-trips.

## Implementation Plan

1. Add explicit TP/EP flags:

```text
--true-ds4-compressed-kv-gate
--true-ds4-indexer-attention-gate
```

2. Extend TP/EP resident dense/control binding for compressor/indexer tensors:

```text
attn_compress_kv.weight
attn_compress_gate.weight
attn_compress_ape
attn_compress_norm.weight
indexer.attn_q_b.weight
indexer.proj.weight
indexer.compress_kv.weight
indexer.compress_gate.weight
indexer.compress_ape
indexer.compress_norm.weight
```

3. Add TP/EP per-rank cache buffers sized for a bounded first gate:

```text
raw_swa: existing
attn_state_kv / attn_state_score
attn_comp_kv
index_state_kv / index_state_score
index_comp_kv
indexer_topk
```

The first target is correctness at the official parity prompt length and the
`32` slot / `256K` serving shape. Full long-context allocation policy can stay
behind the later admission/planner gate, but the row math and cache update
contract must match DS4.

4. Port the decode update sequence from `ds4_v100_layer_execute.c`:

```text
attn_norm -> compressor kv/gate projections
compressor_update(attn)
if ratio == 4:
  indexer compressor kv/gate projections
  compressor_update(indexer)
  indexer q/w projections
  indexer scores
  top-k compressed row selection
```

5. Replace the raw-window-only attention read with:

```text
SWA-only layers:
  raw window attention

ratio-128 layers:
  raw + all available compressed rows

ratio-4 layers:
  raw + indexer-selected compressed rows
```

6. Rerun the same gates:

```text
32 slots / 256K all-layer smoke
HTTP short_reasoning_plain parity
```

## Definition of Done

- CUDA build passes on the V100 pod.
- `--true-ds4-compressed-kv-gate` passes a `32` slot / `256K` all-layer smoke
  with no finite/bad-shape/runtime failures.
- `--true-ds4-indexer-attention-gate` passes the same smoke on ratio-4 layers.
- Logs report, per layer:
  - ratio
  - emitted compressed row count
  - visible compressed row count
  - indexer top-k count for ratio-4
  - raw+compressed attention finite stats
- HTTP parity is rerun for `short_reasoning_plain`.
- `VISION.md`, `TEMP_STATUS_REPORT_035.md`, and cluster artifacts are updated
  and committed.

## Risks

- A bounded first cache may not prove 256K long-context behavior. It still
  proves the layer math and selection sequence before allocating full long
  history.
- The compressor tensors include BF16/F32 control paths, so this sprint may
  uncover another dtype/layout mismatch in the TP pack.
- If the indexer path selects rows but parity does not move, the next audit
  should compare logits/heads against the existing `ds4_v100_layer_execute.c`
  reference path at a single layer and token.
