---
sprint: 326
title: TP/EP Bounded Multi-Row Compressed Attention History
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 326 - TP/EP Bounded Multi-Row Compressed Attention History

## Goal

Move the TP/EP compressed-attention path from a one-row diagnostic to a bounded
multi-row history diagnostic that exercises row append, visible-row accounting,
indexer row selection, and raw+compressed attention over multiple compressed
rows.

## Why This Sprint

Sprint 325 fixed a real layer-local state bug, but the harness still only
stores and reads one visible compressed row. That proves the emit pipeline for
one row, not the history behavior required for long-context DS4 serving.

The next useful step is to validate the same code shape over several emitted
rows while keeping memory bounded. This keeps the work aligned with production
TP/EP compressed KV without jumping directly to the full 256K history allocator
before the row lifecycle is proven.

## Scope

TP/EP only. No PP/layer-split work. No MTP. This sprint remains a diagnostic
and smoke-runtime step, not a full production KV allocator.

## Implementation Plan

1. Replace the hard one-row bounded compressed cache with a small bounded row
   capacity.

```text
kBoundedCompRows = 8
```

2. Track visible compressed rows per layer in the TP/EP smoke state:

- attention compressed rows written
- indexer compressed rows written
- row index = written % bounded capacity
- visible rows = min(written, bounded capacity)

3. Update ratio-4 indexer scoring:

- score every visible bounded indexer row
- fill the top-k index array with all visible bounded rows
- replicate the selected row indices from rank 0 to the other TP ranks

4. Update raw+compressed attention:

- accept multiple visible compressed rows
- include every selected bounded compressed row in the softmax denominator
- accumulate all selected compressed rows into the output head

5. Extend the compact reference diff:

- compare the newly emitted row at its actual bounded row index
- record visible-row count and selected-row count
- keep the same limitation explicit: this is same-layout compact reference,
  not full non-TP reference replay

## Target Validation

Run on the V100 pod with shared resident state:

```text
slots=32
ctx=262144
decode_steps=8
position=262140
```

This crosses two ratio-4 emit points and should exercise at least two bounded
compressed rows for ratio-4 layers. A smaller `slots=1` run is acceptable as a
debugging aid but is not sufficient for the DoD.

## Definition of Done

- [x] CUDA build passes on the V100 pod.
- [x] `32` slot / `256K` / `8` step all-layer smoke passes.
- [x] Logs show at least one ratio-4 layer with `visible_compressed_rows >= 2`.
- [x] Logs show raw+compressed attention reading `selected_compressed_rows >= 2`
  for at least one ratio-4 layer.
- [x] Compact compressed-reference diffs pass for emitted bounded rows.
- [x] `VISION.md`, `TEMP_STATUS_REPORT_038.md`, cluster artifacts, and this
  sprint doc are updated and committed.

## Outcome

Implemented bounded multi-row compressed attention history in the TP/EP
full-layer smoke:

- `kBoundedCompRows` is now `8`.
- Raw-SWA, attention-compressed, and indexer-compressed buffers remain
  layer-local.
- Each layer tracks bounded compressed rows written for attention and indexer
  rows.
- Emitted rows are appended by ring row index.
- Ratio-4 indexer scoring covers every bounded visible row and replicates
  selected indices from rank 0 to the other TP ranks.
- Raw+compressed attention now includes multiple selected compressed rows in
  the softmax denominator and output accumulation.
- Compact reference diffs compare the emitted bounded row against live
  pre-shift compressor state, so multi-step history is represented.

Two bugs were found during execution and fixed:

- The first `8` step run used `position=262140`, which crossed the configured
  `ctx=262144` limit at step `4`. The passing run starts at `262135`, keeping
  all `8` positions inside context while still crossing two ratio-4 emit
  points.
- The compact reference buffer has row capacity `1`; using the actual ring row
  index for compact reference norm/RoPE/round made row `1+` reference ops a
  no-op. The reference now writes/normalizes the compact row at index `0` while
  comparing against the actual TP ring row.

## Validation

Build on `llm/llamacpp-build-8gpu`:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS. Existing warnings remain:

- unused `rms_norm_plain_rows_kernel`
- unused legacy `indexer_score_row0_slots_kernel`

Passing V100 gate:

```text
slots=32
ctx=262144
position=262135
decode_steps=8
layers=43
pass_invocations=344
projected_slot_step_tok_s=20.780883
checksum=2118198918
```

Evidence:

- `tp_ep_token_major_scaffold ... PASS`
- ratio-4 layers report `visible_compressed_rows=2`
- raw+compressed attention reports `selected_compressed_rows=2`
- `grep -E 'DIFF|FAIL'` on the passing log returns no rows

Cluster artifact:

- `logs/from-cluster/sprint326-bounded-multirow/cluster/alllayers-slots32-pos262135-steps8-attn-only-v3.log`

## Follow-Up

This removes the one-row diagnostic limitation, but it is still a bounded
diagnostic cache. The next correctness sprint should move toward production
compressed-KV ownership:

- allocate the production row budget without materializing every diagnostic
  buffer for every layer at once
- validate ratio-128 multi-row history as well as ratio-4
- compare raw+compressed attention output against a full reference-layer path
- rerun HTTP reference parity after the local attention diff is stronger

## Risks

- If memory is tighter than expected, keep the bounded row capacity small and
  document the resulting cap instead of reintroducing one-row semantics.
- The indexer top-k implementation is intentionally bounded. It should prove
  row lifecycle and multi-row attention shape, not final full-history ranking
  performance.
- A pass here still does not prove full long-context parity; it only removes
  the one-row diagnostic limitation.
