---
sprint: 335
title: TP/EP Typed Ratio-4 Indexer KV
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 335 - TP/EP Typed Ratio-4 Indexer KV

## Goal

Route emitted ratio-4 indexer rows through the TP runtime's production typed
F8 row-sharded KV arena in the full-layer attention path.

## Why This Sprint

Sprints 333-334 gave the attention path typed backing for raw-SWA and
compressed attention rows. Ratio-4 indexer rows were the remaining KV row
family still produced and consumed only through bounded f32 diagnostic buffers.

## Scope

TP/EP only. No PP/layer-split work. No MTP. This sprint covers emitted ratio-4
indexer rows. It keeps the bounded staging buffer as the read surface for now,
but the staged row is loaded back from `DS4_V100_TP_KV_ROW_INDEXER`.

## Definition of Done

- [x] Add an opt-in full-layer typed indexer KV gate.
- [x] Use `DS4_V100_TP_KV_ROW_INDEXER` for emitted ratio-4 indexer rows.
- [x] Store and load emitted indexer rows through the TP runtime typed KV APIs.
- [x] Recompute indexer scores/top-k from the loaded typed row before
      downstream raw+compressed attention reads.
- [x] Validate all 43 layers at `32` slots / `256K` with raw-SWA,
      compressed-attention, and indexer typed KV enabled.
- [x] Re-run the compact reference/indexer diagnostic gate.

## Outcome

Added `--true-ds4-attention-typed-kv-indexer-gate`.

When a ratio-4 layer emits an indexer row, the full-layer path now:

1. computes the bounded diagnostic indexer row as before,
2. stores that row through `ds4_v100_tp_runtime_kv_row_store_f32_device` with
   `DS4_V100_TP_KV_ROW_INDEXER`,
3. synchronizes the typed stores,
4. loads the row back through `ds4_v100_tp_runtime_kv_row_load_f32_device`,
5. recomputes indexer scores/top-k from the loaded row, and
6. broadcasts top-k indices to the other ranks as before.

## Validation

Build on `llm/llamacpp-build-8gpu`:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS.

All-layer typed KV validation at `32` slots / `256K`, position `262143`:

```text
typed raw-SWA lines: 43
typed compressed-attention lines: 41
typed indexer lines: 21
tp_ep_all_layer_scaffold ... pass_layers 43 ... PASS
projected_slot_step_tok_s 53.556562
sum_decode_ms_per_token 597.499147
```

Representative indexer row:

```text
layer 2 ratio 4 physical_row 65535 logical_cols 128 logical_row_bytes 129 row_bytes_per_gpu 17
```

Compact reference/indexer validation with all typed KV gates enabled:

```text
typed indexer lines: 21
compressed_reference_diff_summary lines: 21
tp_ep_all_layer_scaffold ... pass_layers 43 ... PASS
projected_slot_step_tok_s 21.859643
sum_decode_ms_per_token 1463.884850
```

Artifacts:

- `logs/from-cluster/sprint335-typed-indexer-kv/cluster/alllayers-typed-raw-compressed-indexer-s32-pos262143.log`
- `logs/from-cluster/sprint335-typed-indexer-kv/cluster/alllayers-typed-raw-compressed-indexer-diff-s32-pos262143.log`

## Next Step

Replace the bounded diagnostic compressed-row staging model with production
typed-row lookup for visible compressed history, then wire this into the
serving path instead of only the full-layer smoke.
