---
sprint: 334
title: TP/EP Typed Compressed Attention KV
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 334 - TP/EP Typed Compressed Attention KV

## Goal

Route emitted compressed attention rows through the TP runtime's production
typed F8 row-sharded KV arena in the full-layer attention path.

## Why This Sprint

Sprint 333 corrected raw-SWA physical addressing. The next production KV gap
was compressed attention rows: ratio-4 and ratio-128 layers still emitted into
bounded f32 diagnostic buffers only. Production serving needs those rows to
land in the runtime's long compressed attention region.

## Scope

TP/EP only. No PP/layer-split work. No MTP. This sprint covers compressed
attention rows only. Ratio-4 indexer rows remain next.

## Definition of Done

- [x] Add an opt-in full-layer typed compressed-attention gate.
- [x] Preserve `DS4_V100_TP_KV_ROW_ATTN_RAW` for raw-SWA.
- [x] Use `DS4_V100_TP_KV_ROW_ATTN` for emitted compressed attention rows.
- [x] Store and load emitted compressed rows through the TP runtime typed KV
      APIs before downstream raw+compressed attention reads.
- [x] Validate all 43 layers at `32` slots / `256K`, with both typed raw-SWA
      and typed compressed-attention gates enabled.

## Outcome

Added `--true-ds4-attention-typed-kv-compressed-gate`.

When a ratio layer emits a compressed attention row, the full-layer path now:

1. computes the bounded diagnostic compressed row as before,
2. stores that row through `ds4_v100_tp_runtime_kv_row_store_f32_device` with
   `DS4_V100_TP_KV_ROW_ATTN`,
3. synchronizes the typed stores,
4. loads the row back through `ds4_v100_tp_runtime_kv_row_load_f32_device`,
   and
5. leaves the loaded row in the existing bounded staging buffer for the
   raw+compressed attention read.

This keeps the existing bounded read path usable while making the backing row
come from the production typed compressed-attention arena.

## Validation

Build on `llm/llamacpp-build-8gpu`:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS.

All-layer shared-state validation at `32` slots / `256K`, position `262143`:

```text
typed raw-SWA lines: 43
typed compressed-attention lines: 41
tp_ep_all_layer_scaffold ... pass_layers 43 ... PASS
projected_slot_step_tok_s 51.386758
sum_decode_ms_per_token 622.728523
```

The first compressed rows show the expected physical rows:

```text
layer 2 ratio 4   physical_row 65663
layer 3 ratio 128 physical_row 2175
```

Artifacts:

- `logs/from-cluster/sprint334-typed-compressed-attn/cluster/alllayers-typed-raw-compressed-s32-pos262143.log`

## Next Step

Apply the same typed KV integration pattern to ratio-4 indexer rows using
`DS4_V100_TP_KV_ROW_INDEXER`, then re-run the compact reference/indexer gates.
