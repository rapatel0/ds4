# TEMP_STATUS_REPORT_046

Date: 2026-05-24

## Topline

Sprint 334 integrated emitted compressed attention rows with the TP runtime's
typed row-sharded KV arena.

The full-layer attention path now has typed KV backing for:

- raw-SWA rows via `DS4_V100_TP_KV_ROW_ATTN_RAW`,
- compressed attention rows via `DS4_V100_TP_KV_ROW_ATTN`.

## What Changed

Updated:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`

New flag:

- `--true-ds4-attention-typed-kv-compressed-gate`

The compressed emit path stores each emitted bounded row into the runtime's
physical compressed attention row, then loads it back into the bounded staging
buffer before raw+compressed attention reads.

## V100 Validation

Build:

- Command: `make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`
- Result: PASS

All-layer gate:

- `32` slots
- `256K` context
- position `262143`
- `--true-ds4-attention-typed-kv-raw-gate`
- `--true-ds4-attention-typed-kv-compressed-gate`
- `--true-ds4-attention-raw-window-gate`

Result:

```text
typed raw-SWA lines: 43
typed compressed-attention lines: 41
tp_ep_all_layer_scaffold ... pass_layers 43 ... PASS
projected_slot_step_tok_s 51.386758
sum_decode_ms_per_token 622.728523
```

Representative compressed rows:

```text
layer 2 ratio 4   physical_row 65663
layer 3 ratio 128 physical_row 2175
```

## Current Gap

Ratio-4 indexer rows still use the bounded f32 diagnostic row path. The next
step is to store/load emitted indexer rows through `DS4_V100_TP_KV_ROW_INDEXER`
and then rerun the compact reference/indexer gates.

## Artifact

- `logs/from-cluster/sprint334-typed-compressed-attn/cluster/alllayers-typed-raw-compressed-s32-pos262143.log`
