# TEMP_STATUS_REPORT_047

Date: 2026-05-24

## Topline

Sprint 335 completed typed KV backing for the three TP/EP attention row
families currently exercised by the full-layer harness:

- raw-SWA rows,
- compressed attention rows,
- ratio-4 indexer rows.

The system still uses bounded f32 staging buffers for the attention read
surface, but those staged rows are now loaded back from the typed TP runtime
KV arena.

## What Changed

Updated:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`

New flag:

- `--true-ds4-attention-typed-kv-indexer-gate`

The ratio-4 indexer emit path now stores the emitted indexer row through
`DS4_V100_TP_KV_ROW_INDEXER`, loads it back into the bounded staging buffer,
then recomputes indexer scores/top-k from the loaded typed row.

## V100 Validation

Build:

- Command: `make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`
- Result: PASS

All-layer typed KV gate:

```text
typed raw-SWA lines: 43
typed compressed-attention lines: 41
typed indexer lines: 21
tp_ep_all_layer_scaffold ... pass_layers 43 ... PASS
projected_slot_step_tok_s 53.556562
sum_decode_ms_per_token 597.499147
```

Compact reference/indexer gate with all typed KV gates enabled:

```text
typed indexer lines: 21
compressed_reference_diff_summary lines: 21
tp_ep_all_layer_scaffold ... pass_layers 43 ... PASS
projected_slot_step_tok_s 21.859643
sum_decode_ms_per_token 1463.884850
```

Representative row:

```text
tp_ep_true_attention_typed_kv_indexer layer 2 slots 32 ratio 4 position 262143 bounded_row 0 visible_rows 1 physical_row 65535 logical_cols 128 logical_row_bytes 129 row_bytes_per_gpu 17 PASS
```

## Current Gap

The typed KV arena is now used for current emitted rows, but the full-layer
path still gathers visible compressed history through bounded staging buffers.
The next step is to replace the bounded diagnostic compressed-row model with
production typed-row lookup over visible compressed history, then promote this
from the smoke harness into the serving path.

## Artifacts

- `logs/from-cluster/sprint335-typed-indexer-kv/cluster/alllayers-typed-raw-compressed-indexer-s32-pos262143.log`
- `logs/from-cluster/sprint335-typed-indexer-kv/cluster/alllayers-typed-raw-compressed-indexer-diff-s32-pos262143.log`
