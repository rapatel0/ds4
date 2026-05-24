# TEMP_STATUS_REPORT_044

Date: 2026-05-24

## Topline

Sprint 332 integrated the TP runtime's typed row-sharded KV arena into the
full-layer raw-SWA attention path.

Raw-SWA rows can now be stored through the production F8 E4M3 block-128
physical KV shards and loaded back into the attention read buffer in the
full-layer TP/EP smoke.

Correction: Sprint 333 found that Sprint 332 used the generic `ATTN` row kind,
which addresses the compressed long-attention row on ratio layers. Sprint 332
proved the store/load plumbing; Sprint 333 corrected raw-SWA physical row
addressing with `DS4_V100_TP_KV_ROW_ATTN_RAW`.

## What Changed

Updated:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`

New flag:

- `--true-ds4-attention-typed-kv-raw-gate`

Runtime plumbing:

- `run_decode_loop` now receives the active `ds4_v100_tp_runtime *`.
- `run_true_ds4_attention_state_update` uses that runtime when the typed raw
  gate is enabled.
- Each slot's `[512]` raw-SWA KV row is written through
  `ds4_v100_tp_runtime_kv_row_store_f32_device`.
- The row is then decoded back through
  `ds4_v100_tp_runtime_kv_row_load_f32_device` into the existing per-rank
  raw-SWA staging buffer.

## V100 Validation

Build:

- Command: `make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`
- Result: PASS

Full shared-state all-layer gate:

- `32` slots
- `256K` context
- position `262140`
- all `43` layers
- `--share-tp-runtime`
- `--shared-expert-bindings`
- `--shared-dense-ops`
- `--true-ds4-attention-typed-kv-raw-gate`
- `--true-ds4-attention-raw-window-gate`

Result:

```text
tp_ep_true_attention_typed_kv_raw lines: 43
tp_ep_all_layer_scaffold ... pass_layers 43 ... PASS
projected_slot_step_tok_s 93.259761
sum_decode_ms_per_token 343.127622
```

The row view matches the expected F8 block-128 physical layout:

```text
logical_cols=512
logical_row_bytes=516
row_bytes_per_gpu=65
```

## Current Gap

This sprint covers raw-SWA only. The remaining KV productionization work is:

- typed compressed attention rows,
- typed ratio-4 indexer rows,
- replacing the bounded diagnostic compressed-row buffers,
- then re-running semantic/reference gates through the serving harness.

## Artifacts

- `logs/from-cluster/sprint332-typed-raw-swa/cluster/alllayers-typed-raw-window-s32-pos262140.log`
- `logs/from-cluster/sprint332-typed-raw-swa/cluster/layer2-typed-raw-window-s32-pos262140.log`
