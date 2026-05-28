# TEMP_STATUS_REPORT_045

Date: 2026-05-24

## Topline

Sprint 333 corrected typed raw-SWA physical addressing.

The TP runtime now has an explicit `DS4_V100_TP_KV_ROW_ATTN_RAW` row kind, and
the full-layer typed raw-SWA gate uses it. On ratio layers this writes
`position % 128`, not the compressed long-attention row.

## Why This Was Needed

Sprint 332 used `DS4_V100_TP_KV_ROW_ATTN` for raw-SWA. That row kind maps to
the compressed attention region on ratio layers:

```text
physical_row = 128 + position / ratio
```

For raw-SWA, the correct physical row is always:

```text
physical_row = position % 128
```

So Sprint 332 proved full-layer typed store/load plumbing, but Sprint 333 is
the correction that makes the raw-SWA gate physically correct.

## What Changed

Updated:

- `ds4_v100_tp_runtime.h`
- `ds4_v100_tp_runtime.cu`
- `tools/ds4-v100-tp-runtime-smoke.cu`
- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`

New row kind:

- `DS4_V100_TP_KV_ROW_ATTN_RAW`

Runtime smoke now accepts:

- `--kind attn_raw`

## V100 Validation

Build:

- Command: `make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-runtime-smoke tools/ds4-v100-tp-ep-full-layer-smoke`
- Result: PASS

Row-view contrast at layer `2`, slot `31`, position `262140`:

| Kind | Physical row | Bad values | Max abs | Result |
|---|---:|---:|---:|---|
| `attn_raw` | `124` | `0` | `0.000000000` | PASS |
| `attn` | `65663` | `0` | `0.000000000` | PASS |

Corrected all-layer gate:

```text
tp_ep_true_attention_typed_kv_raw lines: 43
tp_ep_true_attention_typed_kv_raw layer 0 ... physical_row 124 raw_row 124 ... PASS
tp_ep_all_layer_scaffold ... pass_layers 43 ... PASS
projected_slot_step_tok_s 72.313683
sum_decode_ms_per_token 442.516531
```

## Current Gap

Raw-SWA now has correct typed KV storage. The next production KV work is:

- compressed attention row typed store/load,
- ratio-4 indexer row typed store/load,
- replacing bounded f32 diagnostic compressed-row buffers in the full-layer
  attention read path.

## Artifacts

- `logs/from-cluster/sprint333-raw-swa-row-kind/cluster/device-row-attn-raw-layer2-slot31-pos262140.log`
- `logs/from-cluster/sprint333-raw-swa-row-kind/cluster/device-row-attn-compressed-layer2-slot31-pos262140.log`
- `logs/from-cluster/sprint333-raw-swa-row-kind/cluster/alllayers-typed-raw-window-s32-pos262140.log`
