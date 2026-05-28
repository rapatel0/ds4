# TEMP_STATUS_REPORT_048

Date: 2026-05-24

## Topline

Sprint 336 added typed compressed-history reload.

The full-layer path now reloads visible compressed attention and ratio-4
indexer history from the production typed TP KV arena before raw+compressed
attention reads.

## What Changed

Updated:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`

New flag:

- `--true-ds4-attention-typed-kv-history-gate`

New state:

- emitted compressed attention row source positions
- emitted ratio-4 indexer row source positions

The history gate loads visible rows from typed KV into the bounded staging
buffers and recomputes indexer top-k after indexer row reload.

## V100 Validation

Build:

- Command: `make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`
- Result: PASS

Multi-step all-layer token-major gate:

```text
slots=32
ctx=262144
position=262136
decode_steps=8
layers=43
```

Result:

```text
tp_ep_true_attention_typed_kv_history lines: 328
loaded_attn_rows=2 lines: 21
loaded_indexer_rows=2 lines: 21
tp_ep_token_major_scaffold ... pass_invocations 344 ... PASS
projected_slot_step_tok_s 72.282389
ms_per_token 442.708112
```

Representative history reload:

```text
tp_ep_true_attention_typed_kv_history layer 42 slots 32 ratio 4 visible_attn_rows 2 loaded_attn_rows 2 loaded_indexer_rows 2 PASS
```

## Current Gap

The full-layer smoke now exercises typed current rows and typed visible
history. The next step is to promote these gates into the TP/EP serving decode
path and run tokenizer-enabled HTTP serving with resident session/KV reuse.

## Artifact

- `logs/from-cluster/sprint336-typed-history-reload/cluster/tokenmajor8-typed-history-s32-pos262136.log`
