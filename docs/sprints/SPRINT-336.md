---
sprint: 336
title: TP/EP Typed Compressed History Reload
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 336 - TP/EP Typed Compressed History Reload

## Goal

Reload visible compressed attention and ratio-4 indexer history from the
production typed TP KV arena before the full-layer raw+compressed attention
read.

## Why This Sprint

Sprints 333-335 stored and loaded the current emitted row families through the
typed KV arena. The attention read path still relied on bounded f32 staging
buffers as the source of visible compressed history. This sprint keeps the
bounded staging surface, but repopulates it from typed KV using recorded
emitted-row positions before the attention read.

## Scope

TP/EP only. No PP/layer-split work. No MTP. This sprint operates in the
full-layer smoke harness. It does not yet replace the bounded staging read
kernel with direct production row lookup in the serving path.

## Definition of Done

- [x] Track emitted compressed attention row positions per layer and bounded
      row.
- [x] Track emitted ratio-4 indexer row positions per layer and bounded row.
- [x] Add an opt-in typed history reload gate.
- [x] Reload visible compressed attention rows from `DS4_V100_TP_KV_ROW_ATTN`.
- [x] Reload visible ratio-4 indexer rows from `DS4_V100_TP_KV_ROW_INDEXER`.
- [x] Recompute and broadcast indexer top-k after indexer history reload.
- [x] Validate a multi-step all-layer run that reaches more than one visible
      compressed row.

## Outcome

Added `--true-ds4-attention-typed-kv-history-gate`.

The full-layer path now records the source position for each bounded emitted
compressed row. Before raw+compressed attention reads, the history gate loads
all visible bounded rows from the typed runtime KV arena back into the staging
buffers. On ratio-4 layers it also reloads indexer history, recomputes
indexer scores/top-k from the loaded rows, and broadcasts top-k indices.

## Validation

Build on `llm/llamacpp-build-8gpu`:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS.

Multi-step token-major validation:

```text
slots=32
ctx=262144
position=262136
decode_steps=8
layers=43
typed raw/compressed/indexer/history gates enabled
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

The final ratio-4 layer reports:

```text
layer 42 ratio 4 visible_attn_rows 2 loaded_attn_rows 2 loaded_indexer_rows 2 PASS
```

Artifacts:

- `logs/from-cluster/sprint336-typed-history-reload/cluster/tokenmajor8-typed-history-s32-pos262136.log`

## Next Step

Promote this typed KV history path from the full-layer smoke into the serving
decode path, then run tokenizer-enabled HTTP serving with typed KV gates and
resident session reuse.
