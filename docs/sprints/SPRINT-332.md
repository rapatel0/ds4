---
sprint: 332
title: TP/EP Typed Raw-SWA KV Integration
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 332 - TP/EP Typed Raw-SWA KV Integration

## Goal

Wire the TP runtime's production typed row-sharded KV store/load APIs into the
full-layer DS4 attention state path for raw-SWA rows.

## Why This Sprint

Sprints 330-331 proved typed F8 KV row addressing and device-to-device row
roundtrips in focused runtime smokes. The full-layer attention harness still
wrote raw-SWA rows only into duplicated f32 diagnostic buffers. This sprint
moves the first real attention row through the same typed TP-sharded KV arena
that production serving must use.

## Scope

TP/EP only. No PP/layer-split work. No MTP. This sprint integrates raw-SWA
attention rows only; compressed attention rows and ratio-4 indexer rows remain
next.

## Definition of Done

- [x] Add an opt-in full-layer flag for typed raw-SWA KV.
- [x] Pass the shared/local TP runtime into the decode loop and attention state
      update path.
- [x] Store each slot's raw-SWA row through
      `ds4_v100_tp_runtime_kv_row_store_f32_device`.
- [x] Load the row back through
      `ds4_v100_tp_runtime_kv_row_load_f32_device` into the existing attention
      read staging buffer.
- [x] Validate `32` slots / `256K` on the V100 pod with all 43 layers.
- [x] Record cluster artifacts, status, and vision updates.

## Outcome

Added `--true-ds4-attention-typed-kv-raw-gate` to
`tools/ds4-v100-tp-ep-full-layer-smoke.cu`.

When the gate is enabled, the attention state update now:

1. materializes and RoPE-adjusts the per-slot KV row as before,
2. synchronizes the producing streams,
3. writes all active slots through the TP runtime's F8 E4M3 block-128 physical
   KV shards,
4. synchronizes the stores,
5. decodes the row back into the per-rank raw-SWA staging buffer, and
6. lets the existing raw-read/window attention gate consume that staging buffer.

The f32 raw-SWA buffer remains as a read staging surface for this sprint. The
source of the row is now the production typed arena, not the old direct f32
diagnostic writer.

## Validation

Build on `llm/llamacpp-build-8gpu`:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS.

Full shared-state validation:

```text
slots=32
ctx=262144
position=262140
layers=43
gate=--true-ds4-attention-typed-kv-raw-gate
raw_window=--true-ds4-attention-raw-window-gate
```

Result:

```text
tp_ep_true_attention_typed_kv_raw lines: 43
tp_ep_all_layer_scaffold ... pass_layers 43 ... PASS
projected_slot_step_tok_s 93.259761
sum_decode_ms_per_token 343.127622
```

The typed raw-SWA row view reported:

```text
logical_cols 512
logical_row_bytes 516
row_bytes_per_gpu 65
```

An initial single-layer run failed because that path did not initialize shared
HC controls; the all-layer shared-state run is the valid gate for this sprint.

Artifacts:

- `logs/from-cluster/sprint332-typed-raw-swa/cluster/alllayers-typed-raw-window-s32-pos262140.log`
- `logs/from-cluster/sprint332-typed-raw-swa/cluster/layer2-typed-raw-window-s32-pos262140.log`

## Next Step

Extend the same typed-KV integration to compressed attention rows and ratio-4
indexer rows, then replace the bounded diagnostic compressed row storage in
the full-layer path.
