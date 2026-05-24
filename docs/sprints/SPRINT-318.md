---
sprint: 318
title: TP/EP Block Reduction Broadcast Fix
status: complete
started: 2026-05-24
branch: claude-takeover
---

# Sprint 318 - TP/EP Block Reduction Broadcast Fix

## Goal

Fix the TP/EP `block_sum_256_f32` and `block_max_256_f32` helpers so every
thread receives the block-wide reduction result.

## Scope

This sprint is TP/EP-only. It fixes the helper used by the separate TP/EP
smoke/runtime tool and reruns the attention correctness diagnostics. It does
not modify PP/layer-split paths.

## Why This Sprint

Sprint 317 proved the current helper returns the final reduction only to the
first warp. Threads `32..255` receive zero and normalize with the wrong scale.
That directly explains the huge KV norm drift and likely contaminates other
TP/EP kernels using the same helper.

## Definition of Done

- `block_sum_256_f32` broadcasts the final sum to all threads.
  Complete.
- `block_max_256_f32` broadcasts the final max to all threads.
  Complete.
- V100 build passes.
  Complete.
- The KV norm reference gate rerun shows max-abs drift near numerical noise,
  not thousands.
  Complete.
- The saturation audit rerun no longer shows artificial `kv_normed` explosion
  from the reduction bug.
  Complete.
- The raw-window gate still completes `32` slots / `256K` / `4` steps without
  `FAIL`.
  Complete.
- Sprint doc, temp status report, and vision are updated with the before/after
  evidence.
  Complete.

## V100 Evidence

Logs:

```text
logs/from-cluster/sprint318-block-reduction-fix/build.log
logs/from-cluster/sprint318-block-reduction-fix/run.log
```

Build:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Combined `32` slot / `256K` / `4` step smoke:

```text
tp_ep_true_attention_kv_norm_reference:      172
tp_ep_true_attention_saturation_projection:  172
tp_ep_true_attention_saturation_state:       172
tp_ep_true_attention_projection_prefix:      172
tp_ep_true_attention_rope:                   172
tp_ep_true_attention_state_update:           172
tp_ep_true_attention_raw_window:             172
FAIL lines:                                    0
```

Token-major scaffold:

```text
steps:                4
layers:               43
pass_invocations:     172
slots:                32
ctx:                  262144
sum_decode_ms:        5173.297493
ms_per_token:         1293.324373
projected_slot_tok/s: 24.742439
sum_ep_ms:            2086.853737
sum_hc_current_ms:    2807.257360
sum_final_hc_ms:      92.483992
wall_ms:              7988.229452
result:               PASS
```

## Before / After

Before the fix, Sprint 317 showed:

```text
max_abs_diff:       847034.125
max_rel_diff:       196228.344
stable_max:         1229665.75
raw_swa_row_max:    65504
```

After the fix:

```text
max_abs_diff:       9.53674316e-07
max_rel_diff:       4.02997415e-07
kv_in_max:          1.37322819
kv_normed_max:      6.29025173
raw_swa_row_max:    6.28515625
q_b_pre_head_max:   1.75204742
q_heads_rope_max:   16.0296879
```

The prior `65504` raw-SWA saturation was an artifact of the TP/EP block
reduction helper. It is gone in this gate.

## Outcome

The TP/EP reduction helper is now safe for RMSNorm-style kernels in this tool.
The KV norm reference comparison is back to numerical-noise drift, and the
true-attention raw-window diagnostic remains structurally green at the target
`32` slot / `256K` shape.

## Next Sprint

Rerun the reference parity gate and then continue true DS4 attention semantics:

```text
raw + compressed KV merge
ratio-4 indexer selection
attn_output_a -> attn_output_b
hidden-state promotion
```

The immediate next decision is whether the corrected attention-prefix path is
close enough to feed attention output into hidden state, or whether compressed
KV/indexer rows must land first.
