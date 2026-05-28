# TEMP Status Report 030

Date: 2026-05-24

## Current Topline

Sprint 318 fixed the TP/EP block-reduction broadcast bug. This materially
changes the true-attention diagnostics:

```text
KV norm reference drift: thousands -> ~1e-6
raw-SWA max:             65504 -> ~6.29
true-attention gates:    172/172, zero FAIL
```

TP/EP serving is still not declared model-correct, but the artificial
attention-prefix saturation blocker is removed.

## Change

Fixed the shared helpers:

```text
block_sum_256_f32
block_max_256_f32
```

Before the fix, the final reduction was returned only to the first warp.
Threads `32..255` used the wrong reduction value. After the fix, thread `0`
writes the final reduction to shared memory and all threads read the same
block-wide value after `__syncthreads()`.

## V100 Evidence

Cluster logs:

```text
logs/from-cluster/sprint318-block-reduction-fix/build.log
logs/from-cluster/sprint318-block-reduction-fix/run.log
```

Combined `32` slot / `256K` / `4` step gate:

```text
tp_ep_true_attention_kv_norm_reference:      172
tp_ep_true_attention_saturation_projection:  172
tp_ep_true_attention_saturation_state:       172
tp_ep_true_attention_projection_prefix:      172
tp_ep_true_attention_rope:                   172
tp_ep_true_attention_state_update:           172
tp_ep_true_attention_raw_window:             172

FAIL lines: 0
```

Final scaffold:

```text
tp_ep_token_major_scaffold:
  steps:                4
  layers:               43
  pass_invocations:     172
  slots:                32
  ctx:                  262144
  projected_slot_tok/s: 24.742439
  sum_decode_ms:        5173.297493
  wall_ms:              7988.229452
  result:               PASS
```

## Before / After

Before:

```text
max_abs_diff:       847034.125
max_rel_diff:       196228.344
stable_max:         1229665.75
raw_swa_row_max:    65504
```

After:

```text
max_abs_diff:       9.53674316e-07
max_rel_diff:       4.02997415e-07
kv_in_max:          1.37322819
kv_normed_max:      6.29025173
raw_swa_row_max:    6.28515625
q_b_pre_head_max:   1.75204742
q_heads_rope_max:   16.0296879
```

## Next

Rerun reference parity with the corrected helper, then continue the remaining
true-attention semantics:

```text
compressed KV/indexer row selection
raw + compressed attention score merge
attn_output_a -> attn_output_b
hidden-state promotion
```
