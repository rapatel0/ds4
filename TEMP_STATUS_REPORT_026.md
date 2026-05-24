# TEMP Status Report 026

Date: 2026-05-24

## Current Topline

TP/EP serving is still operational but not model-correct. Sprint 314 added a
separate true-attention raw-window diagnostic gate. It passes structurally
across all 43 layers and four token-major steps at the target `32` slot /
`256K` shape.

Reference parity is still failing:

```text
reference vector:      short_reasoning_plain
expected text:         16
actual text:           wrong token/text
status:                HTTP path operational, parity failed
```

## Sprint 314 Progress

Added the raw-window attention-read gate:

```text
binary flag:           --true-ds4-attention-raw-window-gate
launcher env:          DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW=1
path:                  TP/EP only
PP work:               none
```

The gate extends Sprint 313's one-row read with a resident sliding-window
diagnostic:

```text
valid rows populated by token-major steps + attn_sinks + q-head shard
  -> stable softmax over raw rows plus sink
  -> per-head diagnostic value read
```

## V100 Evidence

Cluster logs:

```text
logs/from-cluster/sprint314-attn-raw-window/build.log
logs/from-cluster/sprint314-attn-raw-window/run.log
```

Gated `32` slot / `256K` / `4` step smoke:

```text
tp_ep_true_attention_projection_prefix:
  pass_lines: 172

tp_ep_true_attention_state_update:
  pass_lines: 172

tp_ep_true_attention_raw_window:
  pass_lines: 172

valid_rows:
  1: 43
  2: 43
  3: 43
  4: 43

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
  projected_slot_tok/s: 35.533525
  sum_decode_ms:        3602.231959
  wall_ms:              6489.082791
  result:               PASS
```

## Important Caveat

This is still an attention plumbing milestone, not correctness. Early-layer
raw-window outputs remain finite but saturated:

```text
true_attn_raw_window_heads max_abs: 65504 in early layers
finite_bad: 0
```

The next correctness gap is reference-faithful attention semantics:
RoPE/inverse-RoPE, compressed-KV row selection/read, ratio-4 indexer behavior,
raw+compressed score merge, and `attn_output_a -> attn_output_b`.

## Next

Next TP/EP-only work should add compressed-KV/indexer diagnostics or a
reference microbench that isolates the raw-KV saturation before attention
output is allowed to drive hidden state.
