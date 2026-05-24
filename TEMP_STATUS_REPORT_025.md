# TEMP Status Report 025

Date: 2026-05-24

## Current Topline

TP/EP serving is still operational but not model-correct. Sprint 313 added a
new true-attention diagnostic gate for `attn_sinks` plus raw-SWA attention
read, and it passes structurally across all 43 layers at the target
`32` slot / `256K` shape.

Reference parity is still failing:

```text
reference vector:      short_reasoning_plain
expected text:         16
actual text:           wrong token/text
status:                HTTP path operational, parity failed
```

## Sprint 313 Progress

Added the raw-SWA attention-read gate:

```text
binary flag:           --true-ds4-attention-raw-read-gate
launcher env:          DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ=1
path:                  TP/EP only
PP work:               none
```

The gate extends Sprint 312's state update with:

```text
attn_sinks + local q-head shard + raw SWA row
  -> sink-aware one-row softmax
  -> per-head diagnostic value read
```

Each rank owns:

```text
d_attn_sinks [8]
d_attn_heads [slots x 8 x 512]
```

## V100 Evidence

Cluster log:

```text
logs/from-cluster/sprint313-attn-raw-read/run.log
```

Gated `32` slot / `256K` smoke:

```text
tp_ep_true_attention_projection_prefix:
  pass_lines: 43
  fail_lines: 0

tp_ep_true_attention_state_update:
  pass_lines: 43
  fail_lines: 0

tp_ep_true_attention_raw_read:
  pass_lines: 43
  fail_lines: 0

tp_ep_token_major_scaffold:
  steps:                1
  layers:               43
  pass_invocations:     43
  slots:                32
  ctx:                  262144
  projected_slot_tok/s: 34.440261
  sum_decode_ms:        929.145120
  wall_ms:              1668.744838
  result:               PASS
```

## Important Caveat

The raw-read path is finite, but it inherits the raw-KV saturation from Sprint
312:

```text
layer 0 true_attn_raw_read_heads max_abs: 4608-6656
layer 1 true_attn_raw_read_heads max_abs: 65504
layer 2 true_attn_raw_read_heads max_abs: 65504
```

This is useful evidence but not model correctness. It narrows the remaining
attention gap to reference semantics: q-head RoPE/inverse-RoPE, full raw-window
selection, compressed-KV rows, ratio-4 indexer behavior, and attention output
projection/composition.

## Next

Next TP/EP-only work should move from one-row raw-SWA read to the full DS4
attention read:

```text
q-head RoPE
raw SWA window score/read
compressed-KV score/read
ratio-4 indexer selection
raw + compressed softmax/value read
attn_output_a -> attn_output_b
```

Do not promote this output into hidden state until the early-layer saturation
is isolated against a reference microbench or stricter current-hidden parity
check.
