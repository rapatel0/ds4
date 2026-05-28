# TEMP Status Report 024

Date: 2026-05-24

## Current Topline

TP/EP serving is still operational but not model-correct. We are now replacing
the simplified attention bridge with executable DS4 attention pieces.

Reference parity is still failing:

```text
reference vector:      short_reasoning_plain
expected text:         16
actual text:           wrong token/text
status:                HTTP path operational, parity failed
```

## Sprint 312 Progress

Added the first attention-state update gate:

```text
binary flag:           --true-ds4-attention-state-gate
launcher env:          DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1
path:                  TP/EP only
PP work:               none
```

The gate extends Sprint 311's projection prefix with:

```text
attn_q_b shard -> local q-head RMSNorm
attn_kv_a_norm -> per-rank raw SWA KV row
```

Each rank owns `8` local heads of `512` values and a diagnostic raw SWA cache:

```text
d_attn_raw_swa [slots x 128 x 512]
```

## V100 Evidence

Cluster log:

```text
logs/from-cluster/sprint312-attn-state-update/run.log
```

Build result:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Gated `32` slot / `256K` smoke:

```text
tp_ep_true_attention_projection_prefix:
  pass_lines: 43
  fail_lines: 0

tp_ep_true_attention_state_update:
  pass_lines: 43
  fail_lines: 0
  local_heads: 8
  head_dim:    512
  raw_rows:    128
  raw_row:     32

tp_ep_token_major_scaffold:
  steps:                1
  layers:               43
  pass_invocations:     43
  slots:                32
  ctx:                  262144
  projected_slot_tok/s: 30.427365
  sum_decode_ms:        1051.684894
  wall_ms:              1786.317692
  result:               PASS
```

## Important Caveat

The state gate is finite, but raw KV saturates in the diagnostic path:

```text
true_attn_q_heads_normed_shard finite_bad=0
true_attn_raw_swa_rank0        finite_bad=0 max_abs=65504
```

This is useful evidence. It says the next blocker is no longer "can we invoke
the q/kv projection and state kernels?" It is whether the upstream HC/current
hidden bridge and KV quantization contract are reference-faithful enough for
attention softmax.

## Next

Next TP/EP-only work should add a narrow attention-read gate:

```text
q-head RoPE
attn_sinks
raw SWA score/softmax/value read
attn_output_a -> attn_output_b
```

Before feeding that output into the hidden state, isolate the raw-KV
saturation with a layer-0 reference microbench or a stricter current-hidden
input parity check.
