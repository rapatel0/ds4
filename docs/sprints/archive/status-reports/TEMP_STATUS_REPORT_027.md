# TEMP Status Report 027

Date: 2026-05-24

## Current Topline

TP/EP serving remains operational but not model-correct. Sprint 315 added the
true-attention RoPE gate, so q-head shards and latent KV rows are now rotated
before the raw-SWA diagnostic read path.

Reference parity is still failing:

```text
reference vector:      short_reasoning_plain
expected text:         16
actual text:           wrong token/text
status:                HTTP path operational, parity failed
```

## Sprint 315 Progress

Added the RoPE attention-semantics gate:

```text
binary flag:           --true-ds4-attention-rope-gate
launcher env:          DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=1
path:                  TP/EP only
PP work:               none
```

The gate applies DS4-style tail RoPE to both:

```text
q-head shards after q_b RMSNorm
latent KV rows before FP8-ish raw-SWA storage
```

`--true-ds4-attention-raw-window-gate` now implies the RoPE gate.

## V100 Evidence

Cluster logs:

```text
logs/from-cluster/sprint315-attn-rope/build.log
logs/from-cluster/sprint315-attn-rope/run.log
```

Gated `32` slot / `256K` / `4` step smoke:

```text
tp_ep_true_attention_projection_prefix:
  pass_lines: 172

tp_ep_true_attention_rope:
  pass_lines: 172

tp_ep_true_attention_state_update:
  pass_lines: 172

tp_ep_decode_loop:
  pass_lines: 172

tp_ep_token_major_item:
  pass_lines: 172

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
  projected_slot_tok/s: 35.665666
  sum_decode_ms:        3588.885770
  wall_ms:              6441.026507
  result:               PASS
```

## Log Caveat

The raw-window diagnostic output has one stdout interleaving artifact:

```text
anchored tp_ep_true_attention_raw_window lines: 171
valid_rows:
  1: 43
  2: 42
  3: 42
  4: 43
```

The malformed line is layer `3`, where a `tp_ep_true_attention_raw_window`
print was overwritten by a concurrent tensor-stats print. The scaffold reports
`172` pass invocations and there are zero `FAIL` lines, so the run is treated
as structurally passing with a logging caveat.

## Important Caveat

RoPE did not solve the numerical correctness issue. Early-layer KV/raw-window
values remain finite but saturate:

```text
true_attn_raw_window_heads max_abs: 65504 in early layers
finite_bad: 0
```

The next TP/EP-only correctness gap is to isolate the activation scale and
quantization contract around the true-attention projection/state path before
implementing compressed-KV/indexer reads or feeding attention output into the
next hidden state.

## Next

Run a focused true-attention saturation sprint:

```text
HC/current hidden input
  -> attn_norm
  -> attn_q_a / attn_kv_latent
  -> q_a_norm / kv_a_norm
  -> q_b + RoPE
  -> KV RoPE + raw-SWA store
```

The diagnostic should compare input/output magnitudes per substage and test
whether reference scaling, explicit clipping, or the raw-KV FP8-ish storage
contract is the cause of the `65504` saturation.
