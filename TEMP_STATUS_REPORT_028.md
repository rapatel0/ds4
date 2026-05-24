# TEMP Status Report 028

Date: 2026-05-24

## Current Topline

TP/EP serving remains operational but not model-correct. Sprint 316 added a
true-attention saturation audit gate and ran it at the target `32` slot /
`256K` shape.

The important result:

```text
first saturated stage: kv_normed at layer 1
not first cause:        q-head RoPE
not first cause:        raw-SWA store
```

Reference parity is still failing:

```text
reference vector:      short_reasoning_plain
expected text:         16
actual text:           wrong token/text
status:                HTTP path operational, parity failed
```

## Sprint 316 Progress

Added the saturation audit gate:

```text
binary flag:           --true-ds4-attention-saturation-audit-gate
launcher env:          DS4_V100_TP_EP_TRUE_DS4_ATTENTION_SATURATION_AUDIT=1
path:                  TP/EP only
PP work:               none
```

The gate reports projection and state-stage max/finite stats around:

```text
current hidden full
attn_normed
attn_q_a / q_a_normed
attn_kv_latent / kv_normed
q_b pre-head-normalization
q heads after RMSNorm/RoPE
KV row after RoPE
raw-SWA stored row
```

## V100 Evidence

Cluster logs:

```text
logs/from-cluster/sprint316-attn-saturation-audit/build.log
logs/from-cluster/sprint316-attn-saturation-audit/run.log
```

Gated `32` slot / `256K` / `4` step smoke:

```text
tp_ep_true_attention_saturation_projection:
  anchored_rows: 172
  parse_clean:   170

tp_ep_true_attention_saturation_state:
  anchored_rows: 171
  parse_clean:   170

tp_ep_true_attention_projection_prefix: 172
tp_ep_true_attention_rope:              172
tp_ep_true_attention_state_update:      172
tp_ep_true_attention_raw_window:        172

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
  projected_slot_tok/s: 25.137208
  sum_decode_ms:        5092.053121
  wall_ms:              7873.373789
  result:               PASS
```

## Finding

Layer `0` is high but not saturated:

```text
kv_normed_max:       6510.59814
kv_post_rope_max:    6510.59814
raw_swa_row_max:     6656
```

Layer `1` saturates before raw-SWA storage:

```text
kv_normed_max:       436616.219
kv_post_rope_max:    434584.688
raw_swa_row_max:     65504
```

The q path normalizes back down:

```text
q_b_pre_head_max:      first >=65504 at layer 17
q_heads_post_rope_max: 15.8982925 max
```

So the immediate correctness target is the `attn_kv_latent ->
attn_kv_a_norm` contract or the upstream HC/current-hidden bridge, not q-head
RoPE.

## Log Caveat

Three audit rows were partially overwritten by existing tensor-stat diagnostics
in the combined `stdout`/`stderr` `tee` stream. The scaffold still reports
`172` pass invocations and there are zero `FAIL` lines.

## Next

Implement a focused `attn_kv_a_norm` reference-math comparison gate:

```text
attn_kv_latent output
  -> current TP/EP RMSNorm path
  -> DS4/reference RMSNorm/scaling path
  -> diff and max-abs report
```

Do this before compressed-KV/indexer reads or before feeding true-attention
output into the hidden state.
