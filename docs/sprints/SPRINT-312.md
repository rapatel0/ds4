---
sprint: 312
title: TP/EP True-Attention State Update Gate
status: complete
started: 2026-05-24
branch: claude-takeover
---

# Sprint 312 - TP/EP True-Attention State Update Gate

## Goal

Extend the TP/EP true-attention diagnostic from projection-only execution into
the first attention-state update: q-head normalization plus raw SWA KV row
materialization.

## Scope

This sprint is TP/EP-only. It does not modify PP/layer-split paths. The new
gate remains diagnostic and does not claim DS4 parity or trusted text
generation.

## Context

Sprint 311 proved that all 43 layers can execute the projection prefix:

```text
attn_norm
  -> attn_q_a -> attn_q_a_norm -> attn_q_b
  -> attn_kv_latent -> attn_kv_a_norm
```

The next reference step is to turn those projection outputs into attention
state:

```text
q_b output -> per-head RMSNorm / RoPE
kv_normed  -> quantized/rounded raw SWA KV row
```

Sprint 312 implements the RMSNorm and raw-KV update part. RoPE and attention
read are intentionally left for the next sprint.

## Implementation

Added a gated TP/EP runtime path:

```text
binary flag:  --true-ds4-attention-state-gate
launcher env: DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1
```

The state gate implies:

```text
--true-ds4-attention-projection-gate
--true-ds4-attention-residency-gate
--tp-hc-current-input-gate
--tp-hc-final-expand-gate
```

The gate adds per-rank resident buffers for:

```text
d_attn_kv_full   [slots x 512]
d_attn_raw_swa   [slots x 128 x 512]
```

For each layer and rank, the gated path now:

1. Normalizes the local `attn_q_b` shard as `8` local heads of `512` values.
2. Broadcasts the normalized latent KV row from device `0` to each rank.
3. Applies a DS4-style FP8 quantize/dequantize pass over the non-rotary KV
   dimensions, then FP16 rounding/saturation.
4. Stores the result into the diagnostic raw SWA row at
   `position % 128`.

## Definition of Done

- The TP/EP binary accepts `--true-ds4-attention-state-gate`. Complete.
- The launcher validates and forwards
  `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1`. Complete.
- The state gate implies projection and residency gates. Complete.
- The state gate runs q-head RMSNorm for all `64` heads across TP8 shards.
  Complete.
- The state gate writes a raw SWA KV row for all active slots. Complete.
- V100 build passes. Complete.
- A `32` slot / `256K` V100 gate completes all 43 layers. Complete.

## V100 Evidence

Log:

```text
logs/from-cluster/sprint312-attn-state-update/run.log
```

Build:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Gated smoke:

```text
tp_ep_all_layer_dense_f16_cache:
  rows:        4096
  cache_bytes: 14451998720
  result:      PASS

tp_ep_all_layer_dense_ops_shared:
  layers:       43
  devices:      8
  loaded_bytes: 5725569024
  result:       PASS

tp_ep_true_attention_projection_prefix:
  pass_lines: 43
  fail_lines: 0
  result:     PASS

tp_ep_true_attention_state_update:
  pass_lines: 43
  fail_lines: 0
  slots:      32
  local_heads: 8
  head_dim:   512
  raw_rows:   128
  raw_row:    32
  result:     PASS

tp_ep_token_major_scaffold:
  steps:                1
  layers:               43
  pass_invocations:     43
  slots:                32
  ctx:                  262144
  sum_decode_ms:        1051.684894
  projected_slot_tok/s: 30.427365
  sum_ep_ms:            561.665486
  sum_hc_current_ms:    409.894041
  sum_final_hc_ms:      23.836192
  wall_ms:              1786.317692
  checksum:             1456469143
  result:               PASS
```

State-update latency for late layers was typically `0.14-0.19 ms`. Layer `0`
had first-use cost at `0.708396 ms`.

## Caveat

The q-head shards are finite after RMSNorm, but the diagnostic raw SWA KV
store reaches FP16 saturation in early layers:

```text
true_attn_q_heads_normed_shard finite_bad=0
true_attn_raw_swa_rank0        finite_bad=0 max_abs=65504
```

That means this sprint proves executable state materialization, not correct
attention numerics. The next parity work must decide whether this saturation
comes from the still-simplified upstream HC/current-hidden bridge, missing
RoPE/reference scaling, or an incorrect quantize/round contract.

## Outcome

The TP/EP runtime now has an executable q-head normalization and raw-SWA-KV
state update gate at the target `32` slot / `256K` shape. This is a necessary
primitive for replacing the simplified attention bridge, but the generated
text is still untrusted.

## Next Sprint

Continue the attention path with:

```text
q-head RoPE / inverse-RoPE contract
attention score path over raw SWA rows
attn_sinks handling
attention value read into per-head output
attn_output_a -> attn_output_b
```

Before promoting attention output into the next hidden state, the raw-KV
saturation should be isolated with a layer-0 reference microbench or a
reference-HC/current-hidden input gate.
