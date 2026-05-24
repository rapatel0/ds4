---
sprint: 311
title: TP/EP True-Attention Projection Prefix
status: complete
started: 2026-05-24
branch: claude-takeover
---

# Sprint 311 - TP/EP True-Attention Projection Prefix

## Goal

Move beyond attention tensor residency by executing the first real DS4
attention projection prefix inside the separate TP/EP runtime.

## Scope

This sprint is TP/EP-only. It does not modify or optimize PP/layer-split
paths. The implementation is explicitly diagnostic: it proves that the true
attention q/kv prefix can execute at the target serving shape, but it does not
yet feed the produced q/k/v data into the attention output path or claim model
parity.

## Context

Sprint 310 made the full DS4 attention projection tensor set resident:

```text
attn_q_a.weight
attn_q_b.weight
attn_kv_latent.weight
attn_output_a.weight
attn_output_b.weight
```

The next semantic gap is the reference DS4 attention sequence:

```text
attn_norm
  -> attn_q_a -> attn_q_a_norm -> attn_q_b
  -> attn_kv_latent -> attn_kv_a_norm
  -> q head norm / RoPE
  -> raw SWA KV update
  -> compressed KV update for ratio-4 / ratio-128 layers
  -> ratio-4 indexer path
  -> attention over raw plus compressed rows
  -> inverse RoPE
  -> attn_output_a -> attn_output_b
```

Sprint 311 implements the first executable prefix of that sequence through
`attn_q_b` and the normalized latent-KV vector.

## Implementation

Added a gated runtime path:

```text
binary flag:  --true-ds4-attention-projection-gate
launcher env: DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
```

The projection gate implies the existing true-attention residency and HC
current/final carry gates. Under the gate, each layer now runs:

```text
current_full
  -> attn_norm.weight RMSNorm
  -> attn_q_a dense op
  -> gather q_a TP shards to 1024-wide full q_a
  -> attn_q_a_norm.weight RMSNorm
  -> attn_q_b dense op

current_full
  -> attn_norm.weight RMSNorm
  -> attn_kv_latent dense op
  -> gather latent KV TP shards to 512-wide full KV
  -> attn_kv_a_norm.weight RMSNorm
```

New runtime buffers hold the normalized attention input, gathered `q_a`,
normalized `q_a`, gathered latent KV, and normalized latent KV. New control
bindings load the three attention norm vectors for all 43 layers:

```text
blk.N.attn_norm.weight
blk.N.attn_q_a_norm.weight
blk.N.attn_kv_a_norm.weight
```

## Definition of Done

- The TP/EP binary accepts `--true-ds4-attention-projection-gate`. Complete.
- The launcher validates and forwards
  `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1`. Complete.
- The projection gate implies true-attention residency and HC carry inputs.
  Complete.
- The gated path executes q_a, q_a_norm, q_b, latent KV, and latent-KV norm for
  every layer. Complete.
- V100 build passes. Complete.
- A `32` slot / `256K` V100 gate completes all 43 layers with finite output.
  Complete.
- The sprint documentation states that this is not yet parity or trusted
  generation. Complete.

## V100 Evidence

Log:

```text
logs/from-cluster/sprint311-attn-projection-prefix/run.log
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
  slots:      32
  q_a_cols:   1024
  kv_cols:    512
  result:     PASS

tp_ep_token_major_scaffold:
  steps:                1
  layers:               43
  pass_invocations:     43
  slots:                32
  ctx:                  262144
  sum_decode_ms:        874.913526
  projected_slot_tok/s: 36.575043
  sum_ep_ms:            547.589663
  sum_hc_current_ms:    246.866646
  sum_final_hc_ms:      24.056091
  wall_ms:              1607.086979
  checksum:             6325410718
  result:               PASS
```

The first layer's projection prefix took `39.078454 ms`, dominated by first-use
setup. Layers `1..42` were approximately `1.1-1.3 ms` each for the projection
prefix in this diagnostic mode.

## Outcome

The TP/EP runtime can now execute the first real DS4 attention projection
prefix across all 43 layers at the target `32` slot / `256K` shape. This
removes the next concrete blocker after residency: the tensors are not only
loaded, they can be invoked in the real ordering through `attn_q_b` and
latent-KV normalization.

This still is not model parity. The implementation does not yet run q-head
RMSNorm/RoPE, raw KV writes, compressed KV updates, ratio-4 indexer selection,
attention over raw plus compressed rows, inverse RoPE, or the
`attn_output_a -> attn_output_b` path using real attention heads. Generated
text remains untrusted until those semantics are wired and the reference
parity gate passes.

## Next Sprint

Continue the true attention sequence after this prefix:

```text
q_b output
  -> q head norm / RoPE
  -> raw SWA KV update
  -> latent KV compression update
  -> ratio-4 indexer row selection
  -> raw + compressed attention scores/softmax/value read
  -> inverse RoPE
  -> attn_output_a -> attn_output_b
```

The next sprint should produce an executable attention-state update gate, not
a PP variant and not another endpoint wrapper.
