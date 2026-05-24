# TEMP Status Report 023

Date: 2026-05-24

## Current Topline

TP/EP serving is still operational but not model-correct. The active blocker
remains DS4 semantic parity inside the layer body, especially the true
attention/compressed-KV/indexer sequence.

Latest reference status remains:

```text
reference vector:      short_reasoning_plain
expected text:         16
actual text:           wrong token/text
status:                HTTP path operational, parity failed
```

## Sprint 311 Progress

Added the first executable true-attention projection prefix:

```text
binary flag:           --true-ds4-attention-projection-gate
launcher env:          DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
path:                  TP/EP only
PP work:               none
```

Under the gate, the TP/EP all-layer smoke now executes:

```text
current hidden -> attn_norm
current hidden -> attn_q_a -> attn_q_a_norm -> attn_q_b
current hidden -> attn_kv_latent -> attn_kv_a_norm
```

for all 43 layers at the target `32` slot / `256K` serving shape.

## V100 Evidence

Cluster log:

```text
logs/from-cluster/sprint311-attn-projection-prefix/run.log
```

Build result:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Gated 32-slot / 256K smoke:

```text
tp_ep_all_layer_dense_f16_cache:
  rows:         4096
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
  q_a_cols:   1024
  kv_cols:    512
  q_width:    32768
  result:     PASS

tp_ep_token_major_scaffold:
  steps:                1
  layers:               43
  pass_invocations:     43
  slots:                32
  ctx:                  262144
  projected_slot_tok/s: 36.575043
  sum_decode_ms:        874.913526
  sum_ep_ms:            547.589663
  sum_compose_ms:       25.305550
  sum_hc_current_ms:    246.866646
  sum_final_hc_ms:      24.056091
  wall_ms:              1607.086979
  checksum:             6325410718
  result:               PASS
```

## Interpretation

This is semantic-wiring progress, not throughput progress and not parity. It
proves the resident TP/EP process can execute the real attention prefix
through q/latent-kv projection and normalization for every layer.

The remaining attention work is still substantial:

```text
q_b output -> q head norm / RoPE
latent KV -> raw SWA KV write
latent KV -> compressed KV update
ratio-4 layers -> indexer row selection
raw + compressed scores -> softmax -> value read
attention heads -> inverse RoPE
attention output -> attn_output_a -> attn_output_b
```

Until that path feeds the next hidden state and the reference-vector gate
matches, generated text remains untrusted.
