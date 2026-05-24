# TEMP Status Report 022

Date: 2026-05-24

## Current Topline

TP/EP serving is still operational but not model-correct. The active blocker
remains DS4 semantic parity, specifically the simplified attention/HC bridge.

Latest correctness status remains:

```text
reference vector:      short_reasoning_plain
expected text:         16
actual text:           wrong token/text
status:                HTTP path operational, parity failed
```

## Sprint 310 Progress

Added the first concrete step toward real DS4 attention in the TP/EP runtime:

```text
binary flag:           --true-ds4-attention-residency-gate
launcher env:          DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
path:                  TP/EP only
PP work:               none
```

When enabled, the shared dense-op residency path now binds the full DS4
attention projection tensor set for all 43 layers:

```text
attn_q_a.weight
attn_q_b.weight
attn_kv_latent.weight
attn_output_a.weight
attn_output_b.weight
```

Before this, the TP/EP serving harness had only `attn_output_b.weight`
resident and therefore could not execute the reference DS4 attention sequence.

## V100 Evidence

Cluster log:

```text
logs/from-cluster/sprint310-attn-residency/run.log
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

tp_ep_all_layer_expert_bindings_shared:
  bytes:         147169738752
  bytes_per_gpu: 18396217344
  result:        PASS

tp_ep_all_layer_dense_ops_shared:
  loaded_bytes: 5725569024
  result:       PASS

tp_ep_token_major_scaffold:
  steps:            1
  layers:           43
  pass_invocations: 43
  slots:            32
  ctx:              262144
  result:           PASS
```

Observed memory during the run peaked around `25.8 GiB` on the busiest V100,
leaving about `6.6 GiB` free.

## Interpretation

This is residency progress, not parity. It proves the resident TP/EP process
can fit and bind the tensor set needed for true DS4 attention at the target
`32` slot / `256K` shape.

The next implementation step is to wire these resident tensors into the actual
DS4 attention dataflow:

```text
attn_norm
  -> attn_q_a -> attn_q_a_norm -> attn_q_b -> q head norm/RoPE
  -> attn_kv_latent -> attn_kv_a_norm -> raw KV update
  -> compressed KV update for ratio-4 / ratio-128 layers
  -> ratio-4 indexer path
  -> attention over raw plus compressed rows
  -> inverse RoPE
  -> attn_output_a -> attn_output_b
```

Until that lands, generated text remains untrusted.
