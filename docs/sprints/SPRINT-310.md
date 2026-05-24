---
sprint: 310
title: TP/EP True-Attention Residency Gate
status: complete
started: 2026-05-24
branch: claude-takeover
---

# Sprint 310 - TP/EP True-Attention Residency Gate

## Goal

Move the TP/EP serving harness off the simplified attention bridge by making
the full DS4 attention projection tensor set resident in the TP/EP runtime.

## Scope

This sprint is TP/EP-only. It does not extend PP/layer-split code. The first
step is a residency and launch-contract gate, not a claim of model parity.

## Context

Sprint 309 localized the current parity blocker: the HTTP harness is
operational, model-router EP and true shared FFN are wired, but the layer body
still feeds simplified attention/HC state through `attn_output_b.weight`
instead of executing the DS4 attention sequence:

```text
attn_norm
  -> attn_q_a -> attn_q_a_norm -> attn_q_b -> q head norm/rope
  -> attn_kv_latent -> attn_kv_a_norm -> raw/compressed KV update
  -> raw + compressed/indexed attention
  -> inverse rope
  -> attn_output_a -> attn_output_b
```

The existing TP/EP dense cache already materializes all dense TP rows into
FP16/cuBLAS-admissible arenas, but the shared runtime only bound
`attn_output_b.weight`, `ffn_down_shexp.weight`, and optionally the shared FFN
gate/up tensors. That made true DS4 attention impossible inside the resident
serving loop.

## Implementation Plan

1. Add an opt-in TP/EP gate:
   `--true-ds4-attention-residency-gate`.
2. Under the gate, bind and keep resident for all 43 layers:
   - `attn_q_a.weight`
   - `attn_q_b.weight`
   - `attn_kv_latent.weight`
   - `attn_output_a.weight`
   - existing `attn_output_b.weight`
3. Expose the gate through the appliance launcher as:
   `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1`.
4. Verify that the expanded resident tensor set fits and builds on the V100
   pod.
5. Keep execution semantics unchanged until the next step wires the actual
   projection/KV/attention sequence. This avoids silently pretending the model
   is correct.

## Definition of Done

- The TP/EP binary accepts `--true-ds4-attention-residency-gate`. Complete.
- The launcher validates and forwards
  `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1`. Complete.
- The shared dense-op residency path loads the full DS4 attention projection
  tensor set for every layer. Complete.
- The default launcher path remains unchanged when the gate is off. Complete.
- V100 build passes. Complete.
- A gated V100 smoke reaches the shared dense-op residency report or fails at
  a concrete tensor/shape/memory boundary. Complete.

## V100 Evidence

Log:

```text
logs/from-cluster/sprint310-attn-residency/run.log
```

Build:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Gated residency smoke:

```text
tp_ep_all_layer_dense_f16_cache:
  rows:                4096
  source_bytes:        8640413696
  cache_bytes:         14451998720
  cache_ms:            8719.681475
  result:              PASS

tp_ep_all_layer_expert_bindings_shared:
  bytes:               147169738752
  bytes_per_gpu:       18396217344
  result:              PASS

tp_ep_all_layer_dense_ops_shared:
  loaded_bytes:        5725569024
  result:              PASS

tp_ep_token_major_scaffold:
  steps:               1
  layers:              43
  pass_invocations:    43
  slots:               32
  ctx:                 262144
  projected_slot_tok/s: 38.817036
  checksum:            8209335039
  result:              PASS
```

Observed V100 memory during the run peaked around `25.8 GiB` on the busiest
GPU, leaving roughly `6.6 GiB` free. That is enough headroom for this
residency step, but not proof that the full attention execution path is free.

## Outcome

The TP/EP runtime can now keep the real DS4 attention projection tensor set
resident alongside full expert bindings, dense cache, TP runtime, and 32-slot
/ 256K KV state. This removes the residency blocker for replacing the
simplified attention bridge.

The actual layer semantics are still not correct: the smoke does not yet run
q/kv/RoPE/raw-KV/compressed-KV/indexer/attention/output. It only proves that
the required tensors fit and can be bound in the resident TP/EP process.

## Next Sprint

Wire the resident attention tensors into the real TP/EP attention dataflow:
q/kv projections, RoPE, raw SWA cache update, compressed KV update, ratio-4
indexer selection, attention over raw plus compressed rows, inverse RoPE, and
grouped attention output.
