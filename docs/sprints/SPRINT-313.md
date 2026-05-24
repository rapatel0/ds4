---
sprint: 313
title: TP/EP True-Attention Raw-SWA Read Gate
status: complete
started: 2026-05-24
branch: claude-takeover
---

# Sprint 313 - TP/EP True-Attention Raw-SWA Read Gate

## Goal

Extend the TP/EP true-attention diagnostic from q/KV state materialization into
the first narrow attention read: `attn_sinks` plus a one-row raw-SWA
score/softmax/value read.

## Scope

This sprint is TP/EP-only. It does not modify PP/layer-split paths. The new
gate remains diagnostic and does not claim reference parity or trusted text
generation.

## Context

Sprint 312 proved that all 43 layers can execute the attention projection
prefix and raw-SWA state update:

```text
attn_norm
  -> attn_q_a -> attn_q_a_norm -> attn_q_b
  -> attn_kv_latent -> attn_kv_a_norm
  -> local q-head RMSNorm
  -> raw SWA KV row
```

Sprint 313 adds the next executable piece:

```text
local q-head shard + raw SWA row + attn_sinks
  -> raw-SWA score
  -> sink-aware softmax over one raw row
  -> per-head value read
```

This is intentionally a one-row diagnostic read. It proves resident bindings,
sharded sink handling, and finite attention-read execution before implementing
full raw-window plus compressed-KV attention.

## Implementation

Added a gated TP/EP runtime path:

```text
binary flag:  --true-ds4-attention-raw-read-gate
launcher env: DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ=1
```

The raw-read gate implies:

```text
--true-ds4-attention-state-gate
--true-ds4-attention-projection-gate
--true-ds4-attention-residency-gate
--tp-hc-current-input-gate
--tp-hc-final-expand-gate
```

The gate adds:

```text
SharedHcControls::d_attn_sinks[layer] [64]
RankState::d_attn_sinks               [8]
RankState::d_attn_heads               [slots x 8 x 512]
```

For each layer and TP rank, the gated path now:

1. Loads source `blk.N.attn_sinks`.
2. Copies the rank-local eight sink values onto the owning GPU.
3. Scores each local head against the diagnostic raw-SWA row
   `position % 128`.
4. Applies a two-item softmax between that raw row and the attention sink.
5. Emits a per-head diagnostic value read into `d_attn_heads`.

## Definition of Done

- The TP/EP binary accepts `--true-ds4-attention-raw-read-gate`. Complete.
- The launcher validates and forwards
  `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ=1`. Complete.
- The raw-read gate implies state, projection, residency, and HC carry gates.
  Complete.
- The runtime loads `blk.N.attn_sinks` for all 43 layers. Complete.
- The gate executes a sink-aware raw-SWA attention read for all local heads on
  all TP ranks. Complete.
- V100 build passes. Complete.
- A `32` slot / `256K` V100 gate completes all 43 layers. Complete.

## V100 Evidence

Log:

```text
logs/from-cluster/sprint313-attn-raw-read/build.log
logs/from-cluster/sprint313-attn-raw-read/run.log
```

Gated smoke:

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

FAIL lines: 0
```

Token-major scaffold:

```text
steps:                1
layers:               43
pass_invocations:     43
slots:                32
ctx:                  262144
sum_decode_ms:        929.145120
projected_slot_tok/s: 34.440261
sum_ep_ms:            547.944916
sum_hc_current_ms:    302.429221
sum_final_hc_ms:      23.423683
wall_ms:              1668.744838
result:               PASS
```

Layer-level raw-read time was typically about `0.10-0.12 ms` in late layers.

## Caveat

This sprint proves executable raw-SWA read plumbing, not correct attention
numerics. The same saturation observed in Sprint 312 remains visible:

```text
layer 0 true_attn_raw_read_heads max_abs: 4608-6656
layer 1 true_attn_raw_read_heads max_abs: 65504
layer 2 true_attn_raw_read_heads max_abs: 65504
```

The next correctness blocker is still the reference-faithful attention
contract: q-head RoPE/inverse-RoPE, full raw-window selection, compressed-KV
selection, ratio-4 indexer behavior, and `attn_output_a -> attn_output_b`
composition into the next hidden state.

## Outcome

The TP/EP runtime now has an executable `attn_sinks` plus raw-SWA attention
read gate at the target `32` slot / `256K` shape. The generated text remains
untrusted until the full attention path is reference-equivalent and parity
passes.

## Next Sprint

Continue the attention replacement with:

```text
q-head RoPE
full raw-window score/read instead of one-row diagnostic read
compressed-KV row selection
ratio-4 indexer row selection
raw + compressed softmax/value read
attn_output_a -> attn_output_b
```

Before promoting attention output into hidden state, isolate the early-layer
saturation with a layer-0 reference microbench or stricter current-hidden
input parity check.
