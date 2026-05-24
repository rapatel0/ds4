---
sprint: 315
title: TP/EP True-Attention Q/KV RoPE Gate
status: complete
started: 2026-05-24
branch: claude-takeover
---

# Sprint 315 - TP/EP True-Attention Q/KV RoPE Gate

## Goal

Close the next explicit DS4 attention-semantics gap by applying the reference
RoPE tail transform to q-head shards and the latent KV row before raw-SWA
state/update/read diagnostics.

## Scope

This sprint is TP/EP-only. It does not modify PP/layer-split paths and does
not yet implement compressed-KV/indexer rows or `attn_output_a -> attn_output_b`.

## Why This Sprint

The reference DS4 decode path does:

```text
q projection -> q-head RoPE
kv projection -> kv RoPE -> FP8-style KV quantization -> raw cache
```

The current TP/EP diagnostic path does:

```text
q projection -> q-head RMSNorm
kv projection -> FP8-style KV quantization -> raw cache
```

That means Sprint 314's raw-window read is still reading unrotated q/KV rows.
Adding RoPE is a model-semantics step toward parity and a required precursor
before compressed-KV/indexer attention can be trusted.

## Definition of Done

- Add `--true-ds4-attention-rope-gate`.
  Complete.
- Add launcher env `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=1`.
  Complete.
- The RoPE gate implies state/projection/residency and HC carry gates.
  Complete.
- q-head shards use the same tail-only YaRN RoPE constants as `ds4.c`.
  Complete.
- KV rows use the same RoPE transform before quantize/round raw-SWA storage.
  Complete.
- A V100 `32` slot / `256K` / `4` step gate completes all 43 layers.
  Complete.
- The V100 log shows RoPE pass lines and raw-window pass lines with
  `valid_rows` reaching `4`.
  Complete, with one stdout interleaving caveat noted below.
- The run has zero `FAIL` lines.
  Complete.
- Outcome and the remaining attention gap are recorded in this sprint doc,
  the temp status report, and `docs/sprints/VISION.md`.
  Complete.

## V100 Evidence

Logs:

```text
logs/from-cluster/sprint315-attn-rope/build.log
logs/from-cluster/sprint315-attn-rope/run.log
```

Build:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Gated `32` slot / `256K` / `4` step smoke:

```text
tp_ep_true_attention_projection_prefix: 172
tp_ep_true_attention_rope:              172
tp_ep_true_attention_state_update:      172
tp_ep_decode_loop:                      172
tp_ep_token_major_item:                 172
FAIL lines:                               0
```

Raw-window diagnostics:

```text
tp_ep_true_attention_raw_window anchored lines: 171
valid_rows:
  1: 43
  2: 42
  3: 42
  4: 43
```

The run has one stdout-interleaved diagnostic line at layer `3` where a
`tp_ep_true_attention_raw_window` print was overwritten by a concurrent tensor
stats print. The scaffold still reports `172` pass invocations and `0` FAIL
lines, so this is recorded as a logging artifact, not a runtime failure.

Token-major scaffold:

```text
steps:                4
layers:               43
pass_invocations:     172
slots:                32
ctx:                  262144
sum_decode_ms:        3588.885770
ms_per_token:         897.221443
projected_slot_tok/s: 35.665666
sum_ep_ms:            2172.568517
sum_hc_current_ms:    1097.965692
sum_final_hc_ms:      95.134756
wall_ms:              6441.026507
result:               PASS
```

## Remaining Risks

- RoPE is magnitude-preserving, so the existing early-layer saturation may
  remain. If so, the next sprint must isolate the HC/current-hidden bridge or
  KV quantization contract.
- Compressed-KV/indexer and inverse-RoPE/output projection remain missing
  after this sprint.

## Outcome

The TP/EP runtime now applies DS4-style tail RoPE to q-head shards and the
latent KV row before raw-SWA quantized storage. The raw-window diagnostic still
executes across the resident token-major path at the target `32` slot /
`256K` shape, and RoPE pass lines appear for all 43 layers across all four
steps.

This is a required model-semantics step, but it does not make generation
trusted yet. The raw-SWA values remain finite but still saturate to `65504`
in early layers, so the next TP/EP-only sprint should isolate the activation
scale/quantization contract before allowing attention output to drive the next
hidden state.

## Next Sprint

Add a focused true-attention saturation microbench or clamp/scale diagnostic
around the HC-current -> attention projection -> KV quantization path:

```text
current hidden shard
  -> attn_norm
  -> attn_q_a / attn_kv_latent
  -> q_a_norm / kv_a_norm
  -> q_b / RoPE
  -> KV RoPE
  -> raw-SWA quantized store
```

The goal is to identify whether saturation is caused by the incoming HC bridge,
the dense projection path, missing reference scaling, or the FP8-ish raw-KV
store contract.
