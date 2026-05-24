---
sprint: 316
title: TP/EP True-Attention Saturation Audit Gate
status: complete
started: 2026-05-24
branch: claude-takeover
---

# Sprint 316 - TP/EP True-Attention Saturation Audit Gate

## Goal

Add a TP/EP-only diagnostic gate that identifies where the true-attention path
first reaches the `65504` saturation observed in Sprint 312 through Sprint 315.

## Scope

This sprint does not modify PP/layer-split code and does not change serving
defaults. It adds instrumentation around the current true-attention
projection/state path:

```text
current hidden full
  -> attn_norm
  -> attn_q_a / attn_kv_latent
  -> q_a_norm / kv_a_norm
  -> attn_q_b
  -> q-head RMSNorm + RoPE
  -> KV RoPE
  -> raw-SWA quantized store
```

## Why This Sprint

Sprint 315 proved RoPE plumbing, but raw-SWA values still saturate to `65504`
in early layers. Before adding compressed-KV/indexer reads or feeding attention
output into the hidden state, we need to know whether saturation enters from:

- the upstream HC/current-hidden bridge,
- attention norm/projection magnitude,
- q/KV RoPE scaling,
- or the FP8-ish raw-KV store contract.

## Definition of Done

- Add `--true-ds4-attention-saturation-audit-gate`.
  Complete.
- Add launcher env
  `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_SATURATION_AUDIT=1`.
  Complete.
- The audit gate implies the true-attention RoPE/state/projection/residency
  gates and HC carry gates.
  Complete.
- The V100 log emits projection-stage audit rows for all 43 layers.
  Complete, with two stdout/stderr interleaving caveats noted below.
- The V100 log emits state/KV-store audit rows for all 43 layers.
  Complete, with one stdout/stderr interleaving caveat noted below.
- A `32` slot / `256K` / `4` step V100 gate completes without `FAIL`.
  Complete.
- The sprint doc, temp status report, and vision identify the first saturated
  stage and the next implementation step.
  Complete.

## Expected Output

The audit rows should make the first saturated stage explicit:

```text
tp_ep_true_attention_saturation_projection ...
tp_ep_true_attention_saturation_state ...
```

Each row reports finite-bad counts and max-abs values for the relevant
intermediate tensors. This is intentionally diagnostic and may be slower than
serving mode because it copies summary inputs to host.

## V100 Evidence

Logs:

```text
logs/from-cluster/sprint316-attn-saturation-audit/build.log
logs/from-cluster/sprint316-attn-saturation-audit/run.log
```

Build:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Gated `32` slot / `256K` / `4` step smoke:

```text
tp_ep_true_attention_saturation_projection: 172 anchored rows, 170 parse-clean
tp_ep_true_attention_saturation_state:      171 anchored rows, 170 parse-clean
tp_ep_true_attention_projection_prefix:     172
tp_ep_true_attention_rope:                  172
tp_ep_true_attention_state_update:          172
tp_ep_true_attention_raw_window:            172
FAIL lines:                                   0
```

The audit rows have three stdout/stderr interleaving artifacts where older
tensor-stat diagnostics printed into the same combined `tee` stream. The final
scaffold still reports `172` pass invocations and zero failures.

Token-major scaffold:

```text
steps:                4
layers:               43
pass_invocations:     172
slots:                32
ctx:                  262144
sum_decode_ms:        5092.053121
ms_per_token:         1273.013280
projected_slot_tok/s: 25.137208
sum_ep_ms:            2077.942847
sum_hc_current_ms:    2717.209394
sum_final_hc_ms:      91.672502
wall_ms:              7873.373789
result:               PASS
```

## Saturation Finding

The first saturated stage is not q-head RoPE and not raw-SWA storage. The
first `>=65504` value appears in `kv_normed_max` at layer `1`, before KV RoPE
and before raw-SWA store:

```text
layer 0:
  kv_normed_max:       6510.59814
  kv_post_rope_max:    6510.59814
  raw_swa_row_max:     6656

layer 1:
  kv_normed_max:       436616.219
  kv_post_rope_max:    434584.688
  raw_swa_row_max:     65504
```

Other maxima from the parse-clean audit rows:

```text
current_max:           1.0 max, never saturated
attn_normed_max:       2500.0 max, never saturated
q_a_max:               820.093872 max, never saturated
q_a_normed_max:        first saturated at layer 23
kv_max:                1334.92078 max, never saturated
kv_normed_max:         first saturated at layer 1
q_b_pre_head_max:      first saturated at layer 17
q_heads_post_rope_max: 15.8982925 max, never saturated
kv_post_rope_max:      first saturated at layer 1
raw_swa_row_max:       first saturated at layer 1
```

This narrows the next correction target to the KV latent normalization/scaling
contract or the upstream HC/current-hidden bridge feeding `attn_kv_latent`.
The q-head path is large before head RMSNorm, but the normalized/rotated q
heads remain bounded.

## Outcome

The TP/EP runtime now has an explicit saturation audit gate for the true DS4
attention prefix/state path. The V100 evidence shows raw-SWA saturation is
inherited from `kv_normed`, not introduced by RoPE or by the raw-SWA store
itself.

## Next Sprint

Compare the TP/EP `kv_a_norm` path against DS4 reference semantics:

```text
attn_kv_latent output
  -> attn_kv_a_norm.weight
  -> reference RMS normalization and scale
  -> raw-KV storage format
```

The next sprint should add a reference-math microbench for `attn_kv_a_norm`
and, if needed, replace the current stable RMSNorm implementation with the
exact DS4 normalization/scaling contract before compressed-KV/indexer work.
