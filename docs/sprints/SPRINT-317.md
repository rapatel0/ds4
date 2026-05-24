---
sprint: 317
title: TP/EP KV Norm Reference Comparison Gate
status: complete
started: 2026-05-24
branch: claude-takeover
---

# Sprint 317 - TP/EP KV Norm Reference Comparison Gate

## Goal

Determine whether Sprint 316's large `kv_normed` values are caused by the
TP/EP stable RMSNorm implementation or are produced by the same RMSNorm formula
used by `ds4.c` / `ds4_cuda.cu`.

## Scope

This sprint is TP/EP-only and diagnostic. It does not modify PP/layer-split
code and does not change serving defaults.

## Why This Sprint

Sprint 316 localized raw-SWA saturation to the `attn_kv_latent ->
attn_kv_a_norm` path. The next decision is whether to fix the TP/EP norm
implementation or to look upstream at the HC/current-hidden bridge and source
activation contract. A direct same-input comparison against the DS4 reference
RMSNorm kernel resolves that.

## Definition of Done

- Add `--true-ds4-attention-kv-norm-reference-gate`.
  Complete.
- Add launcher env
  `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_KV_NORM_REFERENCE=1`.
  Complete.
- The gate implies true-attention projection/residency and HC carry gates.
  Complete.
- For each layer, compute a reference `kv_a_norm` row using the plain DS4
  RMSNorm formula on the same `attn_kv_latent` output.
  Complete.
- Compare the TP/EP stable output to the reference output and report
  max-abs/max-rel drift plus both max values.
  Complete.
- Run the V100 `32` slot / `256K` / `4` step gate without `FAIL`.
  Complete.
- Record whether the next correction target is RMSNorm implementation or
  upstream HC/current-hidden activation semantics.
  Complete.

## V100 Evidence

Logs:

```text
logs/from-cluster/sprint317-kv-norm-reference/build.log
logs/from-cluster/sprint317-kv-norm-reference/run.log
```

Build:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Gated `32` slot / `256K` / `4` step smoke:

```text
tp_ep_true_attention_kv_norm_reference: 171 anchored rows, 170 parse-clean
tp_ep_true_attention_projection_prefix: 172
tp_ep_true_attention_rope:              172
tp_ep_true_attention_state_update:      171
tp_ep_true_attention_raw_window:        172
FAIL lines:                               0
```

The missing/malformed anchored rows are stdout/stderr interleaving artifacts in
the combined `tee` stream. The final scaffold still reports `172` pass
invocations and zero failures.

Token-major scaffold:

```text
steps:                4
layers:               43
pass_invocations:     172
slots:                32
ctx:                  262144
sum_decode_ms:        3557.935983
ms_per_token:         889.483996
projected_slot_tok/s: 35.975914
sum_ep_ms:            2109.028718
sum_hc_current_ms:    1136.959690
sum_final_hc_ms:      92.851819
wall_ms:              6461.425159
result:               PASS
```

## Finding

The same `kv_full` input produces the same large maximum value in both the
stable and plain/reference norm outputs, but the per-element drift is huge:

```text
layer 0:
  kv_in_max:       9.52407551
  kv_weight_max:   1.0625
  stable_max:      6510.59814
  reference_max:   6510.59814
  max_abs_diff:    3830.17261
  max_rel_diff:    2450.28955

layer 1:
  kv_in_max:       283.560913
  kv_weight_max:   1.7265625
  stable_max:      436616.219
  reference_max:   436616.219
  max_abs_diff:    144235.25
  max_rel_diff:    75269.875
```

Code inspection explains the drift: `block_sum_256_f32` and
`block_max_256_f32` return the reduced value only to the first warp. Threads
`32..255` receive zero after the second-stage warp reduction and compute
normalization with the wrong scale. In the RMSNorm kernels, most columns then
use `rsqrt(eps)` or otherwise incorrect row scale.

This is an implementation bug in the TP/EP reduction helper, not a DS4 model
property.

## Outcome

Sprint 317 added the reference comparison gate and identified the next concrete
fix: broadcast block-reduction results to every thread before RMSNorm/GEMV
consumers use them.

## Next Sprint

Fix `block_sum_256_f32` and `block_max_256_f32` so every thread receives the
same block-wide reduction result. Then rerun the KV norm reference gate and the
true-attention saturation gate. Expected outcome:

```text
max_abs_diff near numerical noise
kv_normed no longer artificially amplified by threads 32..255
raw-SWA saturation materially reduced or explained by real reference values
```
