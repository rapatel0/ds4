# TEMP Status Report 029

Date: 2026-05-24

## Current Topline

TP/EP serving remains operational but not model-correct. Sprint 317 found a
real TP/EP implementation bug in the shared block-reduction helper used by
RMSNorm-style kernels.

The important result:

```text
block_sum_256_f32 / block_max_256_f32 do not broadcast to all threads
threads 32..255 use the wrong reduced value
RMSNorm rows are materially wrong
```

This likely explains the Sprint 316 `kv_normed` amplification and may affect
other TP/EP helper kernels that consume `block_sum_256_f32`.

## Sprint 317 Progress

Added the KV norm reference comparison gate:

```text
binary flag:           --true-ds4-attention-kv-norm-reference-gate
launcher env:          DS4_V100_TP_EP_TRUE_DS4_ATTENTION_KV_NORM_REFERENCE=1
path:                  TP/EP only
PP work:               none
```

The gate computes:

```text
kv_full
  -> current stable RMSNorm path
  -> plain DS4/reference RMSNorm path
  -> max-abs/max-rel diff
```

## V100 Evidence

Cluster logs:

```text
logs/from-cluster/sprint317-kv-norm-reference/build.log
logs/from-cluster/sprint317-kv-norm-reference/run.log
```

Gated `32` slot / `256K` / `4` step smoke:

```text
tp_ep_true_attention_kv_norm_reference:
  anchored_rows: 171
  parse_clean:   170

tp_ep_true_attention_projection_prefix: 172
tp_ep_true_attention_rope:              172
tp_ep_true_attention_state_update:      171
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
  projected_slot_tok/s: 35.975914
  sum_decode_ms:        3557.935983
  wall_ms:              6461.425159
  result:               PASS
```

## Finding

The same input produces huge per-element drift between the TP/EP stable
RMSNorm output and the plain reference RMSNorm output:

```text
layer 0:
  stable_max:      6510.59814
  reference_max:   6510.59814
  max_abs_diff:    3830.17261

layer 1:
  stable_max:      436616.219
  reference_max:   436616.219
  max_abs_diff:    144235.25
```

The cause is code-level:

```text
block_sum_256_f32:
  second-stage warp reduction result is returned only to first warp
  threads >= 32 receive zero

block_max_256_f32:
  same broadcast bug
```

RMSNorm kernels then use inconsistent row scales across columns.

## Next

Fix the block-reduction helpers and rerun:

```text
1. KV norm reference gate
2. saturation audit gate
3. raw-window attention gate
```

This should happen before compressed-KV/indexer work or attention-output
promotion.
