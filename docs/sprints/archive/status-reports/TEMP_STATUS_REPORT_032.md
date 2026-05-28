# TEMP Status Report 032

Date: 2026-05-24

## Current Topline

Sprint 320 added and validated a TP/EP true-attention output projection gate.
The gate now runs the real DS4 attention output tensors after the raw-window
attention read:

```text
rank-local raw/window heads [slots][4096]
  -> attn_output_a.weight
  -> gathered [slots][8192]
  -> attn_output_b.weight
  -> rank-local hidden shard [slots][512]
```

This is still diagnostic; it is not yet promoted into the hidden-state/residual
path used for parity.

## V100 Evidence

Cluster artifacts:

```text
logs/from-cluster/sprint320-true-attention-output-gate/cluster/
```

Build:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Final `32` slot / `256K` / `4` step gate:

```text
steps:                  4
layers:                 43
pass_invocations:       172
standalone output rows: 171
FAIL/bad_shape rows:    0
sum_decode_ms:          5685.874149
ms_per_token:           1421.468537
projected_slot_tok/s:   22.511930
sum_ep_ms:              2122.794070
sum_hc_current_input_ms:3269.508386
wall_ms:                8509.907562
checksum:               2999994000
result:                 PASS
```

The anchored output row count is `171` because one layer-2 row was not preserved
as a standalone grep-able line in the high-volume stdout stream. The scaffold
itself reports `172` pass invocations and there are zero failure rows.

Output projection maxima across recorded rows:

```text
max heads_max: 7.34393787
max out_a_max: 6.79435921
max out_b_max: 49.8094406
max output ms: 30.495427
bad counts:    0
```

## Important Correction

The first implementation assumed `attn_output_a.cols=32768`. The pack rejected
that assumption at layer 0:

```text
out_a_cols 4096
out_a_rows_per_gpu 1024
out_b_cols 8192
out_b_rows_per_gpu 512
```

The correct TP layout is better: `attn_output_a` consumes each rank's local
4096-wide attention-head block, then TP gathers the 8192-wide intermediate for
`attn_output_b`.

## Next

Promote the diagnostic attention output into the real layer residual path:

```text
attn_output_b shard -> attention residual / current hidden
then FFN norm/router/shared/routed FFN from that hidden
```

After promotion, rerun the official reference parity vector.
