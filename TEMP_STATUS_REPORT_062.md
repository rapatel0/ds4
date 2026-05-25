# TEMP Status Report 062 - HC Current Breakdown

Date: 2026-05-25

## Current Focus

TP/EP serving performance. Sprint 350 split the broad pre-EP timer to determine
whether the remaining hot path is really HC-current work.

## V100 Result

Direct run:

```text
slots: 32
ctx: 262144
decode steps: 2
stream sync: on
generated tok/s decode: 92.630324
continuation tok/s decode: 99.664366
output_head_finite_bad: 0
```

HC-current substages:

```text
seed:        2.485326 ms
attn mix:   42.340819 ms
split:       1.245295 ms
gather:      6.960973 ms
ffn/router:  1.784974 ms
fill/pack:  28.248863 ms
subtotal:   83.066250 ms
```

The old broad field was:

```text
sum_hc_current_input_ms = 557.301289 ms
```

## Interpretation

The previous timer name was misleading. It includes true-attention and
compressed-KV prefix work before the EP/dense/compose stage starts. Actual
HC-current work is about `83 ms`; about `474 ms` remains in the broader
pre-EP prefix.

## Next Target

Split/optimize the true-attention/compressed-KV prefix, especially compressed
projection/store and dense-fill/WMMA fragmentation. HC-current gather/broadcast
is no longer the main suspected bottleneck.

## Artifacts

```text
logs/from-cluster/sprint350-hc-current-breakdown/cluster/
```
