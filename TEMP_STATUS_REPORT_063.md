# TEMP Status Report 063

Date: 2026-05-25

## Current Focus

TP/EP only. Sprint 351 completed passive pre-EP prefix telemetry for the
resident direct token-major path. No PP/layer-split work and no MTP work.

## Latest V100 Result

Direct TP/EP run:

```text
slots: 32
ctx: 262144
decode steps: 2
stream sync: on
returncode: 0
output_head_finite_bad: 0
generated tok/s decode: 83.265760
continuation tok/s decode: 99.612333
sum_decode_ms: 768.623264
sum_hc_current_input_ms: 626.823138
```

Artifact:

```text
logs/from-cluster/sprint351-true-attn-prefix-breakdown/cluster/
```

## New Prefix Breakdown

Across `86` layer-step invocations:

| Stage | Time |
|---|---:|
| Compressed KV projection/store | `228.813152 ms` |
| Attention projection prefix | `170.865666 ms` |
| Attention state update | `105.654904 ms` |
| HC-current bridge | `85.249101 ms` |
| Raw/window attention read | `34.932798 ms` |
| Typed KV history load | `1.271677 ms` |
| Attention output projection | `0.000000 ms` |
| Post-attention FFN input | `0.000000 ms` |
| Measured prefix total | `626.787298 ms` |

## Interpretation

The broad pre-EP timer is now explained by measured prefix stages. HC-current
is not the dominant bottleneck. The largest measured owner is compressed KV
projection/store, followed by attention projection and attention state update.

## Next Move

Plan and execute a TP/EP-only optimization sprint against compressed KV
projection/store fragmentation. The likely work is to reduce dense-fill/staging
around compressor/indexer paths, batch or fuse emitted-row store work where
safe, and verify with the direct profiler before any HTTP A/B.

