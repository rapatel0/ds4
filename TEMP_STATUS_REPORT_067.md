# TEMP Status Report 067 - Sprint 355

Date: 2026-05-25

## Current Focus

TP/EP only. Sprint 355 tested the larger emitted compressed-KV state/emit
boundary: fusing compressor pooling and row normalization.

## Implemented

- Added `--true-ds4-compressed-kv-fused-pool-norm-gate`.
- Added profiler flag `--fused-compressed-pool-norm`.
- Added a fused emitted-row pool+norm CUDA kernel using one block per slot and
  shared memory for `head_dim <= 512`.
- Added profiler summary counting for fused pool+norm selected layers.

## Validation

Local:

- `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py`: pass
- `git diff --check`: pass

V100 pod:

- rebuilt `tools/ds4-v100-tp-ep-full-layer-smoke` with `CUDA_ARCH=sm_70`
- only known unused-kernel warnings

Same-binary direct A/B:

| Variant | Fused layers | Decode tok/s | Decode ms | Pre-EP compressed-KV ms | First token |
|---|---:|---:|---:|---:|---:|
| control | `0` | `81.189757` | `394.138390` | `131.016911` | `54639` |
| fused pool+norm | `41` | `81.687107` | `391.738686` | `128.201681` | `54639` |

## Decision

Promising but not promoted. The compressed-KV stage improved by `2.77 ms`,
but the one-run topline improvement was only `+0.61%`. Next test should combine
fused pool+norm with fused input fill and repeat the winner before changing
defaults.

## Artifacts

```text
logs/from-cluster/sprint355-fused-compressed-pool-norm/cluster/
```
