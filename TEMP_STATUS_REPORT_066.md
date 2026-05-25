# TEMP Status Report 066 - Sprint 354

Date: 2026-05-25

## Current Focus

TP/EP only. This sprint tested whether fusing compressed-row RoPE with the
following F16 rounding pass reduces the emitted compressed-KV state/emit cost.

## Implemented

- Added `--true-ds4-compressed-kv-fused-rope-round-gate`.
- Added profiler flag `--fused-compressed-rope-round`.
- Added a fused compressed-row RoPE+round CUDA kernel.
- Added profiler summary counting for fused RoPE+round selected layers.

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
| control | `0` | `79.810167` | `400.951423` | `130.520098` | `54639` |
| fused RoPE+round | `41` | `79.344207` | `403.306067` | `130.382524` | `54639` |

## Decision

Reject promotion. The fused kernel is correct and slightly reduces the
state/emit sub-timers, but total decode is flat to slower. This individual
launch/pass is not the material lever.

Next work should target larger state/emit boundaries such as pooling plus
normalization or store plus pooling.

## Artifacts

```text
logs/from-cluster/sprint354-fused-compressed-rope-round/cluster/
```
