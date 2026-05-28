# TEMP Status Report 065 - Sprint 353

Date: 2026-05-25

## Current Focus

TP/EP only. This sprint tested whether collapsing the repeated current-vector
input fills in the emitted compressed-KV path produces a material performance
shift.

## Implemented

- Added `--true-ds4-compressed-kv-fused-input-fill-gate`.
- Added profiler flag `--fused-compressed-input-fill`.
- Added a fused ratio-4 CUDA fill kernel that writes:
  - attention compressor KV input
  - attention compressor gate input
  - indexer projection input
  - indexer compressor KV input
  - indexer compressor gate input
- Added profiler summary counting for fused-fill-selected layers.

## Validation

Local:

- `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py`: pass
- `git diff --check`: pass
- `make test`: not runnable locally without `ds4flash.gguf`

V100 pod:

- rebuilt `tools/ds4-v100-tp-ep-full-layer-smoke` with `CUDA_ARCH=sm_70`
- only known unused-kernel warnings

Same-binary direct A/B:

| Variant | Fused layers | Decode tok/s | Decode ms | Pre-EP compressed-KV ms | First token |
|---|---:|---:|---:|---:|---:|
| control | `0` | `79.011931` | `405.002124` | `130.391665` | `54639` |
| fused | `21` | `80.534845` | `397.343535` | `129.781758` | `54639` |

## Decision

The fused input fill is correct but not a large enough win to promote. It stays
as an opt-in diagnostic. The next work should target compressor/indexer
state+emit fusion or the dense/state boundary, because fill-only work moved
the compressed-KV stage by less than `1 ms`.

## Artifacts

```text
logs/from-cluster/sprint353-fused-compressed-input-fill/cluster/
logs/from-cluster/pause-request-current-tests/cluster/
```
