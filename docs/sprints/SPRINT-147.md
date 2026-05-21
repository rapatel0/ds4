# Sprint 147 - 1536-Route Down-Reduce Checkpoint

Date: 2026-05-21

## Objective

Extend the existing TurboMind down-projection route-reduce epilogue to the
256-slot compact routed shape (`1536` routes) and validate the full scheduler
before moving on to larger fused-kernel work.

## Changes

- Added `ggml_turbomind_ds4_mxfp4_down_1536_m128_reduce` to the TurboMind C
  ABI and runtime loader.
- Reused the existing fixed `m128` SM70 MXFP4 down kernel with the DS4
  route-weighted F32 accumulation epilogue.
- Guarded selection behind `DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=1`.
- Kept production defaults unchanged.

## Validation

Full 43-layer scheduler smoke passed on the 8x V100 node:

```text
ctx=16384
slots=256
tm_layers=43
token=16
result=ok
```

Artifact:

- `logs/from-cluster/sprint147-smoke-down-reduce-1536/`

## Decision

Correctness is proven for the 1536-route down-reduce path, but served A/B was
deferred after the strategy pivot toward a larger software-pipelined fused
kernel. Keep this path as an explicit opt-in probe only.
