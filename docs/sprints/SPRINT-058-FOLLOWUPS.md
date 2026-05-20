# Sprint 058 Follow-ups

## Persistent Or Copy-Free MoE Batch Path

- **Severity**: Critical
- **Target sprint**: Sprint 059
- **Files**: `ds4_v100_layer_execute.c`, `ds4_cuda.cu`, `ds4_gpu.h`,
  `tests/cuda_v100_mxfp4_moe_smoke.c`
- **Issue**: Router readback suppression improved two-slot generated tok/s only
  from `3.662490` to `3.704572`. Average GPU utilization remains about `11%`.
- **Evidence**: Sprint 058 sustained 1M decode measured `3.704572` generated
  tok/s and `3.473036` continuation tok/s at two slots.
- **Next step**: Move from cleanup to architecture: persistent FFN scratch,
  copy-free active-slot input layout, or a persistent grouped MoE kernel that
  makes routed expert work larger per launch.

## Batch FFN Scratch Reuse

- **Severity**: Important
- **Target sprint**: Sprint 059
- **Files**: `ds4_v100_layer_execute.c`, `ds4_v100_scheduler.c`
- **Issue**: The opt-in `DS4_V100_BATCH_LAYER_FFN` path still allocates and
  frees temporary tensors per layer call, then copies per-slot inputs into a
  contiguous batch tensor.
- **Evidence**: Sprint 057 showed the opt-in batched FFN path was correct but
  slower than the default path.
- **Next step**: Add scheduler-owned or layer-owned persistent scratch sized by
  `DS4_V100_LAYER_MAX_BATCH`, then retest the opt-in path before attempting a
  larger kernel rewrite.

## Batch-Level Timing Semantics

- **Severity**: Important
- **Target sprint**: Sprint 059+
- **Files**: `tools/ds4-v100-replay.c`, `ds4_v100_replay.c`,
  `tools/ds4-v100-sustained-decode-bench.sh`
- **Issue**: Batched requests still receive shared batch counters, so
  per-response timing fields can be misleading even when aggregate tok/s is
  correct.
- **Evidence**: Sprint 057 and Sprint 058 sustained rows report meaningful
  aggregate tok/s, but response-local continuation rates are not an isolated
  per-request measurement.
- **Next step**: Report batch-level timing separately from per-response timing.
