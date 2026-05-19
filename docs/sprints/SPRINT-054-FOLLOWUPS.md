# Sprint 054 Follow-ups

## Fuse Routed Down Projection And Accumulation

- **Severity**: Critical
- **Target sprint**: Sprint 055
- **Files**: `ds4_cuda.cu`, `ds4_gpu.h`, `ds4_v100_layer_execute.c`,
  `tests/cuda_v100_mxfp4_moe_smoke.c`
- **Issue**: Sprint 054 fused gate+up+SwiGLU, but each route still launches a
  separate MXFP4 down projection and a separate add into the route accumulator.
- **Evidence**: Sustained decode improved only `2.8-3.4%`; GPU utilization
  stayed around `11%`.
- **Next step**: Add a fused MXFP4 down+weighted accumulation kernel or a
  grouped route-down primitive, then compare against Sprint 054.

## Group Multiple Routed Experts Per Layer

- **Severity**: Critical
- **Target sprint**: Sprint 055
- **Files**: `ds4_cuda.cu`, `ds4_gpu.h`, `ds4_v100_layer_execute.c`
- **Issue**: The executor still loops over the six selected routes serially.
  The fused gate/up primitive reduces launches inside a route but does not
  group routes or slots.
- **Evidence**: Stage decode remains the dominant cost and utilization did not
  rise after Sprint 054.
- **Next step**: Build an owned grouped MXFP4 route primitive for the six
  selected experts, with a fallback to the current per-route path.

## Deterministic Batch Coalescing For Benchmarks

- **Severity**: Important
- **Target sprint**: Sprint 055
- **Files**: `tools/ds4-v100-sustained-decode-bench.sh`,
  `tools/ds4-v100-replay.c`
- **Issue**: The Sprint 054 two-slot benchmark improved throughput but did not
  capture a tensor-batched request group in that run. Sprint 053 proved the
  branch can execute, but the benchmark coalescing is race-sensitive.
- **Evidence**: Sprint 054 `server_status_after` recorded
  `tensor_batched_groups=0` for the two-slot case.
- **Next step**: Add a deliberate benchmark coalescing barrier or a tiny
  configurable server-side batch wait for throughput profiles only.

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Fuse routed down projection and accumulation | Critical | Sprint 055 | `ds4_cuda.cu`, `ds4_gpu.h`, `ds4_v100_layer_execute.c`, `tests/cuda_v100_mxfp4_moe_smoke.c` |
| Group multiple routed experts per layer | Critical | Sprint 055 | `ds4_cuda.cu`, `ds4_gpu.h`, `ds4_v100_layer_execute.c` |
| Deterministic batch coalescing for benchmarks | Important | Sprint 055 | `tools/ds4-v100-sustained-decode-bench.sh`, `tools/ds4-v100-replay.c` |
