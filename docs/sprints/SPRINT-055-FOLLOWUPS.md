# Sprint 055 Follow-ups

## Group Selected Routes

- **Severity**: Critical
- **Target sprint**: Sprint 056
- **Files**: `ds4_cuda.cu`, `ds4_gpu.h`, `ds4_v100_layer_execute.c`,
  `tests/cuda_v100_mxfp4_moe_smoke.c`
- **Issue**: Sprint 054 and 055 reduced launches inside a single route, but the
  executor still loops over six routes serially.
- **Evidence**: Sprint 055 improved generated tok/s by less than 1% over Sprint
  054 and average GPU utilization remains around 11%.
- **Next step**: Build a grouped MXFP4 routed primitive that processes all six
  selected experts in one or two kernels, with the current per-route path as
  fallback.

## Batch Layer Execution Across Slots

- **Severity**: Critical
- **Target sprint**: Sprint 056 or Sprint 057
- **Files**: `ds4_v100_scheduler.c`, `ds4_v100_layer_execute.c`,
  `ds4_cuda.cu`
- **Issue**: `decode_hc_batch` still loops through slots and calls the
  single-slot layer executor. Request batching therefore does not guarantee
  kernel batching.
- **Evidence**: Two-slot sustained runs remain near single-slot tok/s and
  utilization.
- **Next step**: Add a layer executor batch entrypoint for at least one
  representative expensive path.

## Benchmark Coalescing Control

- **Severity**: Important
- **Target sprint**: Sprint 056
- **Files**: `tools/ds4-v100-sustained-decode-bench.sh`,
  `tools/ds4-v100-replay.c`
- **Issue**: The two-slot Sprint 054 and 055 sustained runs did not capture
  tensor-batched HTTP groups, while Sprint 053 did. The current benchmark
  coalescing is race-sensitive.
- **Evidence**: `server_status_after.tensor_batched_groups=0` in Sprint 054
  and Sprint 055 two-slot artifacts.
- **Next step**: Add an explicit coalescing barrier or throughput-profile batch
  wait so multi-slot benchmark cases deterministically exercise the request
  batch branch.

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Group selected routes | Critical | Sprint 056 | `ds4_cuda.cu`, `ds4_gpu.h`, `ds4_v100_layer_execute.c`, `tests/cuda_v100_mxfp4_moe_smoke.c` |
| Batch layer execution across slots | Critical | Sprint 056 or Sprint 057 | `ds4_v100_scheduler.c`, `ds4_v100_layer_execute.c`, `ds4_cuda.cu` |
| Benchmark coalescing control | Important | Sprint 056 | `tools/ds4-v100-sustained-decode-bench.sh`, `tools/ds4-v100-replay.c` |
