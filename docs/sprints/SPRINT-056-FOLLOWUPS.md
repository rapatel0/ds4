# Sprint 056 Follow-ups

## Deterministic Token-Step Coalescing

- **Severity**: Critical
- **Target sprint**: Sprint 057
- **Files**: `tools/ds4-v100-sustained-decode-bench.sh`,
  `tools/ds4-v100-replay.c`, `ds4_v100_scheduler.c`
- **Issue**: The Sprint 056 two-slot sustained run reported
  `tensor_batched_groups=0`, even though the server was configured with
  `tensor_batched_slots=true`.
- **Evidence**:
  `logs/from-cluster/sprint056-grouped-mxfp4-routes/cases/case_2_ctx1048576_s2_sequential_tok16/server_status_after.json`.
- **Next step**: Add an explicit throughput-mode batch wait/coalescing barrier
  so multi-slot benchmark cases deterministically exercise token-step batching.

## Batched Layer Executor

- **Severity**: Critical
- **Target sprint**: Sprint 057
- **Files**: `ds4_v100_layer_execute.c`, `ds4_v100_scheduler.c`,
  `ds4_cuda.cu`, `ds4_gpu.h`
- **Issue**: Grouping selected routes inside one slot improves throughput by
  about `4-5%`, but GPU utilization remains near `11%`.
- **Evidence**: Sprint 056 sustained decode improved generated tok/s from
  `3.410425` to `3.552642` at one slot and from `3.503283` to `3.676873` at
  two slots, with average GPU utilization still around `11%`.
- **Next step**: Add a batch entrypoint for the layer executor so active slots
  reach the expensive attention/FFN kernels together instead of relying only on
  request-loop batching.

## Persistent Or Tensor-Core-Friendly MoE Kernel

- **Severity**: Important
- **Target sprint**: Sprint 058+
- **Files**: `ds4_cuda.cu`, `ds4_gpu.h`, `tests/cuda_v100_mxfp4_moe_smoke.c`
- **Issue**: The grouped route primitive still performs scalar source-MXFP4 row
  reductions and does not turn routed experts into large Volta HMMA or integer
  tensor-core work.
- **Evidence**: Grouping six routes reduced launch overhead but did not raise
  average GPU utilization.
- **Next step**: Prototype a persistent grouped MoE kernel or a packed INT8/INT4
  route-batch path behind a correctness-preserving quality gate.
