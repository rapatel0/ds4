# Sprint 059 Follow-ups

## Pointer-Input Or Gather-Based FFN Batch

- **Severity**: Critical
- **Target sprint**: Sprint 060
- **Files**: `ds4_cuda.cu`, `ds4_gpu.h`, `ds4_v100_layer_execute.c`,
  `tests/cuda_v100_mxfp4_moe_smoke.c`
- **Issue**: Persistent scratch removes allocation churn, but
  `execute_ffn_delta_batch` still copies every per-slot FFN input into a
  contiguous `input_batch_t` before routed MXFP4 execution.
- **Evidence**: Sprint 059 default-batched two-slot throughput improved to
  `3.862932` generated tok/s, but average GPU utilization remains `11.434%`.
- **Next step**: Either add a one-launch pointer-array gather to replace the N
  device-to-device copies, or add a pointer-input variant of the routed MXFP4
  batch primitive and skip `input_batch_t` for the routed path.

## Persistent Batch Views

- **Severity**: Important
- **Target sprint**: Sprint 060
- **Files**: `ds4_v100_layer_execute.c`, `ds4_v100_layer_execute.h`
- **Issue**: The batch path still allocates lightweight tensor views for router
  output and routed output slices on every call.
- **Evidence**: Fermat's Sprint 059 inspection identified these CPU-side view
  allocations as remaining non-math overhead after persistent scratch.
- **Next step**: Persist router and routed-output views inside the scratch
  object if the pointer-input kernel is not implemented first.

## Higher Slot-Tier Retest

- **Severity**: Important
- **Target sprint**: Sprint 060+
- **Files**: `tools/ds4-v100-sustained-decode-bench.sh`,
  `logs/from-cluster/`
- **Issue**: Sprint 059 benchmarked the final default-batched path at one and
  two slots only.
- **Evidence**: The path is faster at two slots, but higher slot tiers may
  expose different KV memory pressure, routing imbalance, or copy overhead.
- **Next step**: Retest 4-slot and 8-slot tiers at practical context sizes
  after the input-copy change.
