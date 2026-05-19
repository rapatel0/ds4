# Sprint 057 Follow-ups

## Persistent Grouped MoE Kernel

- **Severity**: Critical
- **Target sprint**: Sprint 058
- **Files**: `ds4_cuda.cu`, `ds4_gpu.h`,
  `tests/cuda_v100_mxfp4_moe_smoke.c`
- **Issue**: Deterministic request coalescing and the opt-in batched FFN slice
  did not raise GPU utilization.
- **Evidence**: Sprint 057 default two-slot sustained decode measured
  `3.662490` generated tok/s and `10.756%` average GPU utilization. The opt-in
  batched FFN path measured `3.630558` generated tok/s and `10.969%` average
  GPU utilization.
- **Next step**: Prototype a persistent grouped MoE kernel or a tile-level
  packed INT8/INT4/MXFP4 route kernel that avoids per-slot copies and makes
  route batches large enough to feed Volta.

## Remove Batch-Slice Copy Overhead

- **Severity**: Important
- **Target sprint**: Sprint 058
- **Files**: `ds4_v100_layer_execute.c`, `ds4_cuda.cu`, `ds4_gpu.h`
- **Issue**: `DS4_V100_BATCH_LAYER_FFN` copies per-slot FFN inputs into a
  contiguous batch tensor and then views/copies outputs back into per-slot
  paths. At two slots this overhead exceeded the benefit of one batched routed
  primitive.
- **Evidence**: The opt-in path is correct but slower than the default path.
- **Next step**: Either pass slot tensor pointers directly into a pointer-array
  CUDA kernel or move batching earlier so HC/attention/FFN all consume a shared
  active-slot layout.

## Timing Counter Semantics For Batched Responses

- **Severity**: Important
- **Target sprint**: Sprint 058
- **Files**: `tools/ds4-v100-replay.c`, `ds4_v100_replay.c`,
  `tools/ds4-v100-sustained-decode-bench.sh`
- **Issue**: Batched requests currently receive shared batch counters, so
  per-response timing fields can be misleading even when aggregate tok/s is
  correct.
- **Evidence**: Sprint 057 two-slot rows show deterministic batch counters but
  lower per-response continuation tok/s than aggregate continuation tok/s.
- **Next step**: Report batch-level timing separately from per-response timing.

## Configurable Rendezvous Window

- **Severity**: Nice-to-have
- **Target sprint**: Sprint 058+
- **Files**: `tools/ds4-v100-replay.c`, deployment env examples
- **Issue**: The rendezvous window is currently a fixed 5 ms.
- **Evidence**: It works for the benchmark but may not be ideal for latency
  versus throughput in practical serving.
- **Next step**: Add a documented environment variable for the wait budget once
  there is enough traffic data to choose useful defaults.
