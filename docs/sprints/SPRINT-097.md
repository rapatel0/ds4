# Sprint 097 - CUDA Tensor Pool Default

Date: 2026-05-20

## Objective

Remove allocator churn from the warmed multi-slot serving path without changing
the appliance model format, tensor math, or KV residency contract.

## Changes

- Added an opt-in CUDA tensor pool behind `DS4_CUDA_TENSOR_POOL=1`.
  - The pool reuses freed owner allocations from `ds4_gpu_tensor_alloc()`.
  - Reuse is per current CUDA device and best-fit by allocation size.
  - Entries are released during `ds4_gpu_cleanup()`.
  - Managed KV allocations are not pooled.
- Added `DS4_CUDA_TENSOR_POOL_MAX_MIB`, default `2048`.
- Added appliance launcher controls:
  - `DS4_V100_CUDA_TENSOR_POOL=auto|0|1`
  - `DS4_V100_CUDA_TENSOR_POOL_MAX_MIB=2048`
  - `auto` enables the pool when `DS4_V100_ACTIVE_MICROBATCH > 1` and leaves it
    off for one-slot latency configs.
- Updated the deployment env, k8s config, and runbook with the new production
  default.

## Validation

Cluster build:

```text
make -C /workspace/ds4-sprint082 tools/ds4-v100-replay CUDA_ARCH=sm_70 -j8
```

Same-binary soak comparisons:

| Scenario | Pool | Generated tok/s | Continuation tok/s | Correctness |
| --- | --- | ---: | ---: | --- |
| 4 slots, 1M ctx | off | `11.902776` | `11.158852` | `token_match=4/4` |
| 4 slots, 1M ctx | on | `16.881653` | `15.826549` | `token_match=4/4` |
| 8 slots, 256K ctx | off | `17.193119` | `16.118549` | `token_match=8/8` |
| 8 slots, 256K ctx | on | `25.212896` | `23.637090` | `token_match=8/8` |

Launcher-default validation:

| Scenario | Launcher setting | Resolved pool | Generated tok/s | Correctness |
| --- | --- | ---: | ---: | --- |
| 4 slots, 1M ctx | `DS4_V100_CUDA_TENSOR_POOL=auto` | `1` | `17.532887` | `token_match=4/4` |
| 8 slots, 256K ctx | `DS4_V100_CUDA_TENSOR_POOL=auto` | `1` | `25.232220` | `token_match=8/8` |

Sampled VRAM did not increase at `nvidia-smi` sample granularity versus the
no-pool runs. Peak sampled memory stayed at the same per-GPU values for the
paired fixtures.

## Profile Result

Warmed served-path `nvprof --profile-from-start off` with
`--cuda-profiler-window` and the pool enabled:

| Bucket | Time |
| --- | ---: |
| F8 arena matmul | `61.83%`, `750.04 ms`, `11880` calls |
| TurboMind routed GEMM | `19.91%`, `241.60 ms`, `2376` calls |
| F32 matmul | `3.56%`, `43.20 ms` |
| Attention decode | `2.26%`, `27.38 ms` |
| HtoD memcpy | `0.15%`, `1.87 ms` |

CUDA API time after the pool:

| API bucket | Time |
| --- | ---: |
| `cudaMemcpy` | `920.34 ms`, `3168` calls |
| `cudaLaunchKernel` | `210.81 ms`, `39684` calls |
| `cudaFree` | `9.18 ms`, `37` calls |
| `cudaMalloc` | absent from the profiled request window |

This directly addresses the Sprint 096 allocator profile, where the warmed
served path spent `976.23 ms` in `cudaFree` and `141.85 ms` in `cudaMalloc`.

Artifacts:

- `logs/from-cluster/sprint097-tensor-pool/soak-4slot-default/summary.json`
- `logs/from-cluster/sprint097-tensor-pool/soak-4slot-pool/summary.json`
- `logs/from-cluster/sprint097-tensor-pool/soak-8slot-default/summary.json`
- `logs/from-cluster/sprint097-tensor-pool/soak-8slot-pool/summary.json`
- `logs/from-cluster/sprint097-tensor-pool/soak-4slot-auto-default/summary.json`
- `logs/from-cluster/sprint097-tensor-pool/soak-8slot-auto-default/summary.json`
- `logs/from-cluster/sprint097-tensor-pool/profile-4slot-pool/nvprof.log`

## Decision

Ship the tensor pool as the multi-slot appliance default. It is a production
path improvement because it keeps the model layout intact, preserves
correctness, avoids extra sampled VRAM pressure, and improves measured
aggregate decode throughput by roughly `42-47%` on the paired fixtures.

Next optimization should target the remaining served hot path:

1. F8 arena projection/shared matmul kernel shape and launch count.
2. `cudaMemcpy` API overhead from control/result copies.
3. TurboMind grouped GEMM occupancy and route/expert batching after F8 overhead
   is reduced.
