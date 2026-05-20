# Sprint 096 - Served Decode Profiling Window

Date: 2026-05-20

## Objective

Capture GPU profile evidence for the actual warmed HTTP appliance path, not the
cold one-shot replay path. Sprint 095 fixed request coalescing; this sprint
uses that stable serving path to decide the next optimization target.

## Changes

- Extended `--cuda-profiler-window` to `tools/ds4-v100-replay --serve`.
  - Startup warmup still runs before the server listens and is not profiled.
  - Each generation batch calls `cudaProfilerStart/Stop`, so
    `nvprof --profile-from-start off` captures served decode work.
- Added launcher env `DS4_V100_CUDA_PROFILER_WINDOW=0`.
  - Set it to `1` only for `nvprof` or Nsight diagnostic runs.
  - Documented the flag in the deployment env and appliance runbook.

## Validation

Cluster build:

```text
make -C /workspace/ds4-sprint082 tools/ds4-v100-replay CUDA_ARCH=sm_70 -j8
```

Served profile command shape:

```text
nvprof --profile-from-start off --log-file nvprof-server.log \
  ./tools/ds4-v100-run-appliance.sh
```

Environment:

- `DS4_V100_APPLIANCE_DIR=/workspace/ds4-appliance-full-tm-s090`
- `DS4_V100_CTX=1048576`
- `DS4_V100_SLOTS=4`
- `DS4_V100_ACTIVE_MICROBATCH=4`
- `DS4_V100_MICROBATCH_WAIT_US=auto`
- `DS4_V100_CUDA_PROFILER_WINDOW=1`
- `DS4_V100_MAX_REQUESTS=4`

Correctness:

- Four concurrent HTTP responses returned `generated_tokens=16`.
- First generated token remained `3136` in all four responses.

## Profile Result

The warmed served path differs materially from the earlier one-shot profile:

| Bucket | Served HTTP batch |
| --- | ---: |
| F8 arena matmul | `61.64%`, `828.48 ms`, `11880` calls |
| TurboMind routed GEMM | `20.15%`, `270.79 ms`, `2376` calls |
| F32 matmul | `3.48%`, `46.77 ms` |
| Attention decode | `2.28%`, `30.64 ms` |
| HtoD memcpy | `0.14%`, `1.86 ms` |

CUDA API time is dominated by allocator churn:

| API bucket | Time |
| --- | ---: |
| `cudaFree` | `976.23 ms`, `23909` calls |
| `cudaMalloc` | `141.85 ms`, `23760` calls |
| `cudaMemcpy` | `250.03 ms`, `3168` calls |
| `cudaLaunchKernel` | `226.39 ms`, `39684` calls |

Artifacts:

- `logs/from-cluster/sprint096-server-profile/nvprof-server.log`
- `logs/from-cluster/sprint096-server-profile/response_*.json`
- `logs/from-cluster/sprint096-server-profile/runtime/startup.env`

## Decision

HtoD is not the served-path blocker after startup warmup. The next optimization
target should be:

1. F8 projection/shared matmul kernels and launch count.
2. Allocator churn inside the served generation loop.
3. TurboMind grouped GEMM launch shape after F8 and allocation overhead are
   reduced.
