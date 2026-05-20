# Sprint 093: Appliance Startup Warmup And GPU Profile

## Goal

Remove the cold concurrent first-request failure from the production appliance
path and capture profiler-backed evidence for the next throughput optimization.

## Context

Sprint 092 proved the full TurboMind appliance can serve a warm-started
4-request, 4-slot async batch from `/workspace/ds4-appliance-full-tm-s090`.
It also exposed two practical-use gaps:

- Four concurrent cold first requests can race lazy CUDA tensor-cache loading
  and fail in the TurboMind routed FFN path.
- Warm-started 4-slot async serving is correct but only reaches
  `11.256048` generated tok/s at 1M context.

The next stage needs the production launcher to own warmup, not the benchmark
client, and needs a GPU-kernel trace before choosing the next low-level
optimization.

## Implementation Plan

- Add a replay server `--startup-warmup` option that runs one internal base
  generation before the HTTP listener starts, then resets the runtime.
- Add `DS4_V100_STARTUP_WARMUP=auto|0|1` to the appliance launcher. In `auto`,
  enable startup warmup when `active_microbatch > 1`.
- Expose startup warmup in launcher checks, startup logs, `/v100/status`, and
  `/metrics`.
- Teach the appliance soak to run with `--warmup-requests 0` so validation can
  prove server-side warmup rather than client-side warmup.
- Run the full appliance 4-slot async soak with no client warmup.
- Capture a CUDA profiling artifact using available V100 tooling
  (`ncu`, `nvprof`, `cuobjdump`, or `nvdisasm`) against a bounded appliance
  decode run.

## Definition Of Done

- [x] `tools/ds4-v100-replay --serve --startup-warmup` warms and resets before
  listening.
- [x] `tools/ds4-v100-run-appliance.sh --check` reports the resolved startup
  warmup setting.
- [x] `/v100/status` and `/metrics` expose the startup warmup setting.
- [x] A 4-request, 4-slot appliance soak passes with `--warmup-requests 0`.
- [x] At least one GPU profiler artifact is captured from the V100 pod.
- [x] Sprint and vision docs record correctness, throughput, and profiler
  findings.

## Result

Sprint 093 moved warmup responsibility from the benchmark client into the
production appliance launcher/runtime and captured decode-window GPU profiler
evidence.

Implementation:

- Added `tools/ds4-v100-replay --startup-warmup`. The server runs one internal
  base-model generation before binding/listening, frees the warmup output, and
  resets the runtime.
- Added `DS4_V100_STARTUP_WARMUP=auto|0|1` to
  `tools/ds4-v100-run-appliance.sh`. In `auto`, multi-slot serving resolves to
  startup warmup enabled; one-slot serving stays off.
- Added `startup_warmup` to `/v100/status` and
  `ds4_v100_startup_warmup_enabled` to `/metrics`.
- Added a `--cuda-profiler-window` diagnostic option around non-server
  generation so `nvprof --profile-from-start off` and Nsight Compute can
  profile decode instead of cold appliance upload.

Build validation:

```text
make tools/ds4-v100-replay CUDA_ARCH=sm_70 -j8
```

The first cluster rebuild without `CUDA_ARCH=sm_70` failed on V100 DP4A
intrinsics in `ds4_cuda.cu`. The explicit V100 arch build is the correct
cluster build invocation when `ds4_cuda.o` is rebuilt.

Launcher validation:

```text
ds4-v100-run-appliance: config ok ... slots=4 active_microbatch=4 ...
async_pipeline_mode=per-step ... startup_warmup=1
...
./tools/ds4-v100-replay --serve ... --appliance-dir /workspace/ds4-appliance-full-tm-s090 --async-pipeline-mode per-step --startup-warmup
```

No-client-warmup soak:

```text
warmup_requests=0
startup_warmup=true
ds4_v100_startup_warmup_enabled 1
startup warmup ok prompt_tokens=5 token=30594 total_ms=1579.942
token_match=4/4
generated_tokens=64
continuation_tokens=60
elapsed_s=5.693406
latency_ms_avg=5642.920
aggregate_generated_tokens_per_second=11.241074
aggregate_continuation_tokens_per_second=10.538507
```

This proves concurrent traffic can arrive after server-side warmup without the
Sprint 092 cold-load TurboMind failure. Throughput is effectively unchanged
from Sprint 092, which is expected: this sprint removed an operational failure
mode, not a decode bottleneck.

Profiler findings:

- Cold full-process `nvprof` is dominated by appliance upload:
  `296.424s` / `99.53%` in HtoD copies. That profile is useful for startup
  work but not for decode optimization.
- Decode-window `nvprof --profile-from-start off` shows post-open decode GPU
  time led by:
  - `arena_f8_e4m3_b128_matmul_kernel`: `119.18ms`, `42.32%`, `1710` calls.
  - `[CUDA memcpy HtoD]`: `89.235ms`, `31.68%`, `801` calls.
  - TurboMind MXFP4 GEMM: `39.135ms`, `13.90%`, `342` calls.
- Targeted Nsight Compute on one F8 matmul launch reports:
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed = 58.71%`
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed = 12.96%`

Conclusion: the next throughput sprint should attack decode-window HtoD/control
traffic and F8 dense/projection launch shape before further TurboMind tuning.
TurboMind is visible in the hot path, but in this bounded profile it is not the
largest decode-window bucket.

## Stop Conditions

- Stop if server-side warmup changes the expected first token `3136`.
- Stop if startup warmup leaves the runtime in a dirty state after reset.
- Stop if profiler overhead makes full 4-slot profiling impractical; fall back
  to a one-slot or one-token bounded appliance profile and document the limit.
