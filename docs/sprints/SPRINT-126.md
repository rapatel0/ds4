# Sprint 126 - Routed Expert Pipeline Profile

Date: 2026-05-21

## Objective

Move optimization work back to a large production hot-path boundary. The
question this sprint answers is where a CUTLASS/TurboMind-style
software-pipelined routed-expert kernel should start: route staging, fused
gate/up, SwiGLU, down, or scatter.

## Implementation

Added an opt-in production-path profiler:

- `DS4_V100_TURBOMIND_PROFILE=1`
- CUDA-event timing around packed TurboMind routed FFN stages:
  - route build
  - activation gather
  - fused gate/up grouped GEMM
  - SwiGLU
  - down grouped GEMM
  - scatter or route-row reduce
- per-GPU summary at cleanup/exit
- active expert and max routed-row shape summary from the production offsets
- launcher/env/docs wiring

The profiler intentionally synchronizes between stages. Its timing breakdown is
useful for kernel design; its tok/s is not a production number.

## Validation

Local/static:

```text
bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-appliance-soak.sh
git diff --check
```

Cluster build on `llamacpp-build-8gpu` under `/workspace/ds4`:

```text
CUDA_ARCH=sm_70 make -j80 \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tools/ds4-v100-replay
```

Cluster correctness with profiling enabled:

```text
DS4_V100_TURBOMIND_PROFILE=1 \
tests/cuda_v100_stage_scheduler_smoke --stage 0 --slots 16 --ctx 262144

DS4_V100_TURBOMIND_PROFILE=1 \
tests/cuda_v100_full_scheduler_smoke --slots 16 --ctx 262144 --expect-tm-layers 43
```

No-profile served sanity on the new binary:

```text
tools/ds4-v100-appliance-soak.sh \
  --ctx 262144 --slots 16 --active-microbatch 16 \
  --tokens 16 --requests 16 --warmup-requests 1
```

## Results

Stage 0 profile, 6 layers on GPU0:

| Stage | Time |
|---|---:|
| route build | `1.025 ms` |
| activation gather | `0.192 ms` |
| fused gate/up grouped GEMM | `2.067 ms` |
| SwiGLU | `0.163 ms` |
| down grouped GEMM | `0.985 ms` |
| scatter/reduce | `0.216 ms` |
| total | `4.715 ms` |

Full 43-layer profile aggregate across 8 GPUs:

| Stage | Time | Share of profiled routed-FFN time |
|---|---:|---:|
| route build | `4.743 ms` | `16.8%` |
| activation gather | `1.041 ms` | `3.7%` |
| fused gate/up grouped GEMM | `13.268 ms` | `47.0%` |
| SwiGLU | `0.898 ms` | `3.2%` |
| down grouped GEMM | `6.616 ms` | `23.4%` |
| scatter/reduce | `1.303 ms` | `4.6%` |
| total | `28.242 ms` | `100%` |

Shape summary:

```text
avg_tokens=16
avg_routes=96
avg_active_experts=6
max_routes_call=96
max_routes_expert=16
```

No-profile served sanity:

```text
aggregate_generated_tokens_per_second=43.453309
aggregate_continuation_tokens_per_second=40.737477
token_match=16/16
```

## Decision

Ship the profiler default-off. Defaults remain unchanged for throughput.

The data supports the software-pipelining thesis, but it also rules out a small
SwiGLU-only optimization as the main lever. The separate SwiGLU kernel is only
about `3%` of profiled routed-FFN time. The larger buckets are fused gate/up and
down grouped GEMMs, with route build still material at this small routed shape.

The next bounded implementation should be TurboMind gated-SiLU support with an
interleaved fused gate/up pack, because the copied TurboMind GEMM already has a
gated epilogue. That can remove the `total_routes * 2 * mid` intermediate and
the separate activation launch. If that does not move throughput materially, the
next step has to be a deeper persistent routed-expert pipeline rather than more
single-stage fusions.
