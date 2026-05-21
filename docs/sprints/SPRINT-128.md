# Sprint 128 - TurboMind Compact Expert Schedule

Date: 2026-05-21

## Objective

Attack the larger routed-expert boundary left after Sprint 127. The current
packed TurboMind path still presents the full 256-expert grouped schedule to
TurboMind even when a 16-slot decode step usually activates only a small
subset of experts. This sprint adds an opt-in compact schedule so the grouped
GEMMs see active experts first and empty padding groups only up to
`total_routes`, not all 256 experts.

## Implementation

- Added `DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1`, promoted as the launcher
  default after V100 A/B.
- Keep route build, sorted route rows, and final scatter semantics unchanged.
- After route build and cached TurboMind table lookup, build compact device
  tables:
  - compact offsets derived from the existing 256-expert offsets;
  - compact weight/scale pointer tables for gate/up/gate_up/down;
  - active experts first, remaining groups empty and pointed at a valid table
    entry.
- Dispatch gate/up and down GEMMs with `num_experts = total_routes` when the
  compact path is enabled.
- Preserve Sprint 127 gated-SiLU behavior and the route-row-reduce opt-in.

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

Cluster correctness:

```text
DS4_V100_TURBOMIND_GATED_SILU=1 \
DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1 \
tests/cuda_v100_stage_scheduler_smoke --stage 0 --slots 16 --ctx 262144

DS4_V100_TURBOMIND_GATED_SILU=1 \
DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1 \
tests/cuda_v100_full_scheduler_smoke --slots 16 --ctx 262144 --expect-tm-layers 43

DS4_V100_TURBOMIND_GATED_SILU=1 \
DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1 \
DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=1 \
tests/cuda_v100_full_scheduler_smoke --slots 16 --ctx 262144 --expect-tm-layers 43

DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1 \
tests/cuda_v100_full_scheduler_smoke \
  --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 \
  --slots 16 --ctx 262144 --expect-tm-layers 43
```

Launcher default check:

```text
DS4_V100_APPLIANCE_DIR=/workspace/ds4-appliance-full-tm-fused-s111 \
DS4_V100_TURBOMIND_LIB=/workspace/ds4/build/turbomind-v100-s127/libggml-turbomind.so \
tools/ds4-v100-run-appliance.sh --allow-missing --check

turbomind_compact_schedule=1
```

## Results

Profiled full 43-layer routed-FFN aggregate with gated-SiLU plus compact
schedule:

| Stage | Time |
|---|---:|
| route build plus compact schedule | `4.701 ms` |
| activation gather | `1.053 ms` |
| gated gate/up grouped GEMM | `13.226 ms` |
| standalone SwiGLU | `0.000 ms` |
| down grouped GEMM | `5.664 ms` |
| scatter/reduce | `1.232 ms` |
| total | `26.265 ms` |

Compared with Sprint 127's gated profile, compact scheduling lowers down-GEMM
time but raises gate/up time, so the profile is only modestly better. The
served path moves more clearly.

Served A/B at `ctx=262144`, `slots=16`, `active_microbatch=16`:

| Appliance / flags | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|
| Gated appliance, compact off | `43.879880` | `41.137387` | `16/16` token match |
| Gated appliance, compact on | `46.328184` | `43.432672` | `16/16` token match |
| Gated appliance, compact + route-row-reduce | `46.394722` | `43.495052` | `16/16` token match |
| Sprint 111 fused appliance, compact on explicitly | `45.747461` | `42.888244` | `16/16` token match |
| Sprint 111 fused appliance, launcher default | `45.888778` | `43.020729` | `16/16` token match |

## Decision

Promote `DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1` as the launcher default. Keep
`DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=0` because its added gain on top of compact
scheduling was small in this run, and Sprint 124 had noisy route-row-reduce
results. Keep `DS4_V100_TURBOMIND_GATED_SILU=0` by default because it requires
an interleaved gated appliance pack.

## Risk

The main risk is that reducing group count also changes TurboMind's scheduler
shape and may pick a worse SM70 kernel for some routed shapes. Roll back with
`DS4_V100_TURBOMIND_COMPACT_SCHEDULE=0`.
