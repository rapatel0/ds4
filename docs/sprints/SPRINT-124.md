# Sprint 124 - TurboMind Route-Row Reduce Probe

Date: 2026-05-21

## Objective

Test a larger production-path fusion than Sprint 123 without changing the
stage-overlapped serving topology. The candidate targets the packed TurboMind
routed FFN tail: current execution clears the output buffer, then atomically
scatters each sorted route row back to its token. Route compaction already
knows the sorted row for each original token/route pair, so this sprint adds an
opt-in row map and replaces the clear plus atomic scatter-add with a
deterministic per-token reduce.

This is still a bounded probe, not the final fused routed executor. It should
answer whether this FFN boundary is worth deeper CUTLASS/TurboMind-inspired
software pipelining.

## Implementation Plan

1. Add an optional `pair_rows[token * n_routes + route] = sorted_row` output to
   the TurboMind route scatter kernels.
2. Add `DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=1` as an opt-in launcher/runtime
   flag.
3. In the packed TurboMind routed path, allocate the row map only when the flag
   is enabled.
4. Replace the packed output `cudaMemset` plus `atomicAdd` scatter with a
   per-token/per-hidden reduce over the fixed route count.
5. Validate exact selected-token correctness and full-scheduler correctness on
   the V100 node.
6. Run same-binary 16-slot/256K served A/B against the current production
   default.

## Why This Fusion

Fusing kernels helps here only if it removes real memory traffic or
synchronization pressure. Small launch fusions have been correct but mostly
neutral. This candidate removes:

- one output-buffer clear;
- route-row atomic additions into the same token row;
- nondeterministic scatter order.

It does not yet fuse the full routed FFN mainloop. The next larger target would
be a TurboMind routed executor that pipelines route staging, MXFP4 dequant,
HMMA, SwiGLU, down projection, and accumulation at tile granularity.

## Validation

Local/static:

```text
bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-appliance-soak.sh
git diff --check
```

Cluster build:

```text
CUDA_ARCH=sm_70 make -j80 \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_f8_hmma_pair_swiglu_smoke \
  tools/ds4-v100-replay
```

Cluster correctness:

```text
DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=1 \
tests/cuda_v100_stage_scheduler_smoke --stage 0 --slots 16 --ctx 262144

DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=1 \
tests/cuda_v100_full_scheduler_smoke --slots 16 --ctx 262144 --expect-tm-layers 43
```

Served A/B:

```text
tools/ds4-v100-appliance-soak.sh \
  --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 \
  --ctx 262144 --slots 16 --active-microbatch 16 \
  --tokens 16 --requests 16 --warmup-requests 1
```

Run once with `DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=0` and once with
`DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=1`.

## Promotion Bar

- 16/16 token match.
- One 16-request tensor batch.
- No TurboMind fallback.
- At least roughly 2% generated tok/s improvement over same-binary control.

## Status

Completed. Defaults remain unchanged.

## Implementation

Added an opt-in route-row reduce path for the packed TurboMind routed FFN:

- `tm_scatter_routes_kernel` and `tm_build_routes_small_kernel` can now fill an
  optional `pair_rows` map from original token/route pair to sorted route row.
- `tm_reduce_sum_half_to_f32_by_pair_kernel` reduces each token/hidden output
  over the fixed route count using that map.
- `cuda_tm_routed_mxfp4_packed_impl` allocates the map and uses the reduce path
  only when `DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=1`.
- The launcher and operator docs expose the flag, defaulting to `0`.

## Results

Static/local validation:

```text
bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-appliance-soak.sh
git diff --check
```

Cluster build on `llamacpp-build-8gpu` under `/workspace/ds4`:

```text
CUDA_ARCH=sm_70 make -j80 \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_f8_hmma_pair_swiglu_smoke \
  tools/ds4-v100-replay
```

Correctness:

```text
cuda_f8_hmma_pair_swiglu_smoke: ok
cuda_v100_stage_scheduler_smoke: stage=0 gpu=0 layers=0-5 executed=6 tm_layers=6 token=16 pos=16 slots=16 ... ok
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=16 layers=43 tm_layers=43 ... ok
```

Same-binary 16-slot/256K served A/B:

| Candidate | Generated tok/s | Continuation tok/s | Correctness | Decision |
|---|---:|---:|---|---|
| control | `41.423928` | `38.834933` | 16/16 token match | low first control |
| `DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=1` | `43.822500` | `41.083593` | 16/16 token match | correct, not enough alone |
| control repeat | `43.517862` | `40.797995` | 16/16 token match | normal control band |
| candidate repeat | `42.998450` | `40.311047` | 16/16 token match | inside run noise |

Metrics for the normal control repeat and first candidate both reported one
tensor-batched group, 16 tensor-batched requests, and 256 tensor-batched tokens.

## Decision

Do not promote a new default. The route-row reduce is correct, but the measured
effect is inside the 16-slot run band and below the roughly 2% promotion bar
when compared with the normal control repeat.

Keep this diagnostic opt-in:

```text
DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=1
```

The result narrows the next optimization step: removing the final output clear
and atomics is not enough. The next meaningful candidate needs a larger
TurboMind/CUTLASS-style execution boundary, such as route-aware activation
staging plus MXFP4 HMMA plus SwiGLU/down accumulation with software-pipelined
tiles, or the batched attention output-A gap identified by the parallel review.
