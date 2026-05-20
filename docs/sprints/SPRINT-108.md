# Sprint 108 - TurboMind Small-Route Build Probe

Date: 2026-05-20

## Objective

Test whether the per-layer TurboMind route count/prefix/scatter launch chain is
a useful production optimization target for the current DS4 V100 appliance.

## Context

Sprint 106 profiling showed TurboMind SM70 MXFP4 grouped GEMM at about `25%` of
GPU time, with route-build plumbing visible in the launch/API stream. Sprint
107 improved the grouped F8 attention-output path and left TurboMind route-build
fusion as the next concrete probe.

The production decode route shape is small:

- `slots=8`
- top-k routes per token are small
- total route count is under the small-route threshold
- total DS4 experts fit under `256`

That makes a single-block route builder a reasonable experiment.

## Implementation

- Added `tm_build_routes_small_kernel`, a one-block CUDA kernel that clears
  counts/cursors/offsets, counts routes, computes the prefix offsets, and fills
  sorted route pairs/weights for small route counts.
- Added `cuda_tm_build_routes()` to select either the small-route kernel or the
  previous generic count/prefix/scatter launch chain.
- Wired the helper into both the transient and persistent packed TurboMind
  routed-MXFP4 paths.
- Added `DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD`.
- Rebuilt and validated on V100 `sm_70`.

After A/B benchmarking, the launcher and deployment example keep this disabled
by default because it did not improve the primary 8-slot/256K target.

## V100 Validation

Cluster target:

- pod: `llamacpp-build-8gpu`
- node: `gpu-01`
- storage: k8s-local `/workspace`
- host resources: 8x V100-SXM2-32GB, 80 CPU cores, 256 GB RAM

Build:

```text
make tools/ds4-v100-replay tests/cuda_v100_turbomind_adapter_smoke \
  tests/cuda_v100_stage_scheduler_smoke tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke CUDA_ARCH=sm_70 -j80
```

Passed:

- `cuda_v100_turbomind_adapter_smoke`
- `cuda_v100_stage_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-s090 --stage 0 --slots 4`
- `cuda_v100_full_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-s090 --slots 8`
- `cuda_v100_selected_token_smoke --expected-token-hex 3136`
- rollback selected-token smoke with `DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=0`
- rebuilt-default selected-token smoke with default `turbomind_small_route_build=0`

The selected-token smoke selected token id `926`, text hex `3136`.

## Throughput

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Token Match |
|---|---:|---:|---:|---:|---:|
| Small-route opt-in | 262,144 | 8 | 31.393301 | 29.431220 | 8/8 |
| Generic rollback | 262,144 | 8 | 31.571254 | 29.598051 | 8/8 |
| Small-route opt-in repeat | 262,144 | 8 | 31.759013 | 29.774074 | 8/8 |
| Generic rollback repeat | 262,144 | 8 | 31.794180 | 29.807044 | 8/8 |
| Small-route opt-in | 1,048,576 | 4 | 20.249531 | 18.983935 | 4/4 |
| Generic rollback | 1,048,576 | 4 | 20.081695 | 18.826589 | 4/4 |

## Decision

Do not ship the small-route route builder as the production default. It is
correct, but the primary 8-slot/256K serving target is neutral to slightly
slower than the generic route builder. Keep the path as an opt-in diagnostic
because the 4-slot/1M result was slightly faster and it may still be useful for
focused route-build profiling.

The next useful optimization should be a larger hot-path change: F8 arena
matmul tiling/vectorization, TurboMind expert input layout, or a fused boundary
that removes route-expanded activation materialization rather than just route
metadata launches.

## Artifacts

- `logs/from-cluster/sprint108-tm-small-route-build/`
