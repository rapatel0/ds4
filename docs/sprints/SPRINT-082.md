# Sprint 082: TurboMind Routed Expert Adapter Smoke

## Status

Complete.

## Overview

Sprint 081 proved that the copied TurboMind grouped MXFP4 GEMM builds and runs
on V100. Sprint 082 turns that proof into the first DS4 adapter boundary:
source MXFP4 expert bytes are packed through TurboMind, routed rows are grouped
by expert, DS4 SwiGLU and route weights are applied between gate/up and down,
and the final routed output is compared against the existing DS4 source-MXFP4
arena implementation.

This is still opt-in test coverage, not the production scheduler default.

## Goals

1. Add a V100 adapter smoke that uses copied TurboMind from `ds4`.
2. Use DS4 expert dimensions: gate/up `N=2048,K=4096` and down
   `N=4096,K=2048`.
3. Pack source MXFP4 expert rows through `ggml_turbomind_pack_weight_expert`.
4. Run `ggml_turbomind_mul_mat_grouped` for gate, up, and down.
5. Apply DS4 clamp/SwiGLU/route-weight semantics between the two GEMM phases.
6. Compare the TurboMind adapter output to
   `ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_f32`.

## Non-Goals

- Enabling TurboMind in the normal replay/server path.
- Packing every model layer at load time.
- Replacing shared FP8 expert, router, attention, or output-head kernels.
- Solving multi-slot route sorting beyond the adapter contract.

## Definition of Done

- [x] Adapter smoke builds with `CUDA_ARCH=sm_70`.
- [x] TurboMind grouped gate/up/down path runs against source MXFP4 bytes.
- [x] DS4 adapter output compares within an explicit FP16-adapter tolerance.
- [x] V100 log is recorded under `logs/from-cluster/`.
- [x] Sprint report records the decision for runtime integration.
- [x] Artifacts are committed.

## Decision Rule

- If the adapter does not match the DS4 source-MXFP4 arena reference within a
  reasonable FP16 tolerance, do not wire it into runtime. Localize whether the
  gap is source packing, row grouping, SwiGLU/weight semantics, or FP16 output.
- If the adapter matches, the next sprint can add an opt-in runtime path with
  load-time TurboMind packing and a scalar fallback.

## Result

`SHIP_ADAPTER_SMOKE`.

The V100 adapter smoke uses real DS4 expert matrix dimensions with a bounded
expert count, packs source MXFP4 bytes through the copied TurboMind C ABI,
groups routed rows by expert, runs grouped gate/up/down GEMMs, applies DS4
SwiGLU and route weights, and compares the final output against the existing
DS4 source-MXFP4 arena reference.

Cluster result:

```text
cuda_v100_turbomind_adapter_smoke: experts=8 routes=6 gate_kpack=0x341321 down_kpack=0x341321 max_abs=0.00129318 rel=0.000258549 bad=0
cuda_v100_turbomind_adapter_smoke: PASS
```

Decision: use TurboMind as the next opt-in runtime adapter path for routed
experts. Keep the existing source-MXFP4 arena implementation as the fallback
until full 256-expert packing, scheduler integration, and sustained throughput
evidence are complete.
