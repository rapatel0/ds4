# Sprint 056: Grouped MXFP4 Selected-Route Execution

## Status

Complete.

## Overview

Sprint 056 stops one-route-at-a-time routed expert cleanup and groups all
selected MXFP4 routes inside the main FFN executor. Sprint 054 fused routed
gate/up/SwiGLU per route, and Sprint 055 fused routed down projection with
accumulation per route. This sprint collapses the six selected routes into a
two-kernel primitive:

1. grouped gate/up/SwiGLU for all selected experts into route-local mid scratch
2. grouped down projection plus route sum into the routed FFN output

The goal is a larger hot-path primitive that preserves source MXFP4 layout and
selected-token correctness while reducing per-token launch overhead.

## Goals

1. Add a grouped MXFP4 routed FFN primitive that consumes router-selected expert
   ids and weights directly.
2. Wire the grouped primitive into `execute_ffn_delta`.
3. Extend the focused MXFP4 smoke to compare grouped-route output against the
   established per-route path.
4. Re-run real 8-GPU selected-token correctness and sustained decode at 1M
   context for 1-slot and 2-slot cases.
5. Preserve all source-model quality assumptions: source MXFP4 experts remain
   device resident, with dequantization only inside bounded CUDA kernels.

## Out of Scope

- True tensor-batched layer execution across slots.
- Persistent expert kernels.
- Shared F8 expert fusion.
- Attention projection fusion.
- MTP draft commit.
- INT8/INT4 expert repacking.

## Implementation Notes

- `ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_f32` validates the arena ranges,
  expert strides, row strides, selected-route tensors, and scratch/output
  tensor sizes before launching.
- The grouped gate/up/SwiGLU kernel uses grid dimensions `[mid, routes]`.
- The grouped down kernel uses one block per hidden row and loops selected
  routes inside the kernel, summing directly into the routed FFN output.
- The first implementation assumes routed gate and up expert tensors have the
  same MXFP4 row/expert layout. DS4 Flash satisfies this; `execute_ffn_delta`
  checks it before dispatch.
- The primitive still uses scalar source-MXFP4 decode/reduction rather than a
  true Volta HMMA or integer-tensor-core grouped GEMM. It is a launch/dispatch
  reduction step, not the final throughput architecture.

## Definition of Done

- `cc -fsyntax-only -I. ds4_v100_layer_execute.c` passes.
- `cc -fsyntax-only -I. tests/cuda_v100_mxfp4_moe_smoke.c` passes.
- `make ds4_v100_layer_execute.o tests/cuda_v100_mxfp4_moe_smoke.o` passes
  locally.
- `CUDA_ARCH=sm_70 make tests/cuda_v100_mxfp4_moe_smoke tools/ds4-v100-replay`
  passes on the V100 pod.
- `CUDA_VISIBLE_DEVICES=0 ./tests/cuda_v100_mxfp4_moe_smoke` passes on V100.
- Real 8-GPU replay still selects first token hex `3136`.
- Sustained decode artifacts are captured under
  `logs/from-cluster/sprint056-grouped-mxfp4-routes`.
