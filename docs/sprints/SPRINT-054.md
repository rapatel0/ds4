# Sprint 054: Fused MXFP4 Routed Gate-Up Kernel

## Status

Complete.

## Overview

Sprint 054 starts the hot-path occupancy work called out by Sprint 053. The
request loop can now batch same-length requests, but the layer executor still
spends most time in single-slot routed FFN and attention kernels. This sprint
targets the routed MXFP4 expert path because it is already in the source-model
main decode path and it currently launches gate matmul, up matmul, SwiGLU,
down matmul, and add separately for each selected route.

## Goals

1. Add a fused MXFP4 gate+up+SwiGLU primitive for one routed expert.
2. Wire the primitive into `execute_ffn_delta` without changing source layout
   or selected-token semantics.
3. Keep the existing standalone MXFP4 matmul available for down projection,
   tests, diagnostics, and fallback work.
4. Add focused smoke coverage that compares the fused result against the
   previous separate gate/up/SwiGLU sequence.
5. Run V100 correctness and sustained decode comparisons against Sprint 053.

## Scope

- `ds4_cuda.cu` / `ds4_gpu.h`:
  - add `ds4_gpu_arena_mxfp4_pair_swiglu_f32`;
  - preserve the GGML MXFP4 low-half/high-half nibble semantics;
  - keep accumulation and clamp/SwiGLU math equivalent to the previous path.
- `ds4_v100_layer_execute.c`:
  - replace routed gate+up+SwiGLU launches with the fused primitive.
- `tests/cuda_v100_mxfp4_moe_smoke.c`:
  - validate fused output against the previous separate operations.

## Out of Scope

- Fusing routed down projection and route accumulation.
- Grouping all six routed experts into one launch.
- Shared F8 expert fusion.
- Layer-executor batching across slots.
- MTP changes.

## Definition of Done

- `cc -fsyntax-only -I. ds4_v100_layer_execute.c` passes.
- `cc -fsyntax-only -I. tests/cuda_v100_mxfp4_moe_smoke.c` passes.
- `make ds4_v100_layer_execute.o tests/cuda_v100_mxfp4_moe_smoke.o` passes
  locally.
- `CUDA_ARCH=sm_70 make tests/cuda_v100_mxfp4_moe_smoke tools/ds4-v100-replay`
  passes on the V100 pod.
- `CUDA_VISIBLE_DEVICES=0 ./tests/cuda_v100_mxfp4_moe_smoke` passes on V100.
- Real 8-GPU replay still selects first token hex `3136`.
- Sustained decode comparison artifacts are captured under
  `logs/from-cluster/sprint054-fused-mxfp4`.
- The report records whether throughput improved and whether the improvement
  is large enough to change the roadmap.
