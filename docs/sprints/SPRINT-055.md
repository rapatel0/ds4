# Sprint 055: Fused MXFP4 Routed Down Accumulation

## Status

Complete.

## Overview

Sprint 055 continues the routed expert hot-path cleanup started in Sprint 054.
After fusing routed gate+up+SwiGLU, each selected route still launched a
separate MXFP4 down projection and vector add. This sprint fuses that down
projection with route accumulation while keeping the source MXFP4 layout and
the existing per-route scheduler structure.

## Goals

1. Add an MXFP4 matmul+add primitive for routed down projection.
2. Wire the primitive into `execute_ffn_delta`.
3. Extend the focused MXFP4 smoke to compare fused down accumulation against
   the previous separate down matmul plus add path.
4. Re-run real 8-GPU selected-token correctness and sustained decode.

## Out of Scope

- Grouping all six selected experts into one launch.
- Batching layer execution across slots.
- Shared F8 expert fusion.
- Attention projection fusion.
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
- Sustained decode artifacts are captured under
  `logs/from-cluster/sprint055-mxfp4-down-accum`.
