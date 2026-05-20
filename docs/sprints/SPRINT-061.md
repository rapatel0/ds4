# Sprint 061: Batched Shared F8 Expert Path

## Status

Complete.

## Overview

Sprint 060 made routed MXFP4 batching copy-free, but the shared expert path
inside `execute_ffn_delta_batch` still runs one slot at a time. That leaves
multiple F8 matmul and SwiGLU launches per active slot even when the server has
coalesced same-step requests into a tensor batch.

This sprint adds a batched shared expert path for DS4 Flash source
`F8_E4M3_B128` tensors, measures it on V100, and keeps it opt-in because the
measured path did not beat the existing per-slot shared expert schedule.

## Goals

1. Add CUDA arena APIs for batched F8 source matmul and pointer-input
   gate/up/SwiGLU.
2. Resize persistent FFN shared scratch for `[slot x intermediate]` and
   `[slot x hidden]` outputs.
3. Use the new batched shared path from `execute_ffn_delta_batch` behind
   `DS4_V100_BATCH_SHARED_F8=1`.
4. Validate the CUDA primitive against existing per-slot F8 references.
5. Preserve first-token replay correctness and measure sustained 1M decode.
6. Run at least a focused higher-slot tier if the cluster is stable enough.

## Out of Scope

- Tensor-core rewrite of the F8 source matmul.
- Dense attention projection batching.
- MTP draft commit.
- Changing the 8-GPU layer-sharded topology from
  `docs/architecture/DS4-V100-LAYOUT.md`.

## Definition of Done

- Local syntax/object builds pass for touched files.
- `git diff --check` passes.
- V100 `sm_70` build passes for replay and focused smokes.
- Focused CUDA source-dtype smoke validates batched F8 matmul and
  gate/up/SwiGLU against per-slot references.
- Full scheduler/replay still selects first token hex `3136`.
- Sustained decode artifacts are captured for one and two slots; four-slot
  evidence is captured or explicitly documented as blocked.
- Sprint report records before/after throughput and the next bottleneck.
