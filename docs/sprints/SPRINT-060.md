# Sprint 060: Pointer-Input Routed FFN Batch

## Status

Complete.

## Overview

Sprint 059 made the multi-slot layer batch path default by removing per-layer
allocation churn with persistent scratch. The remaining obvious staging cost is
the copy from per-slot FFN input tensors into a contiguous `input_batch_t` before
the routed MXFP4 batch primitive.

This sprint removes that copy for the routed path by adding a pointer-input
variant of the grouped MXFP4 routed FFN primitive.

## Goals

1. Add a pointer-input CUDA API for batched routed MXFP4 SwiGLU/down-sum.
2. Validate separate per-slot input tensors in the focused MXFP4 smoke.
3. Use the pointer-input API from `execute_ffn_delta_batch`.
4. Remove `ffn_input_batch` from persistent batch scratch.
5. Benchmark the default two-slot sustained path after the copy is removed.

## Out of Scope

- Tensor-core or DP4A rewrite of the MXFP4 math.
- Shared expert batching.
- Higher slot-tier benchmarking beyond the focused one/two-slot gate.
- MTP draft commit.

## Definition of Done

- Local syntax and object builds pass for touched files.
- `git diff --check` passes.
- V100 `sm_70` build passes for replay and focused smokes.
- Focused V100 MXFP4 smoke validates pointer-input batched routed FFN.
- Full replay still selects first token hex `3136`.
- Sustained decode artifacts are captured and compared against Sprint 059.
