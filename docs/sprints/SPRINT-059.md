# Sprint 059: Persistent Layer Batch Scratch

## Status

Complete.

## Overview

Sprint 057 proved the first batched layer slice was correct but slower than the
default path. Sprint 058 removed replay-only router readback synchronization
and improved two-slot throughput by about `1.15%`, leaving the main utilization
gap unchanged.

This sprint removes the next obvious runtime overhead from the batched layer
path: per-layer CUDA tensor allocation/free. The goal is not to rewrite the MoE
math yet. It is to make the batch layer path a fairer baseline by reusing
scheduler-owned scratch across layers and decode steps.

## Goals

1. Add a reusable layer batch scratch object sized for
   `DS4_V100_LAYER_MAX_BATCH`.
2. Use that scratch in `ds4_v100_layer_execute_hc_decode_batch` and
   `execute_ffn_delta_batch` when the scheduler provides it.
3. Preserve the existing allocation/free fallback for direct tests and callers.
4. Keep the batched layer path gated until V100 benchmark evidence shows it is
   faster than the default path.
5. If benchmark evidence is positive, enable multi-slot layer batching by
   default with an environment escape hatch.

## Out of Scope

- Changing MXFP4 numerical kernels.
- Removing the per-slot input copy into the current contiguous batch tensor.
- Enabling the batched layer path by default without a measured speedup.
- MTP draft commit.

## Implementation Notes

- The scheduler owns one scratch object per stage/GPU.
- Scratch allocation happens lazily on the stage's selected CUDA device.
- Scratch is reusable across layers with the same hidden/intermediate
  dimensions and route count; if dimensions ever change, it is freed and
  reallocated.
- Direct layer tests still pass `NULL` scratch and exercise the old allocation
  path unless they explicitly opt in.
- Multi-slot layer batching is now enabled by default. Operators can disable it
  with `DS4_V100_BATCH_LAYER_FFN=0`, `off`, or `false`.

## Definition of Done

- `cc -fsyntax-only -I. ds4_v100_layer_execute.c` passes.
- `cc -fsyntax-only -I. ds4_v100_scheduler.c` passes.
- Local object builds pass for touched C files.
- `git diff --check` passes.
- V100 `sm_70` build passes for `tools/ds4-v100-replay` and focused smokes.
- Real 8-GPU replay still selects first token hex `3136`.
- Sustained decode artifacts compare the pre-flip default path, the
  scratch-backed batch path, and the final default-batched path.
