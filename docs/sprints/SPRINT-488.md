# Sprint 488 - Structural Extraction Phase 1E

## Overview

Continue the mechanical extraction by moving normalization and stable reduction
helpers into `kernels/v100/`. The smoke remains the owning translation unit,
with no launch or behavior changes.

## Scope

- Create `kernels/v100/norm.cuh`.
- Move plain RMS norm kernels.
- Move weighted RMS norm kernels.
- Move local-head RMS norm helper.
- Move rank-major normalization scale and current-shard stable norm reducers.
- Include the new header from the smoke after `hc_shards.cuh`.
- Do not change kernel signatures, launch geometry, runtime behavior, or
  appliance flags.

## Definition Of Done

- The moved normalization kernels are no longer defined inline in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- The full-layer smoke still builds on the V100 CUDA toolchain.
- The new header is listed as a Makefile dependency for the full-layer smoke.
- `research/` remains ignored/untracked.

## Execution Log

- Extracted plain, weighted, local-head, rank-major, and current-shard
  normalization helpers into `kernels/v100/norm.cuh`.
- Included the header from the original smoke after `hc_shards.cuh`.
- Added the new header to the full-layer smoke Makefile dependencies.
- Passed `git diff --check`.
- Passed shell syntax checks for the appliance runner scripts.
- Passed remote CUDA build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Confirmed the rebuilt remote binary starts and prints usage with `--help`.

## Follow-Up

Continue Phase 1 with attention/compression, route packing, or compose/reduce
kernels.
