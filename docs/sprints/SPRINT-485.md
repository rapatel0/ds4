# Sprint 485 - Structural Extraction Phase 1B

## Overview

Continue the mechanical kernel extraction from
`tools/ds4-v100-tp-ep-full-layer-smoke.cu` by moving the dense/format utility
group into `kernels/v100/`. This keeps the same compile unit and call sites,
but removes another bounded chunk from the monolithic smoke.

## Scope

- Create `kernels/v100/dense.cuh`.
- Move FP8 block-128 dense kernels and FP8-to-half conversion.
- Move BF16 decode, BF16-to-half conversion, and BF16 dense kernels.
- Move the simple F32 dense kernel.
- Include the new header from the existing smoke after `common.cuh`.
- Do not rename kernels, change signatures, alter launch geometry, or add new
  appliance flags.

## Definition Of Done

- The moved dense/format kernels are no longer defined inline in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- The full-layer smoke still builds on the V100 CUDA toolchain.
- The new header is listed as a Makefile dependency for the full-layer smoke.
- `research/` remains ignored/untracked.

## Execution Log

- Extracted the FP8/BF16/F32 dense and format conversion group into
  `kernels/v100/dense.cuh`.
- Included the header from the original smoke after `common.cuh`.
- Added the new header to the full-layer smoke Makefile dependencies.
- Passed `git diff --check`.
- Passed shell syntax checks for the appliance runner scripts.
- Passed remote CUDA build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.

## Follow-Up

Continue Phase 1 with another bounded group: hidden-current shard movement,
normalization, attention/compression, route packing, or compose/reduce.
