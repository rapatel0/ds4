# Sprint 490 - Structural Extraction Phase 1G

## Overview

Continue Phase 1 by extracting router logits, router selection, and GPU route
plan kernels from the full-layer smoke into `kernels/v100/`.

## Scope

- Create `kernels/v100/router.cuh`.
- Move column-major dense helper used by router paths.
- Move rank-major/allreduce router logits kernels.
- Move top-k and hash-fast router selection kernels.
- Move GPU route plan count/prefix/fill helpers and route-plan audit kernel.
- Include the new header from the smoke after `compose.cuh`.
- Do not change signatures, launch geometry, behavior, or appliance flags.

## Definition Of Done

- The moved router/route-plan kernels are no longer defined inline in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- The full-layer smoke builds on the V100 CUDA toolchain.
- The rebuilt remote binary starts with `--help`.
- The new header is listed as a Makefile dependency.
- `research/` remains ignored/untracked.

## Execution Log

- Extracted router logits, selection, route-plan, and route-plan audit kernels
  into `kernels/v100/router.cuh`.
- Included the header from the original smoke after `compose.cuh`.
- Added the new header to the full-layer smoke Makefile dependencies.
- Passed `git diff --check`.
- Passed shell syntax checks for the appliance runner scripts.
- Passed remote CUDA build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Confirmed the rebuilt remote binary starts and prints usage with `--help`.

## Follow-Up

Continue Phase 1 with attention/compression, diagnostics, or fill/pack kernels.
