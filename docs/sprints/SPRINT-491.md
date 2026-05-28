# Sprint 491 - Structural Extraction Phase 1H

## Overview

Continue Phase 1 by extracting diagnostic parity compare kernels from the
full-layer smoke into `kernels/v100/diagnostics.cuh`.

## Scope

- Create `kernels/v100/diagnostics.cuh`.
- Move shared half-input comparison helpers.
- Move route half-input comparison helpers.
- Include the new header from the smoke after `router.cuh`.
- Do not change signatures, launch geometry, behavior, or appliance flags.

## Definition Of Done

- The moved diagnostic kernels are no longer defined inline in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- The full-layer smoke builds on the V100 CUDA toolchain.
- The rebuilt remote binary starts with `--help`.
- The new header is listed as a Makefile dependency.
- `research/` remains ignored/untracked.

## Execution Log

- Extracted half-input diagnostic compare kernels into
  `kernels/v100/diagnostics.cuh`.
- Included the header from the original smoke after `router.cuh`.
- Added the new header to the full-layer smoke Makefile dependencies.
- Passed `git diff --check`.
- Passed shell syntax checks for the appliance runner scripts.
- Passed remote CUDA build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Confirmed the rebuilt remote binary starts and prints usage with `--help`.

## Follow-Up

Continue Phase 1 with attention/compression or fill/pack kernels.
