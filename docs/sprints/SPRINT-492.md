# Sprint 492 - Structural Extraction Phase 1I

## Overview

Continue Phase 1 by extracting fill/pack kernels from the full-layer smoke into
`kernels/v100/fill_pack.cuh`.

## Scope

- Create `kernels/v100/fill_pack.cuh`.
- Move route packing kernels.
- Move dense/current fill helpers.
- Move shared/routed FFN fill helpers.
- Move fused HC current fill/pack helper and HC expand helper.
- Include the new header from the smoke after `diagnostics.cuh`.
- Do not change signatures, launch geometry, behavior, or appliance flags.

## Definition Of Done

- The moved fill/pack kernels are no longer defined inline in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- The full-layer smoke builds on the V100 CUDA toolchain.
- The rebuilt remote binary starts with `--help`.
- The new header is listed as a Makefile dependency.
- `research/` remains ignored/untracked.

## Execution Log

- Extracted route packing, dense/current fill, shared/routed FFN fill, fused
  HC current fill/pack, and HC expand helpers into `kernels/v100/fill_pack.cuh`.
- Included the header from the original smoke after `diagnostics.cuh`.
- Added the new header to the full-layer smoke Makefile dependencies.
- Passed `git diff --check`.
- Passed shell syntax checks for the appliance runner scripts.
- Passed remote CUDA build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Confirmed the rebuilt remote binary starts and prints usage with `--help`.

## Follow-Up

Continue Phase 1 with attention/compression kernels.
