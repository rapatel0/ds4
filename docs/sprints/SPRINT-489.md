# Sprint 489 - Structural Extraction Phase 1F

## Overview

Continue Phase 1 by extracting compose/reduce utility kernels from the
full-layer smoke into `kernels/v100/`. This remains mechanical: the smoke owns
the translation unit and all launches stay unchanged.

## Scope

- Create `kernels/v100/compose.cuh`.
- Move zero/clamp/add/cast helpers used by compose paths.
- Move EP destination-shard reduce/pack helpers.
- Move next-hidden compose kernels and the compact MoE sum device helper.
- Include the new header from the smoke after `norm.cuh`.
- Do not change signatures, launch geometry, behavior, or appliance flags.

## Definition Of Done

- The moved compose/reduce kernels are no longer defined inline in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- The full-layer smoke builds on the V100 CUDA toolchain.
- The rebuilt remote binary starts with `--help`.
- The new header is listed as a Makefile dependency.
- `research/` remains ignored/untracked.

## Execution Log

- Extracted compose/reduce helpers into `kernels/v100/compose.cuh`.
- Included the header from the original smoke after `norm.cuh`.
- Added the new header to the full-layer smoke Makefile dependencies.
- Passed `git diff --check`.
- Passed shell syntax checks for the appliance runner scripts.
- Passed remote CUDA build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Confirmed the rebuilt remote binary starts and prints usage with `--help`.

## Follow-Up

Continue Phase 1 with attention/compression, route/router, diagnostics, or
fill/pack kernels.
