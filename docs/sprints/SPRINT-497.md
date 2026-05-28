# Sprint 497 - Structural Extraction Phase 2D

## Overview

Move router orchestration out of the full-layer smoke and into bounded
`engine/` files.

## Scope

- Create `engine/router_step.cu`.
- Create `engine/router_plan.cu`.
- Move router dense/rank-major/allreduce logit execution and route-plan upload
  implementations without changing behavior, signatures, flags, launch
  geometry, or promoted serving defaults.
- Include the moved implementations from the original smoke locations.
- Add the new engine files to the full-layer smoke Makefile dependencies.

## Definition Of Done

- The moved router functions are no longer defined inline in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Each new router engine file stays below the 2,000-line target.
- The full-layer smoke builds on the V100 CUDA toolchain.
- The rebuilt remote binary starts with `--help`.
- No serving behavior or appliance flags are changed.

## Validation Policy

Use the latest promoted serving run as control lineage when shape, artifacts,
launcher defaults, and promoted flags are unchanged. This slice is mechanical,
so build/startup gates are the required validation before the next extraction.

## Execution Log

- Created `engine/router_step.cu` and `engine/router_plan.cu`.
- Moved router dense/rank-major/allreduce logit implementations into
  `engine/router_step.cu`.
- Moved route-plan GPU, fixed-capacity, host, and async upload implementations
  into `engine/router_plan.cu`.
- Included the new engine files from the original smoke locations.
- Added the new engine files to the full-layer smoke Makefile dependencies.
- Confirmed both new router engine files are below 2,000 lines.
- Passed `git diff --check`.
- Passed shell syntax checks for the appliance runner scripts.
- Passed remote CUDA build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Confirmed the rebuilt remote binary starts and prints usage with `--help`.
