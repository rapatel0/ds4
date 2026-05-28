# Sprint 498 - Structural Extraction Phase 2E

## Overview

Move EP/compose orchestration out of the full-layer smoke and into bounded
`engine/` files.

## Scope

- Create `engine/ep_dense.cu`.
- Create `engine/ep_executor.cu`.
- Create `engine/ep_compose.cu`.
- Move FFN input fill, resident dense launch wrapper, routed gate/down executor
  wrappers, and next-hidden compose implementations without changing behavior,
  signatures, flags, launch geometry, or promoted serving defaults.
- Include the moved implementations from the original smoke locations.
- Add the new engine files to the full-layer smoke Makefile dependencies.

## Definition Of Done

- The moved EP/compose functions are no longer defined inline in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Each new EP/compose engine file stays below the 2,000-line target.
- The full-layer smoke builds on the V100 CUDA toolchain.
- The rebuilt remote binary starts with `--help`.
- No serving behavior or appliance flags are changed.

## Validation Policy

Use the latest promoted serving run as control lineage when shape, artifacts,
launcher defaults, and promoted flags are unchanged. This slice is mechanical,
so build/startup gates are the required validation before the next extraction.

## Execution Log

- Created `engine/ep_dense.cu`, `engine/ep_executor.cu`, and
  `engine/ep_compose.cu`.
- Moved FFN input fill, resident dense launch wrapper, routed gate/down
  executor wrappers, and next-hidden compose implementations into the new
  engine files.
- Included the new engine files from the original smoke locations.
- Added the new engine files to the full-layer smoke Makefile dependencies.
- Confirmed each new EP/compose engine file is below 2,000 lines.
- Passed `git diff --check`.
- Passed shell syntax checks for the appliance runner scripts.
- Passed remote CUDA build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Confirmed the rebuilt remote binary starts and prints usage with `--help`.
