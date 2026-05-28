# Sprint 495 - Structural Extraction Phase 2B

## Overview

Move the HC-current sublayer orchestration out of the full-layer smoke and into
`engine/hc_current.cu`.

## Scope

- Create `engine/hc_current.cu`.
- Move the existing `run_shared_hc_current_input(...)` implementation without
  changing behavior, signatures, flags, launch geometry, or promoted serving
  defaults.
- Include the moved implementation from the original smoke location so this
  first sublayer extraction stays in the same translation unit.
- Add the new engine file to the full-layer smoke Makefile dependencies.

## Definition Of Done

- `run_shared_hc_current_input(...)` is no longer defined inline in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- The full-layer smoke builds on the V100 CUDA toolchain.
- The rebuilt remote binary starts with `--help`.
- The new engine file is listed as a Makefile dependency.
- No serving behavior or appliance flags are changed.

## Validation Policy

Use the latest promoted serving run as control lineage when shape, artifacts,
launcher defaults, and promoted flags are unchanged. This slice is mechanical,
so build/startup gates are the required validation before the next extraction.

## Execution Log

- Created `engine/hc_current.cu`.
- Moved the existing `run_shared_hc_current_input(...)` implementation from the
  full-layer smoke into `engine/hc_current.cu`.
- Included `engine/hc_current.cu` from the original location so the function
  remains in the same translation unit for this sublayer slice.
- Added `engine/hc_current.cu` to the full-layer smoke Makefile dependencies.
- Passed `git diff --check`.
- Passed shell syntax checks for the appliance runner scripts.
- Passed remote CUDA build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Confirmed the rebuilt remote binary starts and prints usage with `--help`.
