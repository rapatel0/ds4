# Sprint 494 - Structural Extraction Phase 2A

## Overview

Start Phase 2 by moving the decode-loop orchestration out of the full-layer
smoke and into `engine/decode_loop.cu`.

## Scope

- Create `engine/decode_loop.cu`.
- Move the existing `run_decode_loop(...)` implementation without changing
  behavior, signatures, flags, launch geometry, or promoted serving defaults.
- Include the moved implementation from the original smoke at the same lexical
  point so dependent smoke-local types and helpers remain unchanged for this
  first Phase 2 slice.
- Add the new engine file to the full-layer smoke Makefile dependencies.

## Definition Of Done

- `run_decode_loop(...)` is no longer defined inline in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- The full-layer smoke builds on the V100 CUDA toolchain.
- The rebuilt remote binary starts with `--help`.
- The new engine file is listed as a Makefile dependency.
- No serving behavior or appliance flags are changed.

## Validation Policy

Use the latest promoted serving run as the control leg when the serving shape,
model artifacts, launcher defaults, and promoted flags are unchanged. For this
mechanical extraction slice, run build/startup gates and a changed-candidate
serving invariant check if needed; do not spend a duplicate control run unless
the baseline itself changed.

## Execution Log

- Created `engine/decode_loop.cu`.
- Moved the existing `run_decode_loop(...)` implementation from the full-layer
  smoke into `engine/decode_loop.cu`.
- Included `engine/decode_loop.cu` from the original location so the function
  remains in the same translation unit for this first Phase 2 slice.
- Added `engine/decode_loop.cu` to the full-layer smoke Makefile dependencies.
- Passed `git diff --check`.
- Passed shell syntax checks for the appliance runner scripts.
- Passed remote CUDA build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Confirmed the rebuilt remote binary starts and prints usage with `--help`.
