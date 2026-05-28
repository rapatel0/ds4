# Sprint 496 - Structural Extraction Phase 2C

## Overview

Move the true-attention sublayer orchestration out of the full-layer smoke and
into bounded `engine/` files.

## Scope

- Create `engine/attention_projection.cu`.
- Create `engine/compressed_kv_step.cu`.
- Create `engine/attention_read.cu`.
- Create `engine/attention_output.cu`.
- Move the existing attention projection, compressed-KV/state, typed-history /
  raw-read, and attention-output functions without changing behavior,
  signatures, flags, launch geometry, or promoted serving defaults.
- Include the moved implementations from the original smoke location in the
  original order.
- Add the new engine files to the full-layer smoke Makefile dependencies.

## Definition Of Done

- The moved true-attention functions are no longer defined inline in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Each new engine attention file stays below the 2,000-line target.
- The full-layer smoke builds on the V100 CUDA toolchain.
- The rebuilt remote binary starts with `--help`.
- No serving behavior or appliance flags are changed.

## Validation Policy

Use the latest promoted serving run as control lineage when shape, artifacts,
launcher defaults, and promoted flags are unchanged. This slice is mechanical,
so build/startup gates are the required validation before the next extraction.

## Execution Log

- Created `engine/attention_projection.cu`,
  `engine/compressed_kv_step.cu`, `engine/attention_read.cu`, and
  `engine/attention_output.cu`.
- Moved the existing attention projection, compressed-KV/reference/state,
  typed-history/raw-read/raw-window, and attention-output implementations from
  the full-layer smoke into the new engine files.
- Included the new engine files from the original smoke location in the
  original order.
- Added the new engine files to the full-layer smoke Makefile dependencies.
- Confirmed each new attention engine file is below 2,000 lines.
- Passed `git diff --check`.
- Passed shell syntax checks for the appliance runner scripts.
- Passed remote CUDA build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Confirmed the rebuilt remote binary starts and prints usage with `--help`.
