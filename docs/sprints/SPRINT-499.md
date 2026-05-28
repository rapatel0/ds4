# Sprint 499 - Structural Extraction Phase 2F

## Overview

Move the remaining TP/EP layer and token-major orchestration out of the
full-layer smoke and into bounded `engine/` files.

## Scope

- Create `engine/hc_final.cu`.
- Create `engine/post_attention_ffn.cu`.
- Create `engine/layer_decode.cu`.
- Create `engine/layer_runner.cu`.
- Create `engine/token_major_loop.cu`.
- Move final-HC expansion, post-attention FFN input, resident layer decode,
  layer runner, and token-major serving-loop implementations without changing
  behavior, signatures, flags, launch geometry, or promoted serving defaults.
- Include the moved implementations from the original smoke locations.
- Add the new engine files to the full-layer smoke Makefile dependencies.

## Definition Of Done

- The moved orchestration functions are no longer defined inline in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Each new engine file stays below the 2,000-line target.
- The full-layer smoke builds on the V100 CUDA toolchain.
- The rebuilt remote binary starts with `--help`.
- No serving behavior or appliance flags are changed.

## Validation Policy

Use the latest promoted serving run as control lineage when shape, artifacts,
launcher defaults, and promoted flags are unchanged. This slice is mechanical,
so build/startup gates are the required validation before the next extraction.

## Execution Log

- Created `engine/hc_final.cu`, `engine/post_attention_ffn.cu`,
  `engine/layer_decode.cu`, `engine/layer_runner.cu`, and
  `engine/token_major_loop.cu`.
- Moved final-HC expansion, post-attention FFN input, resident layer decode,
  layer runner, and token-major serving-loop implementations into the new
  engine files.
- Included the new engine files from the original smoke locations.
- Added the new engine files to the full-layer smoke Makefile dependencies.
- Confirmed each new engine file is below 2,000 lines.
- Confirmed MTP is not implemented inside this TP/EP full-layer smoke; MTP
  remains separate unfinished work in the existing MTP runtime/smoke files.
- Passed `git diff --check`.
- Passed shell syntax checks for the appliance runner scripts.
- Passed remote CUDA build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Confirmed the rebuilt remote binary starts and prints usage with `--help`.
