# Sprint 486 - Structural Extraction Phase 1C

## Overview

Continue the structural extraction by moving the hidden-current mix/reduction
kernel group out of `tools/ds4-v100-tp-ep-full-layer-smoke.cu`. This is still a
single compile unit include and does not change call sites or launch behavior.

## Scope

- Create `kernels/v100/hc_mix.cuh`.
- Move hidden-current output weighting and weighted-sum kernels.
- Move the HC stable local reduction kernels and reduced mix scaling/split
  kernels.
- Move the shared `hc4_split_one_dev` helper into the new header.
- Include the new header from the smoke after `dense.cuh`.
- Do not change kernel signatures, launches, behavior, or appliance flags.

## Definition Of Done

- The moved HC mix/reduction kernels and helper are no longer defined inline in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- The full-layer smoke still builds on the V100 CUDA toolchain.
- The new header is listed as a Makefile dependency for the full-layer smoke.
- `research/` remains ignored/untracked.

## Execution Log

- Extracted the HC mix/reduction group into `kernels/v100/hc_mix.cuh`.
- Moved `hc4_split_one_dev` into the new header so all existing device callers
  share one definition in the same compile unit.
- Included the header from the original smoke after `dense.cuh`.
- Added the new header to the full-layer smoke Makefile dependencies.
- Passed `git diff --check`.
- Passed shell syntax checks for the appliance runner scripts.
- Passed local CPU build with `make cpu`.
- `make test` is blocked in this checkout because `ds4flash.gguf` is absent.
- Passed remote CUDA build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Confirmed the rebuilt remote binary starts and prints usage with `--help`.

## Follow-Up

Continue Phase 1 with hidden-current shard movement, normalization, attention
and compression kernels, route packing, or compose/reduce.
