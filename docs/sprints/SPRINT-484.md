# Sprint 484 - Structural Extraction Phase 1A

## Overview

Start the structural extraction plan with a small, mechanical kernel move from
`tools/ds4-v100-tp-ep-full-layer-smoke.cu` into `kernels/v100/`. The objective
is not to redesign the appliance yet; it is to prove the extraction pattern
with a low-risk group and keep the serving binary behavior identical.

## Scope

- Create `kernels/v100/common.cuh`.
- Move common utility kernels and device helpers out of the smoke:
  checksum/copy kernels, FP8 decode helpers, warp/block reductions, and FP16
  saturation helper.
- Include the new header from the existing smoke inside the same anonymous
  namespace.
- Do not rename kernels, change signatures, alter dispatch, or change launcher
  behavior.

## Definition Of Done

- The moved kernels are no longer defined inline in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- `tools/ds4-v100-tp-ep-full-layer-smoke.cu` still builds on the V100 CUDA
  toolchain.
- Existing Python and shell checks touched by the prior profiling wrapper work
  still pass.
- No new appliance flags are introduced.
- `research/` remains ignored/untracked.

## Execution Log

- Extracted the common utility kernel/helper group into `kernels/v100/common.cuh`.
- Included the header from the original smoke inside the existing anonymous
  namespace.
- Added the new header to the full-layer smoke Makefile dependencies.
- Passed `git diff --check`.
- Passed shell syntax checks for the appliance runner scripts.
- Passed remote CUDA build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.

## Follow-Up

Continue Phase 1 by extracting bounded domain groups one at a time:
norm/fill, attention/compression, route/compose, router/GPU route plan, and
diagnostics. Each group should be a separate sprint or separate commit with
the same mechanical extraction discipline.
