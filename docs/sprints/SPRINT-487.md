# Sprint 487 - Structural Extraction Phase 1D

## Overview

Continue the mechanical extraction by moving hidden-current shard movement and
dense input packing helpers into `kernels/v100/`. This keeps the smoke as the
only translation unit and preserves all existing launches.

## Scope

- Create `kernels/v100/hc_shards.cuh`.
- Move hidden-current shard gather/seed/synthetic kernels.
- Move current-shard gather and rank/slot-major conversion kernels.
- Move dense shard gather and half input fill helpers.
- Move proxy HC expansion and initial HC seed helpers.
- Include the new header from the smoke after `hc_mix.cuh`.
- Do not change kernel signatures, launch geometry, runtime behavior, or
  appliance flags.

## Definition Of Done

- The moved HC shard movement kernels are no longer defined inline in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- The full-layer smoke still builds on the V100 CUDA toolchain.
- The new header is listed as a Makefile dependency for the full-layer smoke.
- `research/` remains ignored/untracked.

## Execution Log

- Extracted HC shard gather/seed/synthetic helpers into
  `kernels/v100/hc_shards.cuh`.
- Extracted current-shard gather, rank/slot-major conversion, dense-shard
  gather, and half input fill helpers.
- Extracted proxy HC expansion and initial HC seed helpers.
- Included the header from the original smoke after `hc_mix.cuh`.
- Added the new header to the full-layer smoke Makefile dependencies.
- Passed `git diff --check`.
- Passed shell syntax checks for the appliance runner scripts.
- Passed remote CUDA build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Confirmed the rebuilt remote binary starts and prints usage with `--help`.

## Follow-Up

Continue Phase 1 with normalization, attention/compression, route packing, or
compose/reduce kernels.
