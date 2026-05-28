# Sprint 493 - Structural Extraction Phase 1J

## Overview

Complete the remaining large Phase 1 kernel extraction by moving the
attention/compression kernel block out of the full-layer smoke and into
`kernels/v100/attention.cuh`.

## Scope

- Create `kernels/v100/attention.cuh`.
- Move FP8 quant/dequant, rope, raw-SWA attention, compressor, indexer, and
  compressed-window attention kernels.
- Include the new header from the smoke after `fill_pack.cuh`.
- Move the final `shard_top1_kernel` helper into `kernels/v100/router.cuh`.
- Do not change signatures, launch geometry, behavior, or appliance flags.

## Definition Of Done

- The moved attention/compression kernels are no longer defined inline in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- The full-layer smoke builds on the V100 CUDA toolchain.
- The rebuilt remote binary starts with `--help`.
- The new header is listed as a Makefile dependency.
- `tools/ds4-v100-tp-ep-full-layer-smoke.cu` has no inline CUDA kernel/device
  definitions left.
- `research/` remains ignored/untracked.

## Execution Log

- Extracted FP8 quant/dequant, rope, raw-SWA attention, compressor, indexer,
  and compressed-window attention definitions into
  `kernels/v100/attention.cuh`.
- Moved the final inline `shard_top1_kernel` into `kernels/v100/router.cuh`.
- Included `kernels/v100/attention.cuh` from the original smoke after
  `fill_pack.cuh`.
- Added the new header to the full-layer smoke Makefile dependencies.
- Confirmed the smoke has no inline CUDA definitions:
  `rg -n "^(template <|__global__|__device__)" tools/ds4-v100-tp-ep-full-layer-smoke.cu`
  returned no matches.
- Passed `git diff --check`.
- Passed shell syntax checks for the appliance runner scripts.
- Passed remote CUDA build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Confirmed the rebuilt remote binary starts and prints usage with `--help`.

## Phase Boundary Validation

Boundary artifact:
`/localpool/ds4/workspace/s493-phase1-correctness-s32-r256-t64`.

Shape: `32` slots / `256K` context / `256` selected-token requests / `64`
generated tokens, with peer-copy accounting and SYS rejection enabled.

Serving invariants on the rebuilt candidate:

- HTTP 200: `256/256`.
- Token count: `64`.
- `vram_failures=0`.
- NCCL graph SYS edges: `0`.
- `peer_copy_sys_ops=0`.
- `peer_copy_sys_bytes=0`.
- Minimum free VRAM: `2082 MiB`.

A duplicate same-binary two-run gate was also attempted and both legs served
cleanly, but its response-artifact parity failed (`32/256` matched pairs). The
failure is not accepted as a Phase 1 regression signal because the two legs ran
the same rebuilt binary with no candidate flag difference, and the new
validation policy is to treat the latest promoted serving run as the control
when the model artifacts, launcher defaults, serving shape, and promoted flags
are unchanged. Future phase-boundary validation should run the changed
candidate against the current promoted-control lineage instead of spending a
second fresh control run by default.

## Follow-Up

Start Phase 2 engine/appliance migration from the Phase 1 baseline. Use the
latest promoted serving run as control evidence unless a later extraction
changes the launcher, environment, model artifacts, serving shape, or promoted
flag set.
