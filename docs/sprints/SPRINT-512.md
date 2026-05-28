# Sprint 512 - Move One-Off Smoke Sources Out of Tools

Date: 2026-05-28

## Goal

Align one-off experiment source layout with the structural extraction plan:
smoke, gate, and workbench source files should live under `smokes/`, while
operator-facing scripts and built binaries remain under `tools/`.

## Changes

- Added the root `smokes/` source directory.
- Moved 28 one-off smoke/gate/workbench source files from `tools/` to
  `smokes/`.
- Updated Makefile dependencies and compile commands so existing targets still
  produce `tools/...` binaries.
- Updated the moved MTP verify smoke include to reference the retained common
  helper header under `tools/`.

## Validation

- `git diff --check`
- Local CPU-buildable smoke/gate target build:
  `make -B tools/ds4-v100-tp-ep-layer-descriptor-smoke tools/ds4-v100-tp8-kv-shard-smoke tools/ds4-v100-context-smoke tools/ds4-v100-layer-descriptor-gate tools/ds4-v100-mtp-sidecar-gate`
- Remote V100 build of representative moved CUDA/C smoke targets, including
  TP/EP layer smoke, TP runtime smoke, collective smokes, MTP smokes,
  MTP verify/replay, and selected workbenches.
- Remote V100 appliance build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 appliance/ds4-v100-tp-ep-appliance`
- Remote startup:
  `./appliance/ds4-v100-tp-ep-appliance --help`
- Local and remote audits show zero `tools/*smoke.c`, `tools/*smoke.cu`,
  `tools/*workbench.cu`, or `tools/*gate.c` source files remaining.

## Notes

- Shell/Python production gate orchestrators remain under `tools/`.
- Built binary names remain unchanged for existing scripts.
- MTP is still not promoted to serving by this move.
