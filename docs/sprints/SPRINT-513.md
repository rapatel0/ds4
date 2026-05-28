# Sprint 513 - Remove Legacy Prefix From Smoke Source Names

Date: 2026-05-28

## Goal

Finish the smoke-source layout cleanup by removing the old `ds4-v100-` prefix
from filenames under `smokes/`. The directory now carries the source
namespace; built tool names remain stable.

## Changes

- Renamed all `smokes/ds4-v100-*` source files to prefix-free names.
- Updated Makefile dependencies and compile commands to use the new source
  paths while preserving existing `tools/ds4-v100-*` binary targets.
- Updated intra-smoke includes in the MTP forward smoke.

## Validation

- `git diff --check`
- Local CPU-buildable smoke/gate target build:
  `make -B tools/ds4-v100-tp-ep-layer-descriptor-smoke tools/ds4-v100-tp8-kv-shard-smoke tools/ds4-v100-context-smoke tools/ds4-v100-layer-descriptor-gate tools/ds4-v100-mtp-sidecar-gate`
- Remote V100 build of representative renamed smoke/workbench targets,
  including TP/EP layer smoke, TP runtime smoke, collective smokes, MTP
  verify/replay, and selected workbenches.
- Remote V100 appliance build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 appliance/ds4-v100-tp-ep-appliance`
- Remote startup:
  `./appliance/ds4-v100-tp-ep-appliance --help`

## Notes

- This changes source filenames only. Existing operator-facing executable and
  script names are intentionally unchanged.
- MTP remains support/sidecar work and is not promoted to serving here.
