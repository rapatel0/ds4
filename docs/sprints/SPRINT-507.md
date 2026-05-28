# Sprint 507 - Move Remaining V100 Support Files Under Engine

Date: 2026-05-28

## Goal

Complete the root-file migration called out by the structural extraction plan:
the remaining tracked `ds4_v100_*` support modules should no longer live at
repo root.

## Changes

- Moved base V100 context support to `engine/context.{c,h}` and
  `engine/context_cuda.cu`.
- Moved V100 layer state, layer execution, scheduler, replay, and MTP sidecar
  support to `engine/` with shorter file names.
- Updated Makefile object variables/rules and dependent tools/tests to include
  the new `engine/` paths.
- Removed stale root `ds4_v100_*.o` build outputs from the local workspace.

## Validation

- `git diff --check`
- Local object build:
  `make -B engine/context.o engine/layer_state.o engine/layer_execute.o engine/scheduler.o engine/replay.o engine/mtp.o`
- Remote V100 object build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 engine/context.o engine/layer_state.o engine/layer_execute.o engine/scheduler.o engine/replay.o engine/mtp.o`
- Remote V100 appliance build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 appliance/ds4-v100-tp-ep-appliance`
- Remote startup:
  `./appliance/ds4-v100-tp-ep-appliance --help`
- Local and remote root audits found no tracked/stale root `ds4_v100*` source
  files after cleanup.

## Notes

- MTP remains a support/sidecar module here; this does not promote MTP serving
  in the TP/EP appliance.
- Historical docs retain old filenames as sprint lineage.
