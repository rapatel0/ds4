# Sprint 510 - Bound Legacy Engine C Support Units

Date: 2026-05-28

## Goal

Apply the structural extraction file-size discipline to the moved legacy V100 C
support units without changing their exported ABI or build targets.

## Changes

- Split `engine/layer_execute.c` into a thin include spine plus bounded
  private implementation slices:
  `engine/layer_execute_core.inc`, `engine/layer_execute_attention.inc`, and
  `engine/layer_execute_ffn.inc`.
- Split `engine/scheduler.c` into a thin include spine plus
  `engine/scheduler_core.inc` and `engine/scheduler_snapshot_decode.inc`.
- Split `engine/replay.c` into a thin include spine plus
  `engine/replay_core.inc` and `engine/replay_step_pipeline.inc`.
- Updated Makefile dependencies so changes to the private slices rebuild the
  owning object files.

## Validation

- `git diff --check`
- Local object build:
  `make -B engine/layer_execute.o engine/scheduler.o engine/replay.o engine/mtp.o`
- Remote V100 object build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 engine/layer_execute.o engine/scheduler.o engine/replay.o engine/mtp.o`
- Remote V100 appliance build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 appliance/ds4-v100-tp-ep-appliance`
- Remote startup:
  `./appliance/ds4-v100-tp-ep-appliance --help`

## Notes

- The split is mechanical: each original `.c` file remains the owning
  translation unit, so external symbols and link behavior are unchanged.
- This does not implement or promote MTP serving.
