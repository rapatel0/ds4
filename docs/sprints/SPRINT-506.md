# Sprint 506 - Move TP Runtime Under Engine

Date: 2026-05-28

## Goal

Continue the structural migration by removing the TP runtime library from the
repo root and placing it under the engine boundary used by the TP/EP appliance.

## Changes

- Renamed `ds4_v100_tp_runtime.cu` to `engine/tp_runtime.cu`.
- Renamed `ds4_v100_tp_runtime.h` to `engine/tp_runtime.h`.
- Updated the appliance, Makefile targets, and remaining TP runtime smokes to
  include and compile the runtime from `engine/`.

## Validation

- `git diff --check`
- `bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-tp-ep-sustained-bench.sh`
- `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py`
- Remote V100 build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 appliance/ds4-v100-tp-ep-appliance`
- Remote startup:
  `./appliance/ds4-v100-tp-ep-appliance --help`
- Remote root audit confirmed `ds4_v100_tp_runtime.{cu,h}` are absent.
