# Sprint 508 - Finish Appliance Target-Shape Extraction

Date: 2026-05-28

## Goal

Close the remaining target-file gaps from the structural extraction plan after
the V100 runtime files moved under `engine/`.

## Changes

- Renamed the appliance parser include from `appliance/options.cu` to
  `appliance/options.h`.
- Split HTTP session/cache admission and microbatch request draining from
  `appliance/http_server.cu` into `appliance/request_scheduler.cu`.
- Added `engine/api.h` with the narrow appliance entry declaration.
- Updated the appliance include spine and Makefile dependencies for the new
  files.

## Validation

- `git diff --check`
- `bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-tp-ep-sustained-bench.sh`
- `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py`
- Local object build:
  `make -B engine/context.o engine/layer_state.o engine/layer_execute.o engine/scheduler.o engine/replay.o engine/mtp.o`
- Remote V100 appliance build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 appliance/ds4-v100-tp-ep-appliance`
- Remote startup:
  `./appliance/ds4-v100-tp-ep-appliance --help`

## Notes

- This is a mechanical extraction/rename; it does not change runtime behavior.
- MTP remains support code only. This sprint does not promote MTP serving.
