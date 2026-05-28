# Sprint 511 - Drop Legacy V100 Prefix From Code Symbols

Date: 2026-05-28

## Goal

Complete the structural extraction plan's symbol-naming cleanup for the moved
V100 runtime code: exported/runtime identifiers should use the shorter `ds4_`
prefix instead of `ds4_v100_`.

## Changes

- Mechanically renamed lowercase code identifiers from `ds4_v100_*` to
  `ds4_*` across `engine/`, `appliance/`, `kernels/v100/`, `tools/`, and
  `tests/`.
- Renamed the sidecar info accessor to `ds4_mtp_sidecar_get_info()` to avoid
  colliding with the existing `ds4_mtp_sidecar_info` type.
- Updated tests and tools to compile against the renamed runtime API.
- Added missing `ds4_turbomind_pack.o` link dependencies for the MTP logits
  and forward smoke tools, which link `engine/context.o`.

## Validation

- `git diff --check`
- `bash -n` over modified shell scripts.
- `python3 -m py_compile` over modified Python scripts.
- Local object build:
  `make -B engine/context.o engine/layer_state.o engine/layer_execute.o engine/scheduler.o engine/replay.o engine/mtp.o`
- Remote V100 appliance build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 appliance/ds4-v100-tp-ep-appliance`
- Remote startup:
  `./appliance/ds4-v100-tp-ep-appliance --help`
- Remote V100 support object build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 engine/context.o engine/layer_state.o engine/layer_execute.o engine/scheduler.o engine/replay.o engine/mtp.o`
- Remote V100 build of changed scheduler/MTP/replay tools and tests listed in
  this sprint's terminal log.

## Notes

- Uppercase compatibility constants/env vars and dash-based executable names
  were intentionally left unchanged.
- This does not promote MTP serving; it only renames the existing support API.
