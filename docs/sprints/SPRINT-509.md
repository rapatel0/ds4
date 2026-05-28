# Sprint 509 - Remove Unreachable Appliance Gate Paths

Date: 2026-05-28

## Goal

Continue the structural cleanup by deleting code that remained behind
unreachable appliance gates after the public runtime flags were removed.

## Changes

- Removed the standalone output-head gate and resident output-head gate paths.
- Kept the shared output-head serving implementation.
- Removed the unreachable CUDA profiler window plumbing and fields.
- Removed the stale `vram_report` option; VRAM reporting now depends on the
  retained minimum-free-MiB runtime knobs.

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

- This removes only paths that were unreachable through the appliance parser.
- MTP remains sidecar/support work and is not promoted by this sprint.
