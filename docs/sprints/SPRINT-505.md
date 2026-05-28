# Sprint 505 - Remove TP/EP Appliance Path-Selection Flags

Date: 2026-05-28

## Goal

Finish the appliance CLI cleanup discipline from the structural extraction
plan: the production TP/EP appliance should expose only real runtime knobs,
not the old promoted-path and experiment flag matrix.

## Changes

- Replaced the legacy `parse_args` matrix with 15 real long runtime flags plus
  `--help`.
- Removed all `*-gate` flags from the appliance parser and help surface.
- Removed fixed model invariants and diagnostics from the production launcher
  command (`top-k`, `kv-slot`, direct warmup/iters, CUDA graph toggles, CUDA
  profiler toggles, and VRAM report-only mode).
- Updated direct TP/EP profiling and sustained bench command builders so they
  no longer pass removed appliance flags.

## Validation

- `git diff --check`
- `bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-tp-ep-sustained-bench.sh`
- `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py`
- Launcher print-command audit with `--allow-missing`: no `--*-gate`,
  `--top-k`, `--kv-slot`, `--warmup`, `--iters`, `--cuda-profiler*`, or
  `--vram-report` flags in the TP/EP appliance command.
- Remote V100 build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 appliance/ds4-v100-tp-ep-appliance`
- Remote startup:
  `./appliance/ds4-v100-tp-ep-appliance --help`
- Remote parser audit:
  `accepted_real_flags_excluding_help=15`, `accepted_gate_flags=0`.

## Size Audit

- Appliance total: 2,073 lines across `appliance/*.cu`.
- Largest engine file: `engine/decode_loop.cu` at 1,998 lines.
- `kernels/v100/*.cuh` remain below the 1,500-line kernel-file cap.
