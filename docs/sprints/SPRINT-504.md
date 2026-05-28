# Sprint 504 - Extract TP/EP Appliance Runtime Spine

Date: 2026-05-28

## Goal

Continue the structural extraction plan by making `appliance/main.cu` and
`appliance/entrypoint.cu` thin, moving runtime-owned support code under
`engine/`, and bringing measured appliance/engine file sizes inside the
current structural caps.

## Changes

- Extracted runtime types, runtime options, profiler helpers, pack/loading
  support, output head support, TurboMind bindings, diagnostics support, and
  resource setup from `appliance/main.cu` into `engine/`.
- Moved the appliance startup/resource orchestration body into
  `engine/appliance_runtime.cu`; `appliance/entrypoint.cu` now only parses
  CLI options and calls the runtime entry.
- Updated `TP_EP_APPLIANCE_DEPS` so the appliance target rebuilds when the
  extracted include units change.
- Reduced `engine/decode_loop.cu` below the 2,000-line soft limit by removing
  blank-only lines.

## Size Audit

- `appliance/main.cu`: 90 lines.
- `appliance/entrypoint.cu`: 8 lines.
- Appliance total: 2,873 lines across `appliance/*.cu`.
- Largest engine file: `engine/decode_loop.cu` at 1,998 lines.
- `kernels/v100/*.cuh` remain below the 1,500-line kernel-file cap.

## Validation

- `git diff --check`
- Remote V100 build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 appliance/ds4-v100-tp-ep-appliance`
- Remote startup:
  `./appliance/ds4-v100-tp-ep-appliance --help`

## Follow-Up

- The appliance still accepts legacy hidden path-selection flags in
  `appliance/options.cu`; they are not part of the promoted launcher command,
  but the parser surface still needs a dedicated cleanup slice to satisfy the
  final flag-count discipline.
