# Sprint 500 - Appliance Boundary Extraction

Date: 2026-05-28

## Goal

Continue the structural extraction plan by moving appliance-owned code out of
`tools/ds4-v100-tp-ep-full-layer-smoke.cu` without changing the TP/EP serving
path or treating MTP as complete.

## Changes

- Extracted CLI option parsing into `appliance/options.cu`.
- Extracted TP/EP HTTP serving glue into `appliance/http_server.cu`.
- Extracted the current entrypoint into `appliance/entrypoint.cu`.
- Moved the TP/EP translation-unit shell to `appliance/main.cu`.
- Added `appliance/ds4-v100-tp-ep-appliance` as the production TP/EP binary.
- Left `tools/ds4-v100-tp-ep-full-layer-smoke.cu` as a temporary one-line
  compatibility wrapper while any direct smoke references are retired.
- Flipped the launcher default `DS4_V100_TP_EP_BIN` to the new appliance
  binary.
- Added the new appliance files to the CUDA target dependencies.

## MTP Scope

MTP remains separate unfinished work. This sprint does not implement or promote
MTP inside the TP/EP full-layer serving path.

## Validation

- `git diff --check`
- `bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-run-tp-ep-appliance.sh`
- Remote V100 build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 appliance/ds4-v100-tp-ep-appliance`
- Remote startup:
  `./appliance/ds4-v100-tp-ep-appliance --help`
