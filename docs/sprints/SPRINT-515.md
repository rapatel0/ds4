# Sprint 515 - Remove Legacy Prefix From Tool Source Names

Date: 2026-05-28

## Goal

Continue the structural extraction namespace cleanup under `tools/`: source
file names should rely on the directory namespace instead of carrying the
legacy `ds4-v100-` prefix. Existing executable target names stay stable for
operator scripts and sprint lineage.

## Changes

- Renamed the remaining tracked `tools/ds4-v100-*.c`, `*.cu`, and `*.h`
  source files to prefix-free source names.
- Updated Makefile dependencies and compile commands while preserving the
  existing `tools/ds4-v100-*` binary targets.
- Updated the MTP verify smoke include to use `tools/mtp-forward-common.h`.

## Validation

- `git diff --check`
- Local build:
  `make -B tools/ds4-v100-plan tools/ds4-v100-plan-tp tools/ds4-v100-tp-ep-pack-contract tools/ds4-v100-tp-ep-int8-candidates tools/ds4-v100-tp-estimate tools/ds4-v100-pack tools/ds4-v100-turbomind-admit tools/ds4-v100-mtp-forward-common.o tools/ds4-v100-replay.o`
- Remote V100 build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-plan tools/ds4-v100-plan-tp tools/ds4-v100-tp-ep-pack-contract tools/ds4-v100-tp-ep-int8-candidates tools/ds4-v100-tp-estimate tools/ds4-v100-pack tools/ds4-v100-turbomind-admit tools/ds4-v100-turbomind-pack tools/ds4-v100-appliance-pack tools/ds4-v100-tp4-layer-proxy tools/ds4-v100-tp8-layer-proxy tools/ds4-v100-mtp-verify-smoke tools/ds4-v100-replay appliance/ds4-v100-tp-ep-appliance`
- Remote startup:
  `./appliance/ds4-v100-tp-ep-appliance --help`
- Local and remote source audits found no remaining
  `tools/ds4-v100-*.c`, `tools/ds4-v100-*.cu`, or
  `tools/ds4-v100-*.h` files.

## Notes

- This is a source layout cleanup only; it does not rename user-facing
  binaries or shell/Python operator scripts.
- MTP remains sidecar/support only and is not promoted to serving here.
