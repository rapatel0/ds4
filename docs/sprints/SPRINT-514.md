# Sprint 514 - Mark MTP Engine Support As Sidecar Only

Date: 2026-05-28

## Goal

Continue the structural extraction cleanup without overstating MTP readiness:
the current MTP engine code is sidecar residency/support, not the future
serving `mtp_step.cu` path.

## Changes

- Renamed `engine/mtp.{c,h}` to `engine/mtp_sidecar.{c,h}`.
- Renamed the Makefile object group to `V100_MTP_SIDECAR_OBJS`.
- Updated MTP smokes, replay, and MTP forward common headers to include the
  sidecar header explicitly.

## Validation

- `git diff --check`
- Local build:
  `make -B engine/mtp_sidecar.o tools/ds4-v100-mtp-sidecar-gate tools/ds4-v100-mtp-residency-smoke`
- Remote V100 build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 engine/mtp_sidecar.o tools/ds4-v100-mtp-residency-smoke tools/ds4-v100-mtp-verify-smoke tools/ds4-v100-replay appliance/ds4-v100-tp-ep-appliance`
- Remote startup:
  `./appliance/ds4-v100-tp-ep-appliance --help`

## Notes

- This is a naming and ownership cleanup only.
- MTP serving remains unpromoted; the structural target files
  `engine/mtp_step.cu` and `kernels/v100/mtp.cuh` should only appear when the
  actual serving MTP head is implemented.
