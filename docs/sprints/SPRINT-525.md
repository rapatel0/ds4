# Sprint 525 - Move MTP Forward Step Into Engine

Date: 2026-05-28

## Goal

Move the reusable MTP forward implementation out of `tools/` and into
`engine/` so MTP step logic is owned by the inference layer rather than by a
tool helper.

## Changes

- Moved `tools/mtp-forward-common.c` to `engine/mtp_step.cu`.
- Moved `tools/mtp-forward-common.h` to `engine/mtp_step.h`.
- Updated replay and MTP verify smoke includes to use `engine/mtp_step.h`.
- Updated Makefile object ownership from `tools/ds4-v100-mtp-forward-common.o`
  to `engine/mtp_step.o`.
- Renamed public constants from `DS4_V100_MTP_FORWARD_*` to `DS4_MTP_STEP_*`.

## Validation

- Local compile:
  `make -B engine/mtp_step.o tools/ds4-v100-replay.o tools/ds4-v100-mtp-verify-smoke.o`
- `git diff --check -- Makefile engine/mtp_step.cu engine/mtp_step.h smokes/mtp-forward-smoke.c smokes/mtp-verify-smoke.c tools/replay.c tools/mtp-forward-common.c tools/mtp-forward-common.h`
- `rg -n "tools/mtp-forward-common|mtp-forward-common|DS4_V100_MTP_FORWARD|tools/ds4-v100-mtp-forward-common" Makefile engine tools smokes docs/sprints/SPRINT-5*.md`
- Synced the move to the V100 pod and removed the old tool helper there.
- Remote build:
  `CUDA_ARCH=sm_70 make -B -j80 engine/mtp_step.o tools/ds4-v100-replay.o tools/ds4-v100-mtp-verify-smoke.o tools/ds4-v100-replay tools/ds4-v100-mtp-verify-smoke`

## Notes

- This is an ownership move, not a claim that TP/EP appliance MTP serving is
  complete or promoted.
- The MTP serving appliance integration remains unfinished.
