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

## Follow-on: Makefile Cleanup

Appended to the previous structural cleanup instead of opening a new sprint.

### Changes

- Split the single-line `TP_EP_APPLIANCE_DEPS` into readable groups:
  `V100_KERNEL_HEADERS`, `ENGINE_API_HEADERS`, `ENGINE_SUPPORT_SRCS`,
  `ENGINE_STEP_SRCS`, and `APPLIANCE_SRCS`.
- Replaced the long `clean` command body with `CLEAN_OBJS` and `CLEAN_BINS`
  groups.
- Added missing `tools/ds4-v100-tp-ep-int8-candidates` coverage to `clean`.
- Kept smoke binary output paths under `tools/` for now because active shell
  harnesses still invoke those paths.
- Kept `tools/ds4-source-oracle-vector` because `tools/ds4-v100-gate.sh`
  still builds and runs it.
- Did not add `appliance/options.cu` because that file does not exist in the
  current tree.

### Validation

- `git diff --check -- Makefile`
- `make -n clean`
- Local representative build:
  `make -B tools/ds4-v100-plan tools/ds4-v100-plan-tp tools/ds4-v100-tp-ep-pack-contract tools/ds4-v100-tp-ep-int8-candidates tools/ds4-v100-tp-estimate tools/ds4-v100-context-smoke tools/ds4-v100-layer-descriptor-gate tools/ds4-source-oracle-vector tools/ds4-v100-mtp-sidecar-gate`
- Remote V100 dry-run:
  `CUDA_ARCH=sm_70 make -n appliance/ds4-v100-tp-ep-appliance`
- Remote V100 build:
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
