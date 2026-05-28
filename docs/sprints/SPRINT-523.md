# Sprint 523 - Drop Legacy TP/EP Engine Env Aliases

Date: 2026-05-28

## Goal

Remove stale TP/EP-prefixed env aliases from the older TP2 routed execution
path while keeping the generic TP2/TURBOMIND controls intact.

## Changes

- Removed `DS4_V100_TP_EP_ASYNC_INPUT` and `DS4_V100_TP_EP_VERBOSE` aliases.
- Removed `DS4_V100_TP_EP_PARALLEL_HALVES` alias.
- Removed `DS4_V100_TP_EP_ROUTED_FFN` alias.
- Kept `DS4_V100_TP2_*` and `DS4_V100_TP_ROUTED_FFN*` controls.

## Validation

- `rg -n "DS4_V100_TP_EP_(ASYNC_INPUT|VERBOSE|PARALLEL_HALVES|ROUTED_FFN)" engine tools deploy appliance smokes Makefile`
- `git diff --check -- engine/layer_execute_core.inc engine/scheduler_core.inc`
- Synced the engine include changes to the V100 pod.
- Remote forced build: `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`

## Notes

- This is a namespace cleanup only; it does not delete the TP2 routed support
  path.
