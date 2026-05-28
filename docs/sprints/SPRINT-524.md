# Sprint 524 - Remove Legacy TP/EP Routed-FFN Input Aliases

Date: 2026-05-28

## Goal

Finish removing active TP/EP-prefixed env aliases from the older TP2
routed-FFN scheduler path.

## Changes

- Removed `DS4_V100_TP_EP_LAYER_FIRST`.
- Removed `DS4_V100_TP_EP_LAYER_COUNT`.
- Removed `DS4_V100_TP_EP_SHARD_DIR`.
- Removed `DS4_V100_TP_EP_PEER`.
- Kept the existing generic replacements:
  - `DS4_V100_TP_ROUTED_FFN_LAYER`
  - `DS4_V100_TP_ROUTED_FFN_LAYER_COUNT`
  - `DS4_V100_TP_ROUTED_FFN_SHARD_DIR`
  - `DS4_V100_TP_ROUTED_FFN_PEER_GPU`
  - `DS4_V100_TP2_LAYER`
  - `DS4_V100_TP2_LAYER_COUNT`
  - `DS4_V100_TP2_SHARD_DIR`
  - `DS4_V100_TP2_PEER_GPU`

## Validation

- `rg -n "DS4_V100_TP_EP_(LAYER_FIRST|LAYER_COUNT|SHARD_DIR|PEER)" engine tools deploy appliance smokes Makefile`
- `git diff --check -- engine/scheduler_core.inc`
- Synced `engine/scheduler_core.inc` to the V100 pod.
- Remote forced build: `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`

## Notes

- This does not remove the diagnostic TP2 routed-FFN scheduler path.
