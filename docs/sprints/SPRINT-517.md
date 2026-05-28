# Sprint 517 - Trim TP/EP Launcher Experiment Surface

Date: 2026-05-28

## Goal

Continue the structural extraction by making the production appliance launcher
reflect the thin TP/EP appliance contract instead of the old experiment harness
environment surface.

## Changes

- Removed stale `DS4_V100_TP_EP_*` experiment defaults from
  `tools/ds4-v100-run-appliance.sh`.
- Removed matching validation, startup logging, and exports for TP/EP variables
  no longer consumed by the TP/EP appliance command.
- Kept the real TP/EP launcher inputs: binary, contract, TurboMind index,
  tokenizer model, position, VRAM/NCCL free-memory guards, and extra appliance
  arguments.

## Validation

- `bash -n tools/ds4-v100-run-appliance.sh`
- `git diff --check -- tools/ds4-v100-run-appliance.sh`
- Local negative TP/EP config check confirmed the default context is rejected:
  `DS4_V100_SERVE_MODE=tp-ep ... tools/ds4-v100-run-appliance.sh --check --allow-missing`
- Local positive TP/EP config check:
  `DS4_V100_SERVE_MODE=tp-ep DS4_V100_CTX=262144 DS4_V100_SLOTS=1 DS4_V100_ACTIVE_MICROBATCH=1 ... tools/ds4-v100-run-appliance.sh --check --allow-missing`
- Synced the launcher to the V100 pod and repeated:
  `bash -n tools/ds4-v100-run-appliance.sh`
- Remote negative TP/EP config check confirmed the default context is rejected.
- Remote positive TP/EP config check passed with `DS4_V100_CTX=262144`.

## Notes

- This is launcher cleanup only; it does not change the TP/EP appliance binary.
- MTP serving is still not implemented for TP/EP mode.
