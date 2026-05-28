# Sprint 518 - Trim TP/EP Deployment Env Surface

Date: 2026-05-28

## Goal

Keep the deployment examples aligned with the thin TP/EP appliance launcher by
removing experiment-era TP/EP environment variables that are no longer
operator-facing inputs.

## Changes

- Removed stale TP/EP experiment variables from
  `deploy/v100/ds4-v100-appliance.env.example`.
- Removed the same stale TP/EP variables from the K8s embedded appliance env.
- Kept only the real TP/EP appliance deployment inputs: binary, contract,
  TurboMind index, position, and VRAM/NCCL free-memory guards.

## Validation

- `git diff --check -- deploy/v100/ds4-v100-appliance.env.example deploy/v100/ds4-v100-appliance.k8s.yaml`
- Local YAML parse of `deploy/v100/ds4-v100-appliance.k8s.yaml`
- Local env-example config check through `tools/ds4-v100-run-appliance.sh --check --allow-missing`
- Local K8s embedded-env config check through `tools/ds4-v100-run-appliance.sh --check --allow-missing`
- Synced both deploy files to the V100 pod.
- Remote env-example config check through `tools/ds4-v100-run-appliance.sh --check --allow-missing`
- Remote K8s embedded-env config check through `tools/ds4-v100-run-appliance.sh --check --allow-missing`
- Remote grep confirmed only the retained TP/EP appliance env inputs remain.

## Notes

- This changes deployment documentation/configuration only; it does not change
  appliance execution.
- MTP serving is still not implemented for TP/EP mode.
