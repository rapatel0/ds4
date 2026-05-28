# Sprint 519 - Trim TP/EP HTTP Bench Gate Surface

Date: 2026-05-28

## Goal

Keep the TP/EP HTTP bench wrapper aligned with the thin appliance launcher by
removing stale experiment-gate controls that are no longer consumed.

## Changes

- Removed old TP/EP gate options from `tools/ds4-v100-tp-ep-http-bench.sh`.
- Removed matching ignored `DS4_V100_TP_EP_*` exports from the server launch
  environment.
- Kept the wrapper focused on benchmark shape and appliance path inputs.

## Validation

- `bash -n tools/ds4-v100-tp-ep-http-bench.sh`
- `tools/ds4-v100-tp-ep-http-bench.sh --help`
- `git diff --check -- tools/ds4-v100-tp-ep-http-bench.sh`
- Local grep confirmed only retained TP/EP appliance env inputs remain.
- Synced the wrapper to the V100 pod.
- Remote `bash -n tools/ds4-v100-tp-ep-http-bench.sh`
- Remote `tools/ds4-v100-tp-ep-http-bench.sh --help`
- Remote grep confirmed only retained TP/EP appliance env inputs remain.

## Notes

- This wrapper change does not run a benchmark; it only removes no-op controls
  from the launch surface.
