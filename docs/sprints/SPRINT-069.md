# Sprint 069: Appliance Launcher Soak Harness

## Status

Complete.

## Overview

Sprint 068 wired the preferred async profile into the appliance launcher and
proved it with a one-off loopback smoke. Sprint 069 turns that manual smoke
into a reusable operator harness and runs the practical 4-slot profile through
the launcher, so future deployment changes can be validated without rewriting
ad hoc shell/Python.

## Outcome

`SHIP`.

The new soak harness starts the appliance through
`tools/ds4-v100-run-appliance.sh`, validates health/status/metrics, runs
concurrent generation requests, checks first-token hex `3136`, and writes a
summary artifact. The V100 4-slot, 1M-context practical run completed with
`4/4` token matches, `async_pipeline_mode=per-step`, `7.518610` aggregate
generated tok/s, and `7.048697` continuation tok/s.

## Goals

1. Add a reusable `tools/ds4-v100-appliance-soak.sh` harness that starts
   `tools/ds4-v100-run-appliance.sh`.
2. Validate `/health`, `/v100/status`, `/metrics`, and generation responses.
3. Support practical config overrides for context, slots, tokens, requests,
   async mode, queue policy, and log directory.
4. Verify expected first-token hex `3136`.
5. Archive a V100 4-slot practical-profile run through the launcher.

## Non-Goals

- New model math.
- MTP commit.
- External exposure or auth.
- Replacing the sustained decode benchmark.

## Definition of Done

- [x] `bash -n tools/ds4-v100-appliance-soak.sh` passes.
- [x] Local `--help` or argument parsing is valid.
- [x] V100 harness run starts the appliance through
  `tools/ds4-v100-run-appliance.sh`.
- [x] V100 status reports `async_pipeline_mode=per-step` for the practical
  multi-slot profile.
- [x] V100 generation responses all return token hex `3136`.
- [x] Harness artifacts include request, status, metrics, responses, startup env,
  command, server log, and summary JSON.
- [x] Sprint report and vision are updated.
