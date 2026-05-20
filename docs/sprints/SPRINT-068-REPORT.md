# Sprint 068 Report: Appliance Async Serving Profile

## Outcome

`SHIP`.

Sprint 068 wires the measured preferred async path into the appliance launcher
and deployment config. Practical multi-slot appliance configs now resolve
`DS4_V100_ASYNC_PIPELINE_MODE=auto` to `per-step`, while one-slot configs stay
serial unless the operator explicitly selects an async mode.

## Implementation

- Added `DS4_V100_ASYNC_PIPELINE_MODE` to the appliance environment contract.
- Supported modes in `tools/ds4-v100-run-appliance.sh`:
  - `off`;
  - `auto`;
  - `per-step`;
  - `per_step`;
  - `persistent`.
- Resolved `auto` to:
  - `per-step` when `DS4_V100_ACTIVE_MICROBATCH > 1`;
  - `off` for one-slot latency configs.
- Added `--async-pipeline-mode <resolved>` to the replay command when the
  resolved mode is not `off`.
- Updated startup artifacts:
  - `startup.env` records both configured and resolved async mode;
  - `command.txt` includes the resolved replay command.
- Updated deployment defaults:
  - `DS4_V100_SLOTS=4`;
  - `DS4_V100_ACTIVE_MICROBATCH=4`;
  - `DS4_V100_QUEUE_POLICY=sequential`;
  - `DS4_V100_TOKENS=16`;
  - `DS4_V100_ASYNC_PIPELINE_MODE=auto`.

## Validation

Local:

- `bash -n tools/ds4-v100-run-appliance.sh`
- `tools/ds4-v100-run-appliance.sh --env deploy/v100/ds4-v100-appliance.env.example --allow-missing --check`
- `tools/ds4-v100-run-appliance.sh --env deploy/v100/ds4-v100-appliance.env.example --allow-missing --print-command`
- `git diff --check`

V100:

- `bash -n tools/ds4-v100-run-appliance.sh`
- `tools/ds4-v100-run-appliance.sh --env deploy/v100/ds4-v100-appliance.env.example --check`
- `tools/ds4-v100-run-appliance.sh --env deploy/v100/ds4-v100-appliance.env.example --print-command`
- Loopback launcher smoke with:
  - `DS4_V100_CTX=262144`;
  - `DS4_V100_SLOTS=2`;
  - `DS4_V100_ACTIVE_MICROBATCH=2`;
  - `DS4_V100_ASYNC_PIPELINE_MODE=auto`.

Evidence:

- `logs/from-cluster/sprint068-launcher-check/check.log`
- `logs/from-cluster/sprint068-launcher-smoke/server.log`
- `logs/from-cluster/sprint068-launcher-smoke/status_before.json`
- `logs/from-cluster/sprint068-launcher-smoke/status_after.json`
- `logs/from-cluster/sprint068-launcher-smoke/responses.json`
- `logs/from-cluster/sprint068-launcher-smoke/runtime/startup.env`
- `logs/from-cluster/sprint068-launcher-smoke/runtime/command.txt`

## Result

The V100 launcher smoke proves the deployment path, not just the benchmark
path:

- `/v100/status` reports `async_pipeline_decode=true`.
- `/v100/status` reports `async_pipeline_mode="per-step"`.
- Two loopback generation requests returned HTTP 200.
- Both generation responses selected first-token hex `3136`.
- Both generation responses included `timing_ms.async_pipeline`.

This means the preferred measured async path is now operator-facing through the
appliance launcher and deployment config.

## Decision

- Keep practical deployment defaults on 4 slots, sequential queueing, and
  `DS4_V100_ASYNC_PIPELINE_MODE=auto`.
- Keep one-slot latency configs serial through the `auto` resolver.
- Leave MTP verify off by default; true MTP commit remains separate.
- The next blocker is no longer selecting the async path. It is a longer
  launched-appliance soak and the next throughput lever: either true MTP draft
  commit or stream/event handoff for inter-stage transfer.
