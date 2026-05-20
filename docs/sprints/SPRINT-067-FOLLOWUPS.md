# Sprint 067 Follow-Ups

## 1. Wire Preferred Async Mode Into Appliance Deployment

- **What**: Update the appliance launcher/config/runbook path so practical
  serving can opt into `--async-pipeline-decode` deliberately, with
  `async_pipeline_mode=per-step` visible in status and benchmark artifacts.
- **Why**: Sprint 067 proves per-step async is the fastest measured path, but
  it remains opt-in. Practical use needs a clear operator switch rather than a
  benchmark-only flag.
- **Severity**: Important.
- **Suggested sprint**: Sprint 068.
- **Files**: `deploy/v100/ds4-v100-appliance.env.example`,
  `tools/ds4-v100-run-appliance.sh`, `tools/ds4-v100-replay.c`,
  `docs/sprints/VISION.md`.

## 2. Replace Persistent Global Broadcasts Before Retrying Persistent Workers

- **What**: If persistent workers are revisited, replace the single
  mutex/condition-variable control plane with targeted stage-to-stage wakeups
  or stream/event handoff.
- **Why**: Sprint 067 timing shows persistent workers lose more in
  wait-for-previous-slot accumulation than they save in thread setup.
- **Severity**: Important.
- **Suggested sprint**: After the preferred async mode is wired into deployment.
- **Files**: `ds4_v100_replay.c`, `ds4_v100_scheduler.c`, `ds4_cuda.cu`.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| Wire preferred async mode into appliance deployment | Important | Sprint 068 | deployment env, launcher, replay status |
| Replace persistent global broadcasts before retrying persistent workers | Important | Later optimization | replay runtime, scheduler, CUDA handoff |
