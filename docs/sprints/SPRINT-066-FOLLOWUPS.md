# Sprint 066 Follow-Ups

## 1. Profile Persistent Async Dispatch And Handoff

- **What**: Add timing visibility around the persistent async pipeline's
  dispatch generation, condition-variable waits, per-slot done broadcasts,
  handoff calls, and per-device synchronizes. Compare those timings directly
  against the Sprint 065 per-step worker implementation or a controlled
  equivalent.
- **Why**: Sprint 066 preserved correctness and remained faster than serial,
  but persistent async measured `7-15%` slower than Sprint 065. That invalidates
  the assumption that removing thread creation would automatically improve
  throughput.
- **Severity**: Important.
- **Suggested sprint**: Sprint 067.
- **Files**: `ds4_v100_replay.c`,
  `tools/ds4-v100-sustained-decode-bench.sh`,
  `docs/sprints/SPRINT-066-REPORT.md`.

## 2. Decide Async Default Only After Dispatch Regression Is Resolved

- **What**: Keep `--async-pipeline-decode` opt-in until either persistent
  dispatch beats Sprint 065 or the faster per-step worker shape is restored as
  the preferred implementation.
- **Why**: Four-slot persistent async is more than 2x faster than serial, but
  below the prior opt-in async result. Defaulting it now would preserve a real
  speedup but obscure the regression against the best known path.
- **Severity**: Important.
- **Suggested sprint**: Sprint 067 or after profiling evidence.
- **Files**: `ds4_v100_replay.c`, `tools/ds4-v100-replay.c`,
  `docs/sprints/VISION.md`.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| Profile persistent async dispatch and handoff | Important | Sprint 067 | `ds4_v100_replay.c`, benchmark harness |
| Decide async default only after regression is resolved | Important | Sprint 067 or after profiling | replay runtime, replay CLI, vision |
