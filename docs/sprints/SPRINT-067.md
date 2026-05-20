# Sprint 067: Async Pipeline Profiling And A/B Dispatch

## Status

Complete.

## Overview

Sprint 065 proved that per-stage host concurrency is a real throughput lever.
Sprint 066 made those workers persistent, but the persistent implementation
measured `7-15%` slower than Sprint 065's per-step worker shape while still
beating serial. Sprint 067 should stop guessing and make the async pipeline
measurable and A/B testable in one binary.

## Outcome

`SHIP` as the preferred opt-in async path.

Sprint 067 restores the Sprint 065 per-step worker shape as an explicit
same-binary mode, adds async timing counters, and proves that per-step async is
faster than the Sprint 066 persistent worker implementation on the standard
V100 matrix. The bare `--async-pipeline-decode` flag now selects the faster
`per-step` mode; `--async-pipeline-mode persistent` keeps the Sprint 066
implementation available for diagnostics.

## Goals

1. Add explicit async pipeline profiling counters for dispatch, wait, handoff,
   synchronization, and worker completion overhead.
2. Expose those counters in replay JSON and sustained benchmark artifacts.
3. Restore a same-binary diagnostic path for the Sprint 065 per-step worker
   shape so persistent and per-step async can be compared on the same build.
4. Run paired V100 benchmarks for serial, persistent async, and per-step async
   on the standard 1M/256K, 2/4-slot matrix.
5. Decide which async implementation should remain the preferred opt-in path,
   and whether any part is default-ready.

## Non-Goals

- MTP draft commit.
- New low-bit kernels.
- Changing queue/admission semantics.
- Externally exposing the appliance.
- Making async decode default before the A/B evidence exists.

## Implementation

1. Extend replay options with an async pipeline mode:
   - `off`: current serial/default behavior;
   - `persistent`: Sprint 066 persistent workers;
   - `per-step`: Sprint 065 diagnostic worker-per-token-step shape.
2. Keep `--async-pipeline-mode persistent` for the Sprint 066 implementation,
   and make the bare `--async-pipeline-decode` flag select the measured faster
   `per-step` mode.
3. Add counters to `ds4_v100_replay_counters` for:
   - total async dispatch wall time;
   - dispatch setup/broadcast time;
   - host wait-for-workers time;
   - per-stage wait-for-previous-slot time;
   - per-stage handoff time;
   - per-stage device synchronize time;
   - worker completion/teardown time for the per-step path.
4. Print the new counters under `timing_ms.async_pipeline` in
   `tools/ds4-v100-replay --json`.
5. Teach `tools/ds4-v100-sustained-decode-bench.sh` to:
   - pass through the new async mode;
   - preserve the mode in TSV/JSON metadata;
   - average the async timing arrays into each case result.
6. Benchmark the three modes:
   - serial baseline;
   - persistent async;
   - per-step async.

## Files Summary

- `ds4_v100_replay.h`: async mode enum/option and profiling counters.
- `ds4_v100_replay.c`: persistent profiling plus restored per-step diagnostic
  path.
- `tools/ds4-v100-replay.c`: CLI parsing, status JSON, and timing JSON.
- `tools/ds4-v100-sustained-decode-bench.sh`: mode pass-through and summary
  metadata.
- `docs/sprints/SPRINT-067-REPORT.md`: benchmark interpretation and default
  decision.
- `logs/from-cluster/sprint067-*`: cluster build, smoke, and benchmark
  artifacts.

## Definition of Done

- [x] Local object builds pass for touched C files.
- [x] `bash -n tools/ds4-v100-sustained-decode-bench.sh` passes.
- [x] `git diff --check` passes.
- [x] V100 build passes for `tools/ds4-v100-replay` and existing CUDA smokes.
- [x] Existing selected-token and wavefront smokes still pass.
- [x] Replay JSON includes `timing_ms.async_pipeline` when async mode is used.
- [x] Sustained benchmark artifacts record async mode and averaged async timing
  fields.
- [x] V100 benchmark matrix is archived for serial, persistent async, and
  per-step async.
- [x] Sprint report explains the Sprint 066 regression with timing evidence or
  clearly states what remains uncertain.
- [x] Vision is updated with the chosen async path and the next practical-use
  blocker.

## Risks

- Profiling counters can perturb the very synchronization overhead we are
  measuring. Keep them simple wall-clock counters and compare with profiling
  both enabled and disabled if results look suspicious.
- Adding a per-step diagnostic path increases code surface. Keep it isolated
  and opt-in.
- If persistent and per-step measurements vary run-to-run, repeat the focused
  1M/4-slot case before making a default decision.
