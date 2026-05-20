# Sprint 073: Persistent Stage Pipeline Mailboxes

## Status

Complete.

## Overview

Sprint 072 proved exact MTP commit is correct but not the next throughput
lever. The best measured practical path remains the Sprint 065/067 per-step
async stage pipeline: it creates one host worker per V100 stage for each
token-step batch and reaches about `8.6` generated tok/s at 1M/4 slots. The
Sprint 066 persistent worker path avoids thread creation but regresses by
`7-15%`, mainly due global wakeup/wait behavior and per-slot synchronization.

Sprint 073 should make the persistent pipeline competitive by replacing the
single global condition-variable fanout with per-stage mailboxes and explicit
stage/slot readiness counters. This is still an opt-in scheduling mode until
V100 evidence proves it beats or matches the per-step path.

## Goals

1. Add a new async pipeline mode for persistent mailbox workers.
2. Preserve existing `off`, `per-step`, and `persistent` behavior for A/B
   comparison.
3. Replace global wake-all scheduling in the new mode with per-stage worker
   mailboxes:
   - one persistent worker per V100 stage;
   - per-dispatch generation id;
   - per-stage/per-slot ready counters;
   - stage 0 starts from token embeddings;
   - stages 1-7 wake only when the previous stage has produced the matching
     slot HC;
   - the host waits for stage 7 completion instead of every worker wake cycle.
4. Keep the CUDA execution path unchanged for the first implementation:
   blocking handoff, current stream behavior, and the existing per-stage device
   synchronize stay in place so the sprint isolates host scheduling overhead.
5. Extend benchmark/status output so `mailbox` timing is visible under the
   existing `timing_ms.async_pipeline` fields.
6. Run a paired V100 matrix against `off`, `per-step`, `persistent`, and
   `mailbox`.
7. Decide whether appliance `auto` should keep `per-step` or move to
   `mailbox`.

## Non-Goals

- Changing MTP serving or enabling multi-slot MTP.
- Rewriting routed MXFP4/F8 kernels.
- Replacing blocking peer copies with CUDA event/stream handoff in this sprint.
- Changing the default operator profile before measured evidence.
- Chasing the `1k+ tok/s` aspiration directly.

## Implementation

1. Extend `ds4_v100_replay_async_pipeline_mode` with a `MAILBOX` value.
2. Update CLI/config parsing:
   - `tools/ds4-v100-replay --async-pipeline-mode mailbox`;
   - `tools/ds4-v100-sustained-decode-bench.sh --async-pipeline-mode mailbox`;
   - `tools/ds4-v100-run-appliance.sh` accepts `mailbox` but does not select it
     from `auto` until benchmark evidence is positive.
3. Add a new mailbox runtime in `ds4_v100_replay.c` rather than mutating the
   old persistent runtime in place. Keep the Sprint 066 implementation
   available as `persistent` for control measurements.
4. Structure the mailbox runtime around per-stage state:
   - `pthread_mutex_t` and `pthread_cond_t` per stage;
   - `start_generation` for stage 0;
   - `ready_generation[slot]` from the previous stage;
   - `done_generation[slot]` for the current stage;
   - local error propagation to the dispatcher;
   - aggregate timing into the existing replay counters.
5. Reuse the current worker body operations:
   - stage 0: `ds4_v100_stage_scheduler_decode_token_slot_span`;
   - stages 1-7: `ds4_v100_stage_scheduler_handoff_slot_span`, then
     `ds4_v100_stage_scheduler_decode_hc_slot_span`.
6. Add one targeted compatibility smoke if needed, or reuse
   `tests/cuda_v100_stage_wavefront_smoke` plus selected-token/full-scheduler
   smokes for correctness.
7. Add a Sprint 073 report and update `docs/sprints/VISION.md` with the
   measured decision.

## Parallel Work

- Scheduling explorer: review `ds4_v100_replay.c` mailbox state and failure
  propagation for races, missed wakeups, and stale generation reuse.
- Benchmark explorer: run a 1M/2-slot quick matrix while implementation
  continues locally, then run the full 1M/4-slot and 256K/4-slot matrix after
  correctness is green.

## Definition of Done

- [x] Local compile passes for changed C files.
- [x] `bash -n tools/ds4-v100-sustained-decode-bench.sh` passes.
- [x] `git diff --check` passes.
- [x] CLI rejects invalid async modes and accepts `mailbox`.
- [x] `/v100/status` reports `async_pipeline_mode="mailbox"` when selected.
- [x] Existing serial, per-step, and persistent modes still build and run.
- [x] V100 two-stage/wavefront correctness smoke passes.
- [x] V100 selected-token or short sustained smoke returns token hex `3136` in
  mailbox mode.
- [x] V100 A/B matrix records `off`, `per-step`, `persistent`, and `mailbox`
  at 1M/2 slots and 1M/4 slots.
- [x] Sprint report records whether `mailbox` beats, matches, or regresses
  versus `per-step`.
- [x] Vision document is updated with the scheduling decision.

## Outcome

`SHIP`, but keep `mailbox` diagnostic and leave appliance `auto` on
`per-step`.

The mailbox runtime adds persistent per-stage workers with per-stage condition
variables and readiness signaling. It is correct and reduces the old persistent
pipeline's measured wait-prev counter, but the same-build V100 matrix shows it
does not beat the existing per-step implementation:

| Mode | 1M/2 generated tok/s | 1M/4 generated tok/s | Decision |
|---|---:|---:|---|
| off | `3.862534` | `3.801132` | serial control |
| per-step | `5.562124` | `8.649395` | keep as appliance `auto` default |
| persistent | `5.118536` | `7.865004` | old persistent control |
| mailbox | `5.123876` | `8.053284` | opt-in diagnostic |

At 1M/4 slots, mailbox is `2.394%` faster than old persistent but `6.892%`
slower than per-step. The next throughput sprint should not spend more time on
host condition-variable scheduling alone; it should target CUDA event/stream
handoff or kernel-side work.

Artifacts:

- `logs/from-cluster/sprint073-mailbox-smoke`
- `logs/from-cluster/sprint073-ab-off`
- `logs/from-cluster/sprint073-ab-per-step`
- `logs/from-cluster/sprint073-ab-persistent`
- `logs/from-cluster/sprint073-ab-mailbox`
- `logs/from-cluster/sprint073-ab-comparison`

## Decision Rule

- If `mailbox` is at least `5%` faster than `per-step` on 1M/4 slots and does
  not regress 1M/2 slots, update appliance `auto` to select `mailbox`.
- If `mailbox` is within `2%` of `per-step`, keep `per-step` as the default
  for now but retain `mailbox` as an opt-in mode for follow-up stream/event
  handoff work.
- If `mailbox` regresses by more than `2%`, keep it diagnostic only and pivot
  Sprint 074 to a kernel-side or CUDA event handoff sprint using the new timing
  evidence.

## Risks

- Persistent workers can deadlock if generation counters or slot-ready signals
  are mishandled.
- The existing per-stage device synchronizes may dominate enough that mailbox
  scheduling only matches per-step rather than improving it.
- Handoff timing in the 4-slot async path is noisy; use paired same-build
  comparisons rather than comparing against older sprint artifacts alone.
- A faster mailbox scheduler still leaves GPU utilization far below the
  aspirational throughput envelope, so this is one scheduling step, not the
  final performance solution.

## Security

No new external serving surface. The new mode is an opt-in internal scheduling
path exposed through existing loopback appliance configuration.
