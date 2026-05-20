# Sprint 078: Opt-In Event-Ordered Stage Handoff

## Status

Complete. Event-ordered stage handoff is correct and opt-in, but it remains
disabled by default because paired V100 throughput evidence showed only a
`0.12%` generated tok/s improvement, below the `3%` default threshold.

## Overview

Sprint 077 proved output-head batching is not the next useful throughput lever.
The current best practical path remains Sprint 076 per-slot device top-1 plus
per-step async stage scheduling at about `9` generated tok/s for the 1M/4-slot
fixture. The timing counters still show large host wait and stage synchronization
buckets, and the per-step path currently synchronizes each device after each
stage/slot decode before marking that slot ready for the next stage.

Sprint 078 adds a narrow, opt-in CUDA event handoff path. The goal is to replace
the per-stage/slot `cudaDeviceSynchronize()` readiness gate with CUDA event
record/wait ordering: record an event after a stage queues decode work, make the
next stage wait on that event before its peer copy, and keep the final all-stage
sync for correctness and error surfacing.

## Goals

1. Add minimal opaque CUDA event APIs to the GPU abstraction.
2. Add an event-aware async HC handoff path between stage schedulers.
3. Add an opt-in replay path for per-step async mode that records stage-ready
   events and waits on them before peer copies.
4. Preserve existing per-step async, persistent, mailbox, wavefront, and serial
   paths unchanged by default.
5. Expose the experiment through CLI/deployment/benchmark controls.
6. Validate selected-token correctness and sustained 1M/4-slot throughput on
   V100 against the current default.

## Non-Goals

- Full explicit CUDA stream plumbing through every kernel helper.
- Per-stream cuBLAS handle management.
- Changing model math, MTP, output-head selection, or routed MoE kernels.
- Removing the final batch-level synchronization in the first implementation.
- Making event handoff default without same-fixture V100 evidence.

## Implementation

1. Extend `ds4_gpu.h`, `ds4_cuda.cu`, and fallback backends:
   - opaque `ds4_gpu_event`;
   - create/free/record helpers;
   - stream wait helper;
   - async tensor copy after an event.
2. Extend `ds4_v100_scheduler.{h,c}`:
   - add `ds4_v100_stage_scheduler_handoff_slot_span_after_event_async`;
   - leave existing blocking and async handoff APIs untouched.
3. Extend `ds4_v100_replay.{h,c}`:
   - add `async_event_handoff` option;
   - allocate reusable stage-ready events for configured slots;
   - in per-step async mode, record events after decode and mark slots ready
     without synchronizing when event handoff is enabled;
   - wait on the previous stage event before the peer copy;
   - keep final `replay_sync_all_stages`.
4. Extend tools:
   - `tools/ds4-v100-replay.c`: `--async-event-handoff`;
   - `tools/ds4-v100-sustained-decode-bench.sh`: pass-through flag;
   - `tools/ds4-v100-run-appliance.sh` and deploy examples:
     `DS4_V100_ASYNC_EVENT_HANDOFF=0`.

## Definition of Done

- [x] Local compile passes for changed C/CUDA-facing objects.
- [x] `bash -n` passes for changed shell tools.
- [x] `git diff --check` passes.
- [x] V100 CUDA build passes for replay and relevant smokes.
- [x] V100 selected-token smoke passes for the default path.
- [x] V100 selected-token or short replay smoke passes with event handoff
  enabled.
- [x] Sustained V100 A/B at `ctx=1048576`, `slots=4`, `tokens=16`,
  `requests=4`, `async_pipeline_mode=per-step` records generated tok/s,
  continuation tok/s, async timing, token matches, and GPU utilization for:
  - default per-step;
  - event handoff opt-in.
- [x] Sprint report records whether event handoff remains opt-in or becomes
  default.
- [x] Vision document is updated.
- [x] Artifacts are committed.

## Results

| Path | Generated tok/s | Continuation tok/s | Avg latency ms | Handoff sum ms | Device sync sum ms | Avg GPU util | Token match |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default per-step | `9.147418` | `8.575704` | `6994.719` | `248.432` | `6.946` | `19.958%` | `4/4` |
| Event handoff opt-in | `9.158602` | `8.586189` | `6986.078` | `193.909` | `0.000` | `19.766%` | `4/4` |

Event handoff removed the per-stage device-sync timing bucket and reduced the
handoff sum by `21.95%`, but generated/continuation tok/s improved only
`0.12%`. The path is useful as an ordering/timing primitive, but not a default
throughput win.

## Decision Rule

- Make event handoff default only if it improves generated tok/s by at least
  `3%` without selected-token or sustained-token correctness regression.
- Keep opt-in if it is correct but below the default threshold.
- Disable/remove the path if any race, token mismatch, or instability appears.

## Risks

- Destination stages may consume HC before the peer copy completes if event
  ordering is wrong.
- Default-stream semantics may serialize enough work that the event path is
  correct but neutral.
- CUDA errors move from per-stage sync to final batch sync, making diagnosis
  less local.
- Event allocation/reuse must match slot lifetimes exactly.

## Security

No new external serving surface. The feature is an internal opt-in performance
path.
