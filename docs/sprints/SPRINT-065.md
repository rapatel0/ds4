# Sprint 065: Async Stage Pipeline Decode

## Status

Complete.

## Overview

Sprint 064 proved that simply reordering stage/slot work from one host thread is
not enough. The V100 appliance needs real concurrent stage execution: while
gpu1 is decoding slot 0, gpu0 should be able to start slot 1, gpu2 should be
able to consume slot 0 as soon as gpu1 completes, and so on.

Sprint 065 implements an opt-in per-stage worker pipeline for same-length
non-MTP batch decode. It keeps the current serial scheduler as the default,
uses the existing slot-span scheduler APIs, and measures whether actual
cross-GPU overlap improves sustained serving throughput.

## Goals

1. Add an opt-in async pipeline decode path for active non-MTP batches.
2. Use one worker thread per V100 stage inside the replay runtime.
3. Preserve per-slot ordering, prompt/continuation semantics, and token hex
   `3136` correctness.
4. Keep the default serial batch path unchanged.
5. Benchmark async pipeline versus paired serial control at 1M/2, 1M/4,
   256K/2, and 256K/4.

## Non-Goals

- Enabling the async pipeline by default.
- MTP commit or MTP batching.
- Rewriting low-bit kernels.
- Changing queue/admission policy.
- Changing the model format, pack format, or layer ownership topology.

## Architecture

The async path lives in `ds4_v100_replay.c` and calls the existing scheduler
entrypoints:

- stage 0: `ds4_v100_stage_scheduler_decode_token_slot_span`
- stages 1-7: `ds4_v100_stage_scheduler_handoff_slot_span` followed by
  `ds4_v100_stage_scheduler_decode_hc_slot_span`

The runtime builds a bounded in-process pipeline for a single token-step batch:

1. The batch owner initializes stage/slot state for `n_slots`.
2. Stage worker 0 decodes slot 0, publishes completion to stage 1, then starts
   slot 1 without waiting for downstream stages.
3. Stage workers 1-7 wait for the previous stage's slot completion, perform
   handoff plus decode, then publish to the next stage.
4. The owner waits until stage 7 has completed all slots, then synchronizes all
   devices once before token selection.

This is different from Sprint 064: stage work is submitted from different host
threads, so CUDA work on different devices can overlap. The pipeline only
applies when `n_slots > 1`; otherwise it falls back to the serial path.

## Implementation

1. Add replay option and CLI flag:
   - `ds4_v100_replay_options.async_pipeline_decode`
   - `tools/ds4-v100-replay --async-pipeline-decode`
   - sustained bench pass-through and TSV status field.
2. Add `replay_feed_token_batch_async_pipeline` in `ds4_v100_replay.c`:
   - local worker structs, mutexes, condition variables, reports, and error
     propagation;
   - one worker per stage for the duration of one token-step batch;
   - stage completion counters indexed by slot;
   - final all-stage synchronization before returning.
3. Keep `replay_feed_token_batch` and `--wavefront-decode` intact. Selection
   order should be:
   - async pipeline if explicitly enabled and batch size is greater than one;
   - wavefront if explicitly enabled;
   - serial default otherwise.
4. Extend the benchmark harness to pass the async flag and record it in the
   summary.
5. Add a focused V100 smoke or reuse the sustained smoke to prove token
   correctness and no request errors.

## Parallel Work

Parallel agents can inspect and validate these areas independently:

- MTP commit feasibility, to decide whether Sprint 066 should pivot there.
- Stage-worker hazards: CUDA current-device behavior, shared globals, handoff
  synchronization, and per-device scratch safety.

## Files Summary

- `ds4_v100_replay.h`
- `ds4_v100_replay.c`
- `tools/ds4-v100-replay.c`
- `tools/ds4-v100-sustained-decode-bench.sh`
- `docs/sprints/SPRINT-065-REPORT.md`
- `docs/sprints/VISION.md`
- `logs/from-cluster/sprint065-*`

## Definition of Done

- [x] Local object builds pass for touched C files.
- [x] `bash -n tools/ds4-v100-sustained-decode-bench.sh` passes.
- [x] `git diff --check` passes.
- [x] V100 build passes for `tools/ds4-v100-replay` and scheduler/token smokes.
- [x] Existing V100 source/full scheduler/selected-token smokes still pass.
- [x] Async pipeline served smoke returns token hex `3136` with no request errors.
- [x] Paired async-vs-serial V100 benchmark artifacts are archived.
- [x] Sprint report records whether async pipeline improves generated and
  continuation tok/s enough to keep pursuing the path.

## Risks

- Some CUDA helper globals may still be process-global rather than
  per-device/thread-safe. Sprint 063 fixed the known temp scratch issue, but the
  async smoke must be treated as the real proof.
- The handoff API uses synchronous tensor copies. The pipeline can still
  overlap stage compute across slots, but handoff latency may limit gains.
- Spawning worker threads per token step may add overhead. If correctness
  passes but throughput is flat, the follow-up should move to persistent stage
  workers rather than per-step thread creation.
- Multi-slot MTP remains disabled in this sprint because MTP service state is
  currently one-slot diagnostic verify.

## Result

The opt-in async pipeline is the first practical execution-shape win. It keeps
token correctness and improves the paired V100 sustained decode matrix from
about `3.8` generated tok/s to `5.56-5.57` generated tok/s at 2 slots and
`8.67-8.68` generated tok/s at 4 slots. It remains opt-in because the current
worker implementation is per token-step; the next sprint should turn it into a
persistent stage-worker service before considering a default flip.
