# Sprint 066: Persistent Async Stage Workers

## Status

Complete.

## Overview

Sprint 065 proved that true per-stage host concurrency is a real throughput
lever: 4 active slots rose from about `3.8` generated tok/s to about `8.7`.
The implementation still creates and joins eight worker threads for every
token-step batch. Sprint 066 keeps the same opt-in serving contract but turns
those workers into a persistent replay-runtime service.

## Outcome

`SHIP` as an opt-in implementation detail, but do not make async pipeline
decode the default yet.

Persistent workers preserve correctness and remain materially faster than the
serial stage-synchronous path. The paired V100 matrix measured `5.132695`
generated tok/s at 1M/2 slots and `7.942345` at 1M/4 slots, versus serial
`3.851964` and `3.788708`. However, this is slower than Sprint 065's per-step
worker path (`5.571149` and `8.668248`), so the next performance sprint should
profile the persistent dispatch/handoff synchronization before treating it as a
default-ready path.

## Goals

1. Start one async pipeline worker per V100 stage when replay opens with
   `--async-pipeline-decode`.
2. Reuse those workers across prompt replay and continuation token steps.
3. Preserve the Sprint 065 correctness and benchmark contract.
4. Measure whether persistent workers improve 2-slot and 4-slot sustained
   throughput.
5. Decide whether the async path is close enough to default-ready or needs more
   stream/copy work first.

## Non-Goals

- Enabling async pipeline by default.
- MTP commit or MTP batching.
- CUDA stream/event rewrites.
- Replacing blocking `cudaMemcpyPeer` handoff.
- Changing queue/admission semantics.

## Implementation

1. Add a replay-owned `replay_pipeline_runtime` with:
   - one `pthread_t` per stage;
   - one shared mutex/condition variable;
   - per-dispatch token/position pointers, slot count, completion state, and
     reports;
   - shutdown/error handling.
2. Create the runtime during `ds4_v100_replay_open` when
   `async_pipeline_decode` is enabled.
3. Stop and join workers in `ds4_v100_replay_close` before scheduler teardown.
4. Replace per-token-step worker creation in
   `replay_feed_token_batch_async_pipeline` with a dispatch to the persistent
   runtime.
5. Preserve fallback behavior:
   - one-slot batches use the serial batch path;
   - non-async mode remains serial by default;
   - `--wavefront-decode` remains a separate diagnostic path.

## Definition of Done

- [x] Local object builds pass for touched C files.
- [x] `bash -n tools/ds4-v100-sustained-decode-bench.sh` passes.
- [x] `git diff --check` passes.
- [x] V100 build passes for `tools/ds4-v100-replay` and scheduler/token
  smokes.
- [x] Existing V100 source/full scheduler/selected-token/wavefront smokes pass.
- [x] Async persistent served smoke returns token hex `3136` with no request
  errors.
- [x] Paired persistent-async vs serial V100 benchmark artifacts are archived.
- [x] Sprint report records whether persistent workers improve over Sprint 065.

## Risks

- Persistent workers make replay close/error handling more important; the
  workers must stop before stage scheduler teardown.
- If the per-step thread creation overhead was small, performance may be flat.
- The pipeline still uses blocking handoff and per-slot device synchronizes, so
  stream/copy work may remain the next bottleneck.
