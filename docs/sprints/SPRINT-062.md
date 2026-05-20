# Sprint 062: Decode Timing And Execution-Shape Decision

## Status

Complete.

## Overview

Sprint 061 proved that shared F8 batching and additional active slots do not
materially improve aggregate throughput under the current layer-synchronous
schedule. The project needs a timing contract before making the next larger
change.

This sprint adds an opt-in decode timing surface that records where a generated
token spends time across the stage scheduler and layer executor. The output
should be concrete enough to decide between committed MTP, stage wavefronting,
or a focused kernel rewrite.

## Goals

1. Add low-overhead timing structs for stage/layer decode.
2. Keep timing disabled by default.
3. Emit timing summaries from replay/benchmark paths when enabled.
4. Capture at least one 1M two-slot V100 timing run and one shorter-context
   higher-slot run.
5. Use the timing evidence to choose the next implementation sprint.

## Out of Scope

- Shipping committed MTP draft acceptance.
- Implementing stage wavefront scheduling.
- Rewriting F8/MXFP4 kernels.
- Changing default benchmark semantics.

## Definition of Done

- Local syntax/object builds pass for touched files.
- `git diff --check` passes.
- V100 `sm_70` build passes for replay and scheduler smokes.
- Default replay still selects token hex `3136`.
- Timing-enabled replay or sustained benchmark emits stage/layer timing data.
- Sprint report maps measured time to a concrete next implementation choice.

## Outcome

`SHIP`. The replay tool now has an explicit `--profile-decode` option that
enables the existing synchronized decode profiler without changing default
runtime behavior. The sustained decode benchmark forwards that option and
preserves averaged `stage_profile_ms` arrays in each case result.

Cluster evidence was captured under
`logs/from-cluster/sprint062-profile/`. All four profiled cases returned token
hex `3136` with no HTTP errors or token mismatches.

The timing decision is to pursue an opt-in stage-wavefront execution proof next.
The current runtime remains stage-synchronous: stage-profile totals almost
exactly match stage-decode totals, and 4 active slots roughly double latency
without improving aggregate tok/s. That means the next useful serving win is
overlapping independent token/request microbatches across GPU stages, not more
slot admission, shared-F8 cleanup, or MTP commit work first.
