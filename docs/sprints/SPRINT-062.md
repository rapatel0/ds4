# Sprint 062: Decode Timing And Execution-Shape Decision

## Status

In progress.

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
