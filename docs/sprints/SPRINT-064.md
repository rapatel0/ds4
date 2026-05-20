# Sprint 064: Opt-In Served Wavefront Decode

## Status

In progress.

## Overview

Sprint 063 proved slot-lane scheduler mechanics for two-stage wavefront order.
Sprint 064 wires those primitives into the replay server's same-length non-MTP
batch path behind an opt-in flag and measures whether stage wavefronting beats
the Sprint 062 profiled sustained baseline.

## Goals

1. Add an opt-in replay/runtime flag for wavefront batch decode.
2. Preserve default serial stage scheduling and all existing APIs.
3. Route same-length non-MTP server batches through wavefront decode when the
   flag is enabled.
4. Extend the sustained benchmark launcher to pass the flag and report status.
5. Validate correctness and measure V100 throughput against Sprint 062 cases.

## Out of Scope

- Enabling wavefront decode by default.
- MTP wavefronting or MTP draft commit.
- Kernel rewrites.
- Changing admission limits beyond the explicit opt-in path.

## Definition of Done

- Local syntax/object builds pass for touched files.
- `git diff --check` passes.
- V100 replay/server build passes.
- Existing V100 source/full scheduler/selected-token smokes still pass.
- Opt-in served wavefront benchmark returns token hex `3136` with no errors.
- Sprint report records whether wavefront improves aggregate tok/s enough to
  continue toward default serving.
