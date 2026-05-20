# Sprint 070: Persistent MTP Forward Runtime

## Status

Complete.

## Overview

Sprint 069 proved the practical 4-slot appliance launcher path, but the next
throughput gain must come from real execution work. The current MTP serving path
is still diagnostic verify, and its forward helper allocates/free's all scratch
tensors for every draft. That shape is not suitable for true MTP commit.

Sprint 070 makes the MTP forward object production-shaped: allocate reusable
scratch once at `ds4_v100_mtp_forward_open`, reuse it for each draft, keep the
current verify semantics unchanged, and expose enough timing/memory evidence to
decide whether MTP commit is a viable next sprint.

## Goals

1. Move per-draft MTP scratch tensor allocation out of
   `ds4_v100_mtp_forward_run_host`.
2. Keep MTP serving correctness unchanged: exact top-1 verify, first token hex
   `3136`, and existing MTP response JSON.
3. Add MTP forward runtime counters/report fields for scratch residency and
   draft call count where useful.
4. Validate the MTP serving smoke on the V100 cluster with the real base model,
   real MTP sidecar, and real pack index.
5. Compare post-change `mtp.draft_ms` against the prior served baseline
   (`~4.6 ms` from Sprint 045) and record whether the change materially reduces
   MTP overhead.

## Non-Goals

- True speculative token commit.
- Recursive MTP drafting.
- Enabling MTP by default in the practical multi-slot appliance profile.
- Rewriting the base target verifier.
- Changing async pipeline scheduling.

## Implementation

1. Introduce a private scratch struct inside
   `tools/ds4-v100-mtp-forward-common.c`.
2. Allocate the existing MTP forward tensors and host logits buffer once in
   `ds4_v100_mtp_forward_open`.
3. Reuse those tensors in `ds4_v100_mtp_forward_run_host`; keep the current
   raw-cache reset behavior so Sprint 045 verify semantics do not change.
4. Free scratch tensors in `ds4_v100_mtp_forward_close`.
5. Extend `ds4_v100_mtp_forward_report` with scratch/runtime fields if this can
   be done without disturbing existing JSON consumers.
6. Update `tools/ds4-v100-replay.c` MTP JSON/status if new report fields are
   added.
7. Add Sprint 070 report and update `docs/sprints/VISION.md`.

## Parallel Work

- MTP commit explorer: identify the smallest next step from persistent scratch
  to target-state commit.
- Async handoff explorer: verify whether stream/event handoff is likely to beat
  the current per-step async path enough to supersede MTP work.

## Definition of Done

- [x] Local object compile passes for changed C files.
- [x] `git diff --check` passes.
- [x] CUDA build on the V100 cluster passes for `tools/ds4-v100-replay` and
  `tools/ds4-v100-mtp-verify-smoke`.
- [x] Focused V100 MTP serving smoke passes with `accepted=true`, first token
  hex `3136`, and no readiness regression.
- [x] MTP draft timing is recorded before/after or compared to the Sprint 045
  baseline.
- [x] Sprint report records commands, outputs, artifacts, and the decision on
  whether Sprint 071 should implement true MTP commit or pivot to another
  throughput lever.
- [x] Vision document is updated.

## Outcome

`SHIP`.

The MTP forward helper now owns reusable resident scratch for each opened MTP
service instead of allocating every draft call. The serving path reports scratch
residency and a monotonic forward run counter, and MTP serving is explicitly
one-slot while the shared verify state remains diagnostic-only.

The V100 serving smoke accepted `3/3` drafts with first token hex `3136`.
Measured draft timing stayed in the same range as the Sprint 045 baseline:
`4.800 ms`, `4.560 ms`, and `4.562 ms`. That means scratch allocation was not
the dominant MTP draft cost, but the persistent runtime shape is now ready for a
true one-slot commit API.

## Risks

- Per-call allocation may not dominate `draft_ms`; if timing is flat, the sprint
  still establishes the stateful runtime object needed for future commit.
- Reusing scratch across requests can accidentally preserve draft state. Keep the
  current raw reset semantics for this sprint.
- The MTP helper is tool-local today. If it becomes production runtime, it may
  need to move out of `tools/` in a later cleanup sprint.

## Security

No new external serving surface. MTP remains opt-in and loopback-only through
the existing appliance endpoints.
