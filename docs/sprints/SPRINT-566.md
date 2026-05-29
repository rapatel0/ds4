# Sprint 566 - C1 Rank-Major Replay Snapshot Repair

Date: 2026-05-29

## Goal

Explain the Sprint 565 `current_full_rank_major` divergence without changing
the served default or relaxing the no-suffix full-capture position key.

## Context

Sprint 565 compared eager per-stage checksum logs against full-graph replay
checksum logs. That localized the first sequential difference to layer 0
`hc_current.current_full_rank_major`, while adjacent `current_shard` and
slot-major `current_full` records matched.

That evidence is suspicious because the eager log is captured immediately after
the HC-current stage, but a full-graph replay can only snapshot buffers after
the entire captured graph has completed. Later stages are allowed to reuse or
refresh rank-major scratch buffers, so the old diagnostic may be comparing two
different buffer lifetimes.

## Plan

1. Keep full capture position-keyed.
2. Improve `--decode-stage-checksum-gate` so it can distinguish real tensor
   drift from a weak checksum or timing artifact.
3. Add a comparable end-of-step tensor snapshot for both eager and replay paths
   under the existing diagnostic gate.
4. Re-run the relaxed-position diagnostic only after rebuilding a clean remote
   copy from the current tree.

## Definition of Done

- Remote V100 container build passes from a clean rsync.
- The relaxed-position diagnostic either:
  - identifies the first real tensor drift from comparable end-of-step
    snapshots, or
  - proves the Sprint 565 rank-major HC-current difference was only a
    diagnostic timing artifact and updates the steering accordingly.
- No new production feature flag is added.

## Result

Implemented two diagnostic-only improvements under the existing
`--decode-stage-checksum-gate`:

- `diagnostic_checksum_bytes_kernel`, an order-sensitive 64-bit diagnostic hash
  for stage checksums. The existing pack checksum kernel is unchanged.
- `step_snapshot` records for both eager and full-graph replay, logged at a
  comparable end-of-step point instead of comparing eager mid-stage logs with
  replay post-graph snapshots.

Validation:

- Remote clean production build passed in `/workspace/s566-rank-major-snapshot`.
- A remote-only relaxed-position diagnostic build also passed and ran the
  three-request same-session selected-token probe with artifacts under
  `/workspace/s566-stage-artifacts`.

Response-level diagnostic result:

- eager: `24426` / `128829740021`, `2039` / `106648190597`,
  `117465` / `17092309830`
- relaxed replay: `24426` / `128829740021`, `2039` / `106648190597`,
  `128818` / `81184816026`

Comparable `step_snapshot` result:

- occurrence 0 matched completely.
- occurrence 1 matched selected token/checksum and every step-snapshot tensor
  except `route_a` across all layers/ranks.
- occurrence 2 first had only layer-0 `route_a` drift; layer-1 and later then
  diverged broadly across current, post-attention, route, compose, and final-HC
  tensors.

## Decision

The Sprint 565 `hc_current.current_full_rank_major` finding was a diagnostic
timing artifact: eager logged HC-current immediately after the HC-current stage,
while full-graph replay can only inspect that buffer after the entire graph has
completed.

Full capture remains position-keyed. The next C1 repair should focus on the
route-plan/route-pack replay boundary, especially why `route_a` scratch state
differs after a cross-position replay that still returns the correct token, and
why the next replay turns that state into layer-1 visible drift. Do not retry
the no-suffix full-capture position-key relaxation until the route replay path
has a clean comparable snapshot.
