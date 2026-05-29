# Sprint 567 - C1 Route Replay Boundary Repair

Date: 2026-05-29

## Goal

Localize and repair the route-plan/route-pack replay boundary that Sprint 566
identified as the next full-capture cross-position blocker.

## Context

Sprint 566 proved the rank-major HC-current clue was a timing artifact. The
first comparable end-of-step drift under temporary relaxed-position replay is
now `route_a`: the first cross-position replay returns the correct token but
leaves route scratch different across all layers/ranks; the following replay
turns into layer-1 visible drift and a selected-token mismatch.

`route_a` is a scratch tensor with active and inactive fixed-capacity rows. A
full-buffer hash cannot distinguish semantic active-route drift from harmless
inactive-tail garbage. The next diagnostic must include the route metadata that
defines active rows.

## Plan

1. Extend the existing `--decode-stage-checksum-gate` `step_snapshot` to log
   route metadata alongside `route_a`: route totals, route slots, and route
   weights.
2. Rebuild on the V100 node from a clean rsync.
3. Re-run the temporary relaxed-position three-request diagnostic.
4. If active route metadata differs, repair that path. If only inactive scratch
   differs, move the investigation to the first active output tensor that
   diverges.

## Definition of Done

- Remote V100 container build passes from a clean rsync.
- The relaxed diagnostic identifies whether `route_a` drift is semantic active
  route drift or inactive scratch drift.
- Full capture remains position-keyed unless comparable snapshots become clean.

## Result

Implemented route metadata snapshot logging under the existing diagnostic gate:

- `route_totals`
- `route_slots`
- `route_weights`

Validation:

- Remote clean production build passed in `/workspace/s567-route-replay`.
- A temporary relaxed-position build passed in the same remote tree.
- The eager-vs-relaxed diagnostic under `/workspace/s567-stage-artifacts`
  reproduced the request-3 selected-token/checksum failure.

The eager-vs-relaxed route metadata was not sufficient because graph mode uses
the fixed-capacity route planner even on cache misses. A cleaner graph-vs-graph
comparison was then run:

- position-keyed graph control: `/workspace/s567-route-control`
- relaxed-position graph candidate: `/workspace/s567-route-replay`
- artifacts: `/workspace/s567-graph-compare-artifacts`

Graph-vs-graph response result:

- control: `24426` / `128829740021`, `2039` / `106648190597`,
  `117465` / `17092309830`
- relaxed: `24426` / `128829740021`, `2039` / `106648190597`,
  `128818` / `81184816026`

Graph-vs-graph `step_snapshot` result:

- occurrence 0 matched completely.
- occurrence 1 differed only in full-buffer `route_a`; route totals, route
  slots, route weights, current tensors, compose tensors, and final-HC tensors
  matched, and the selected token/checksum matched.
- occurrence 2 again had only layer-0 `route_a` scratch drift at layer 0, then
  layer 1 and later diverged broadly across current, post-attention, compose,
  and final-HC tensors. Route totals began differing after the layer-1 current
  drift, not before it.

## Decision

`route_a` full-buffer drift is not the semantic blocker. It is scratch-state
noise visible in the broad diagnostic hash; it appears while route metadata and
served outputs still match.

Full capture remains position-keyed. The next repair should return to the
inter-layer current/HC handoff, with the narrower hypothesis that CUDA graph
replay and eager host pointer swaps can make the next layer read a captured
buffer address that is not the current post-layer hidden state, even when the
post-layer snapshot logs the expected final-HC buffer after replay.
