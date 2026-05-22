# Sprint 172 - Small Route Builder Recheck

Date: 2026-05-22

## Objective

Recheck the existing `DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=1` path on the
current production 16-slot/256K served appliance before moving to a larger
persistent routed-FFN or TP/EP boundary.

The served hot path presents only six routed rows per request. The normal
TurboMind route builder uses separate count, prefix, and scatter kernels. The
small-route path collapses route construction into one kernel for
`total_routes <= 128`, so it is the last low-risk route-construction lever to
settle.

## Scope

- No new runtime code.
- Run 16-slot/256K served appliance soaks with:
  - `DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=0`
  - `DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=1`
- Use the current production gates:
  - `DS4_V100_TURBOMIND_GATED_SILU=1`
  - `DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1`
  - `DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=0`
  - `DS4_V100_TURBOMIND_GRAPH=0`
  - per-step async pipeline
  - event handoff
- Keep defaults unchanged unless the repeat evidence is clearly positive.

## Definition of Done

- [x] Candidate 16-slot/256K served run completed.
- [x] Control repeat completed.
- [x] Candidate repeat completed.
- [x] Token correctness preserved.
- [x] Result recorded in cluster logs.
- [x] Result recorded in `docs/sprints/VISION.md`.
- [x] Changes committed.

## Outcome

All runs completed on the V100 pod (`llm/llamacpp-build-8gpu`, gpu-01) with
`16/16` token match.

| Mode | Generated tok/s | Continuation tok/s | Prompt tok/s | Token match |
|---|---:|---:|---:|---:|
| small-route candidate | `46.550101` | `43.640720` | `52.368864` | `16/16` |
| control repeat | `46.136775` | `43.253227` | `51.903872` | `16/16` |
| small-route repeat | `45.927784` | `43.057298` | `51.668757` | `16/16` |

Nearby controls from Sprint 171 were:

| Mode | Generated tok/s | Continuation tok/s | Prompt tok/s | Token match |
|---|---:|---:|---:|---:|
| Sprint 171 control | `45.941120` | `43.069800` | `51.683760` | `16/16` |

The first small-route run was mildly positive, but the repeat fell below the
fresh control repeat. Averaging the two candidates against the two nearby
controls is only about `0.4%` positive, which is within run noise.

Evidence: `logs/from-cluster/sprint172-small-route-build/`.

## Decision

Keep `DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=0` as the production default.

This closes the last cheap route-construction variant. The next material work
should stop testing wrapper/layer variants and move to one of the larger
execution-boundary options:

- persistent routed-FFN executor: gate/up + gated-SiLU + down + route reduce as
  one persistent or fused boundary
- persistent TP/EP scheduler boundary: tensor/expert parallel execution with
  copy and launch overhead designed into the boundary rather than bolted onto a
  single-layer overlay
