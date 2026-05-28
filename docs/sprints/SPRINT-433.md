# Sprint 433: Device-Actual Route Sync Diagnostic

## Objective

Quantify whether using device-produced actual route totals reduces routed FFN
work enough to justify a graph-safe actual-route executor.

This sprint stays TP/EP-only. No PP or layer-split variants are in scope.

## Implementation

Added an opt-in diagnostic gate:

```text
--post-attention-device-actual-route-sync-gate
```

The gate uses the existing post-attention GPU route planner, then synchronizes
rank streams and reads back only the eight `d_route_totals` values. It updates
host `RankState::routes` from the actual device route totals before launching
the routed FFN executor.

It deliberately fails closed under graph-event ordering:

```text
--post-attention-device-actual-route-sync-gate
--decode-cudagraph-persistent-replay-gate
```

is not a production-valid combination, because the host route-count readback
cannot be captured into persistent CUDA graph replay.

## V100 Evidence

Build:

```text
gpu-01:/localpool/ds4/workspace/ds4-sprint181
CUDA_ARCH=sm_70 make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Logs:

```text
/localpool/ds4/workspace/logs/sprint432-device-actual-route-sync/alllayers-actual-sync-slots8.stdout
/localpool/ds4/workspace/logs/sprint432-device-actual-route-sync/alllayers-actual-sync-slots8-skipstats.stdout
/localpool/ds4/workspace/logs/sprint432-device-actual-route-sync/alllayers-fixed-current-slots8-skipstats.stdout
/localpool/ds4/workspace/logs/sprint432-device-actual-route-sync/alllayers-fixed-current-slots8-graph.stdout
```

8 slots / 256K / direct mode / actual-route sync / skip stats:

```text
result: PASS
projected_slot_step_tok_s: 17.260141
aggregate_routes: 48
ep_return_bytes: 688128
```

8 slots / 256K / direct mode / current fixed route-plan / skip stats:

```text
result: PASS
projected_slot_step_tok_s: 17.371569
aggregate_routes: 48
ep_return_bytes: 688128
```

8 slots / 256K / persistent graph replay / current fixed route-plan:

```text
result: PASS
projected_slot_step_tok_s: 39.491776
aggregate_routes: 384
ep_return_bytes: 5505024
layer-42 graph replay: 1395.338447 slot-step tok/s
```

## Decision

Do not promote host-synchronized actual-route execution.

The diagnostic is valid and useful, but it does not improve direct-mode
throughput. More importantly, it is structurally incompatible with persistent
graph replay, which is the faster serving-style path.

The important result is the separation of regimes:

- direct mode can already operate with actual route totals;
- persistent graph replay still carries the fixed per-rank route envelope;
- the production fix must be device-side and graph-safe.

## Next

Implement the actual-route routed FFN executor at the graph boundary:

- keep graph launch topology static;
- consume device `d_route_totals`, offsets, or a route mask inside the executor;
- avoid useful TurboMind/CUTLASS work for inactive fixed-capacity rows;
- keep host launch parameters fixed so persistent replay remains valid;
- validate 4-slot and 8-slot 256K graph replay before returning to 28/32 slots.
