# Sprint 434: Static Graph Route-Cap Probe

## Objective

Test whether the persistent graph routed FFN envelope can be reduced with a
static per-rank route cap while staying TP/EP-only and avoiding host route-count
readback.

No PP/layer-split variants are in scope.

## Implementation

Added a default-off diagnostic gate:

```text
--post-attention-static-rank-route-cap N
```

When combined with the post-attention fixed-capacity route planner, the gate:

- keeps the route-plan buffers sized at `slots * top_k`;
- keeps device route totals and route-total-aware pack/compose guards;
- launches routed FFN with `min(route_capacity, N)` rows per rank;
- emits `tp_ep_static_route_cap_audit` lines from device `d_route_totals`.

The overflow audit proves whether the chosen static cap covers the actual
per-rank route distribution for that layer.

## V100 Evidence

Build:

```text
gpu-01:/localpool/ds4/workspace/ds4-sprint181
CUDA_ARCH=sm_70 make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Logs:

```text
/localpool/ds4/workspace/logs/sprint434-static-route-cap/alllayers-cap16-slots8-graph.stdout
/localpool/ds4/workspace/logs/sprint434-static-route-cap/alllayers-cap32-slots8-graph.stdout
/localpool/ds4/workspace/logs/sprint432-device-actual-route-sync/alllayers-fixed-current-slots8-graph.stdout
/localpool/ds4/workspace/logs/sprint435-output-head-parity/fullcap-slots8-graph-head.stdout
/localpool/ds4/workspace/logs/sprint435-output-head-parity/cap16-slots8-graph-head.stdout
```

8 slots / 256K / persistent graph replay:

```text
full cap: projected_slot_step_tok_s=39.491776, aggregate_routes=384, checksum=3211778491
cap 32:   projected_slot_step_tok_s=44.163120, aggregate_routes=256, checksum=1709346105
cap 16:   projected_slot_step_tok_s=50.502275, aggregate_routes=128, checksum=6493007747
```

Overflow audit:

```text
cap 32: 0 overflow layers
cap 16: 0 overflow layers
max observed per-rank route total at cap 16: 16
```

Output-head parity follow-up:

```text
full cap: first_token=50845, first_logit=18.253084183, projected_slot_step_tok_s=36.846896
cap 16:   first_token=106720, first_logit=17.242784500, projected_slot_step_tok_s=50.408429
```

## Decision

Do not promote static route caps.

The performance direction is real, but the output-head follow-up proves that
changing TurboMind `total_tokens` / `Ddesc.rows` is not behavior-preserving:
cap 16 changed the actual selected token even though the cap covered all actual
route totals. The grouped SM70 scheduler and epilogue are sensitive to the
launch shape. Final graph checksums also change, but they are noisy across
repeated full-cap runs; token-level output parity is the stronger rejection
criterion.

The useful result is architectural:

- reducing the graph envelope is worth pursuing;
- host static caps are not the correct mechanism;
- the production path must leave the captured host launch shape intact and move
  the skip/no-op decision inside the TurboMind executor or a dedicated graph-safe
  routed FFN kernel.

## Next

Implement a true graph-safe routed executor experiment:

- keep `total_tokens = route_capacity` for host launch and capture;
- pass a device route-total/mask pointer into the TurboMind DS4 probe path;
- make inactive fixed-capacity CTAs/rows return before useful MMA or before
  output contribution;
- validate checksum parity against the full-cap graph baseline before reading
  throughput as a candidate metric.
