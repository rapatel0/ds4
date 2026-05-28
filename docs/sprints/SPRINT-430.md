# Sprint 430: Route-Total Gated Rank-Major Packing

## Objective

Reduce Sprint 429's fixed-capacity graph route-plan over-compute without
leaving the TP/EP-only persistent CUDA graph path.

No PP/layer-split work is in scope.

## Implementation

Changed the rank-major routed FFN input packers so they can consume the
device-resident per-rank route totals produced by
`--post-attention-fixed-capacity-route-plan-gate`.

When both gates are enabled:

```text
--routed-ffn-rank-major-route-input-gate
--post-attention-fixed-capacity-route-plan-gate
```

the route input packer now:

- keeps the graph launch shape static at `route_capacity`;
- reads `d_route_totals[rank]` on device;
- skips route-input packing for inactive fixed-capacity route rows;
- preserves the existing graph-safe post-attention route audit path.

This deliberately targets only one over-compute consumer. It does not yet make
the TurboMind routed FFN executor launch against actual route totals.

## V100 Evidence

Build:

```text
gpu-01:/localpool/ds4/workspace/ds4-sprint181
make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
CUDA_ARCH=sm_70
PASS
```

8-slot probe:

```text
log: /localpool/ds4/workspace/logs/sprint430-route-total-pack/alllayers-fixed-pack-gated-slots8.stdout
shape: 8 slots, 256K ctx, 43 layers, 1 decode step
mode: token-major all-layer direct decode, persistent CUDA graph replay
result: PASS
route audit: 43 layers, 2064 routes checked, 0 missing, 0 weight mismatch, 0 invalid slots
projected_slot_step_tok_s: 34.571189
```

Comparison:

```text
Sprint 429 fixed-capacity graph route plan: 34.738433 tok/s
Sprint 430 route-total gated packer:       34.571189 tok/s
delta:                                     -0.48%
```

## Decision

Do not promote this as a performance gate. It is correctness-clean, but the
topline is flat/slightly worse versus Sprint 429 and well inside expected run
noise.

The useful result is diagnostic: route input packing is not the dominant
fixed-capacity cost. The remaining over-compute is lower in the routed FFN
execution path, where each rank still reports:

```text
routes=48
route_capacity=48
active_local_experts=32
max_routes_per_expert=48
```

At 8 slots and top-k 6, the actual routed entries are 48 total, while the fixed
capacity executor still presents 384 aggregate rank routes.

## Next

Target the actual routed FFN execution boundary:

- keep graph-safe device route planning from Sprint 429;
- keep static graph shape for capture/replay;
- make the TurboMind grouped-GEMM path skip inactive route rows, or introduce a
  graph-safe compact executor keyed by fixed upper bounds but actual
  `route_totals`;
- then rerun the same 8-slot / 256K probe and compare against the Sprint 429
  `34.738433 tok/s` baseline.
