# Sprint 432: Graph-Safe Rank-Major FFN Half-Input Parity

## Objective

Continue the TP/EP-only rank-major path and determine whether post-attention
rank-major FFN inputs are still a correctness blocker under persistent CUDA
graph replay.

No PP/layer-split work is in scope.

## Context

Sprint 429 proved graph-safe fixed-capacity route planning. Sprint 430 showed
route-total-gated input packing was flat. Sprint 431 showed host route-count
oracles are not production-safe even though fewer executor rows improve
throughput.

The remaining question was whether rank-major `shared_gate`, `shared_up`, or
routed `route_a` half inputs still diverge in the graph replay regime.

## Implementation

Added post-replay half-input parity audit output:

```text
tp_ep_rank_major_half_input_diff
tp_ep_rank_major_half_input_diff_total
```

The audit runs after successful graph replay or persistent graph cache hit and
is guarded by the existing diagnostic:

```text
--routed-ffn-rank-major-input-parity-gate
```

It adds no production overhead when that gate is off.

The first audit showed false routed-input mismatches because fixed-capacity
graph route planning stores the actual route count in device `d_route_totals`,
while the host launch shape remains static. I added a route-total-limited
compare kernel so only active route rows are compared:

```text
compare_route_half_input_with_current_limited_kernel
collect_route_half_input_diff_limited
```

## V100 Evidence

Build:

```text
gpu-01:/localpool/ds4/workspace/ds4-sprint181
CUDA_ARCH=sm_70 make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Final run:

```text
log: /localpool/ds4/workspace/logs/sprint432-rank-major-parity/alllayers-combined-slots4-limited.stdout
shape: 4 slots, 256K ctx, 43 layers, 1 decode step/layer
mode: token-major all-layer direct decode, persistent CUDA graph replay
result: PASS
```

Route audit:

```text
43/43 layers
24 checked active routes per layer
0 missing selected experts
0 route weight mismatches
0 invalid slots
```

Half-input parity:

```text
43/43 layers PASS
graph_shared_gate mismatches: 0
graph_shared_up mismatches:   0
graph_route_a mismatches:     0
```

Diagnostic throughput:

```text
projected_slot_step_tok_s: 20.780357
checksum: 8339404968
```

This number is not a promotion metric because the run enables parity audit
and still uses fixed-capacity routed FFN execution.

## Decision

Rank-major FFN input layout is no longer the correctness blocker under graph
replay.

Do not spend another sprint on shared/route input parity unless a later
consumer change regresses it.

## Next

Implement the actual-route routed FFN executor:

- keep graph launch bounds static;
- consume device `d_route_totals` or an equivalent device mask;
- avoid running TurboMind/CUTLASS work for inactive fixed-capacity route rows;
- keep compact compose reading only active route indices;
- validate with 4-slot and 8-slot 256K graph replay before returning to the
  larger 28/32-slot serving path.
