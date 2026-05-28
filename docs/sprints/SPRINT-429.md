# Sprint 429: Fixed-Capacity Graph Route Plan

## Objective

Convert Sprint 428's route-plan audit into a first graph-safe post-attention
route planner for the TP/EP path.

No PP/layer-split work is in scope.

## Implementation

Added default-off:

```text
--post-attention-fixed-capacity-route-plan-gate
```

When enabled during persistent CUDA graph capture/replay, the post-attention
FFN path now:

- recomputes router logits and selected experts from the post-attention
  FFN-normalized hidden state;
- copies selected experts and route weights through graph-capturable device
  copy kernels;
- performs device-only route count, prefix, offset copy, route-slot fill, route
  weight fill, and compact-route metadata initialization;
- avoids host route-count reads during capture/replay;
- keeps host route launch shape static by setting each rank to fixed capacity
  `slots * top_k`;
- can run the Sprint 428 audit after filling the device route plan.

This is intentionally a correctness-first fixed-capacity implementation. It
over-computes by launching each rank at full capacity, so it is not a promotion
candidate yet.

## V100 Evidence

Build:

```text
gpu-01:/localpool/ds4/workspace/ds4-sprint181
make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
CUDA_ARCH=sm_70
PASS
```

4-slot probe:

```text
log: /localpool/ds4/workspace/logs/sprint429-post-attn-fixed-route/alllayers-fixed-slots4.stdout
shape: 4 slots, 256K ctx, 43 layers, 1 decode step
mode: token-major all-layer direct decode, persistent CUDA graph replay
result: PASS
route audit: 43 layers, 1032 routes checked, 0 missing, 0 weight mismatch, 0 invalid slots
projected_slot_step_tok_s: 24.292901
```

8-slot probe:

```text
log: /localpool/ds4/workspace/logs/sprint429-post-attn-fixed-route/alllayers-fixed-slots8.stdout
shape: 8 slots, 256K ctx, 43 layers, 1 decode step
mode: token-major all-layer direct decode, persistent CUDA graph replay
result: PASS
route audit: 43 layers, 2064 routes checked, 0 missing, 0 weight mismatch, 0 invalid slots
projected_slot_step_tok_s: 34.738433
```

Both probes used local per-layer expert bindings because the current shared
contiguous TurboMind expert residency path OOMs before graph capture.

## Decision

The route correctness blocker is cleared in a diagnostic graph path. The fixed
capacity route plan is not ready for promotion because every rank launches at
`slots * top_k` routes:

```text
4 slots: aggregate_routes = 192 instead of actual 24
8 slots: aggregate_routes = 384 instead of actual 48
```

That deliberate over-compute explains the lower decode throughput versus the
Sprint 428 stale-route audit run.

## Next

Replace fixed-capacity over-compute with a graph-safe static launch shape that
still skips inactive routes:

- keep graph shape static, but use device `route_totals` / compact counts to
  gate work inside kernels;
- avoid making TurboMind grouped GEMM process inactive rows;
- keep compact compose reading only actual route indices;
- then rerun same-binary eager-vs-graph parity and 8-slot / 256K performance.
