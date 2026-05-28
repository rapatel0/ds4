# Sprint 428: Graph-Safe Post-Attention Route Plan

## Objective

Fix the remaining rank-major FFN input blocker by making post-attention route
planning graph-safe instead of reusing stale HC-current route metadata during
persistent graph replay.

No PP/layer-split work is in scope.

## Context

Sprint 427 proved that rank-major shared and routed half inputs are
byte-identical to the legacy slot-major half inputs when route planning is
synchronous and recomputed from post-attention FFN norm/router state.

That means the Sprint 425 persistent-graph divergence is not caused by the
rank-major half-input kernels. The remaining suspect is route metadata:

- eager synchronous route-plan upload recomputes the post-attention route plan
  and matches control;
- persistent graph replay skips post-attention route recomputation through
  `reuse_model_router_route_plan`;
- the existing async/GPU route plan helpers still rely on host-side route counts
  and are not sufficient as a fully graph-safe dynamic route planner.

## Implementation Direction

Add a default-off graph-safe route-plan mode for post-attention FFN:

```text
--post-attention-fixed-capacity-route-plan-gate
```

The mode should:

- keep route capacity fixed at `slots * top_k` per rank for graph capture;
- write `d_route_slots`, `d_route_weights`, offsets, and compact route plan on
  device from `d_router_selected` / `d_router_weights`;
- zero or mark inactive routes instead of changing host `r.routes` during graph
  replay;
- keep host-side `r.routes` at fixed capacity while kernels consume route
  weights / slots to skip inactive work where possible;
- avoid device-to-host route count reads inside captured graph replay;
- preserve the existing synchronous route-plan path as the correctness oracle.

If full fixed-capacity integration is too large for one sprint, first implement
a graph-safe diagnostic that emits device-resident route-plan checksums for
control, shared-only, and route-only graph runs.

## Definition of Done

- V100 sm_70 build passes.
- The new route-plan mode is default-off and visible in CLI usage/scaffold
  logging.
- A graph-mode run no longer reuses HC-current route metadata for
  post-attention FFN when the gate is enabled.
- `8` slot / `256K` persistent-graph shared-only and route-only runs complete
  without the Sprint 425 checksum divergence, or the sprint records a narrower
  graph-safe route-plan blocker with route metadata evidence.
- Rank-major FFN input gates remain default-off until graph-mode checksum parity
  is proven.

## Outcome

Implemented the narrower graph-safe diagnostic:

```text
--post-attention-route-reuse-audit-gate
```

The gate is default-off. During persistent CUDA graph capture/replay it:

- recomputes the post-attention router selection from the post-attention
  FFN-normalized hidden state;
- copies selected experts and route weights through graph-capturable device
  copy kernels instead of graph-illegal peer memcpy calls;
- compares the recomputed post-attention selected experts against the reused
  route slots/offsets on each rank;
- prints per-rank and aggregate route-plan mismatch counters after graph
  replay.

## V100 Evidence

Build:

```text
gpu-01:/localpool/ds4/workspace/ds4-sprint181
make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
CUDA_ARCH=sm_70
PASS
```

Primary run:

```text
log: /localpool/ds4/workspace/logs/sprint428-post-attn-route-audit/alllayers-audit-local-experts.stdout
shape: 8 slots, 256K ctx, 43 layers, 1 decode step
mode: token-major all-layer direct decode, persistent CUDA graph replay
experts: local per-layer bindings, because current shared contiguous expert
         bindings OOM before graph capture
```

Topline:

```text
pass_invocations: 43/43
cudagraph captures: 43
cudagraph replay sum: 178.898944 ms
projected_slot_step_tok_s: 44.717983
serving_bench aggregate_generated_tok_s_decode: 44.717983
```

Route reuse audit:

```text
audit_layers=43
routes_checked=2064
missing_selected=2014
weight_mismatch=50
invalid_slot=0
```

This proves the persistent-graph path is usually executing the post-attention
FFN with route metadata from the earlier HC-current router state. The mismatch
rate is effectively 100%: every checked route is either assigned to an expert
that the post-attention router did not select for that slot, or has a mismatched
route weight.

Additional blockers found:

- The original diagnostic attempt used `cudaMemcpyPeerAsync` inside capture and
  failed with `operation not permitted when stream is capturing`; the fix was to
  use device copy kernels for graph capture.
- Full shared expert residency currently OOMs in the contiguous TurboMind pack
  allocation before graph capture:
  `pack_descriptor_set cudaMalloc(&out->d_w_contiguous): out of memory`.
  The route-plan evidence above used local expert bindings to keep the probe
  moving. This is a memory-layout issue, not a route-plan correctness result.

## Decision

Do not promote `--routed-ffn-rank-major-input-gate` yet. Sprint 427 proved the
rank-major half inputs are byte-identical in eager synchronous route planning;
Sprint 428 proves persistent graph replay is using stale post-attention route
metadata.

Next implementation sprint should build the actual graph-safe post-attention
route plan:

- device-only count/prefix/fill from post-attention selected experts;
- no host route-count reads during capture/replay;
- fixed route capacity or equivalent static graph shape;
- compact-route metadata updated on device;
- same-binary eager-vs-graph parity at 8 slots / 256K before HTTP promotion.
