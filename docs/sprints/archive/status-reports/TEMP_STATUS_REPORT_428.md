# TEMP Status Report 428

Date: 2026-05-27

## Focus

TP/EP only. Sprint 428 isolated the persistent CUDA graph/rank-major FFN
blocker by auditing post-attention route metadata during graph replay.

## What Changed

- Added default-off `--post-attention-route-reuse-audit-gate`.
- Added graph-capturable route-audit buffers per rank.
- Added graph-safe int/float device-copy path for router selected experts and
  weights during capture.
- Added per-rank and aggregate audit output:
  `tp_ep_post_attention_route_reuse_audit*`.

## V100 Result

Build passed on `gpu-01` with `sm_70`.

Primary evidence:

```text
log: /localpool/ds4/workspace/logs/sprint428-post-attn-route-audit/alllayers-audit-local-experts.stdout
shape: 8 slots, 256K ctx, 43 layers, 1 decode step
mode: token-major all-layer direct decode, persistent CUDA graph replay
expert residency: local per-layer bindings
```

Topline:

```text
43/43 layer invocations passed
43/43 CUDA graph captures replayed successfully
sum graph replay: 178.898944 ms
projected decode: 44.717983 slot-step tok/s
```

Route audit:

```text
routes_checked:   2064
missing_selected: 2014
weight_mismatch:  50
invalid_slot:     0
```

## Interpretation

The rank-major half-input math is not the blocker. Sprint 427 already proved
byte parity in eager synchronous route planning.

The persistent graph path is reusing HC-current route metadata for the
post-attention FFN. When the post-attention router is recomputed inside graph
capture, essentially every reused route disagrees with the recomputed
post-attention selected experts or route weight.

## Blockers / Notes

- A first diagnostic implementation used `cudaMemcpyPeerAsync` inside capture
  and failed with `operation not permitted when stream is capturing`; fixed by
  using graph-capturable copy kernels.
- Full shared expert residency currently OOMs in the contiguous TurboMind pack
  allocation before graph capture. The successful audit used local expert
  bindings to keep the route-plan experiment isolated.

## Next

Implement a real graph-safe post-attention route planner:

- device-only count/prefix/fill from post-attention router selected experts;
- fixed route capacity or another static graph shape;
- no host route-count reads during capture/replay;
- compact route metadata updated on device;
- prove eager-vs-graph parity at 8 slots / 256K before promoting rank-major
  FFN input or moving back to HTTP serving benchmarks.
