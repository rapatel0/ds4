# Sprint 431: Actual-Route Executor Boundary Investigation

## Objective

Determine whether Sprint 429's fixed-capacity routed FFN cost is worth
attacking at the TurboMind executor boundary, while keeping the TP/EP path
separate from all PP/layer-split work.

## Implementation

Kept the production tree conservative:

- wired existing route-total-aware compose kernels through their current call
  sites so the final harness builds cleanly;
- did not keep the host route-count oracle gate after testing, because it is
  not correctness-preserving for persistent graph replay.

The rejected oracle attempted to seed host `r.routes` before graph capture and
then let persistent graph replay run with fewer TurboMind rows than
`route_capacity`.

## V100 Evidence

Final build:

```text
gpu-01:/localpool/ds4/workspace/ds4-sprint181
make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
CUDA_ARCH=sm_70
PASS
```

Rejected oracle logs:

```text
/localpool/ds4/workspace/logs/sprint431-host-route-count-oracle/alllayers-oracle-slots8.stdout
/localpool/ds4/workspace/logs/sprint431-host-route-count-oracle/alllayers-oracle-slots8-clean.stdout
```

Clean repeat topline:

```text
shape: 8 slots, 256K ctx, 43 layers, 1 decode step
route audit: 43 layers, 2064 checked, 0 missing, 0 weight mismatch, 0 invalid
projected_slot_step_tok_s: 44.270973
aggregate_routes reported to executor: 48
ep_return_bytes: 688128
```

Baseline comparison:

```text
Sprint 429 fixed capacity: 34.738433 tok/s, aggregate_routes=384, ep_return_bytes=5505024
Sprint 430 gated packer:  34.571189 tok/s, aggregate_routes=384, ep_return_bytes=5505024
Sprint 431 oracle:        44.270973 tok/s, aggregate_routes=48,  ep_return_bytes=688128
```

## Decision

Do not promote the oracle. It is a directional measurement only.

Reason: the post-attention router distribution is produced inside the captured
path. A warmup/host snapshot can preserve a smaller `r.routes` launch shape,
but it does not guarantee the same per-rank route totals during capture/replay.
The logs show fixed `routes=6` per rank while the route audit for the same
captured layers reports imbalanced counts such as `6,8,4,0,4,7,8,11`.

That means the oracle can under-execute routed rows even while the audit checks
the device route plan itself. It is useful proof that row count matters, not a
valid serving candidate.

## Next

Implement actual-route execution without host oracle semantics:

- keep the graph-safe device route plan from Sprint 429;
- add a TurboMind/DS4 routed executor ABI that takes a device route mask or
  per-rank route totals and skips inactive fixed-capacity rows internally; or
- compact active rows into a fixed upper-bound graph buffer on device and run
  grouped GEMM over a static launch envelope with device-side no-op guards.

The production target is still persistent graph replay with correct
post-attention route metadata and no host route-count dependency.
