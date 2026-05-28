# TEMP Status Report 430

Date: 2026-05-27

## Focus

TP/EP only. Tested whether fixed-capacity graph route-plan over-compute could
be reduced by gating rank-major route-input packing with device route totals.

## Added

The rank-major routed FFN input pack kernels now accept optional
`d_route_totals`. Under the fixed-capacity post-attention route planner, they
skip inactive route rows on device while keeping the CUDA graph launch shape
static.

## V100 Result

```text
log: /localpool/ds4/workspace/logs/sprint430-route-total-pack/alllayers-fixed-pack-gated-slots8.stdout
shape: 8 slots, 256K ctx, 43 layers, 1 decode step
43/43 layers PASS
route audit: 2064 checked, 0 missing, 0 weight mismatch, 0 invalid
projected decode: 34.571189 tok/s
```

Baseline:

```text
Sprint 429 fixed-capacity route plan: 34.738433 tok/s
Sprint 430 gated route packer:       34.571189 tok/s
```

## Interpretation

Correctness remains clean, but route-input packing is not the missing lever.
The fixed-capacity cost is still dominated by the routed FFN executor shape:
each rank launches against `route_capacity=48`, producing 384 aggregate rank
routes for only 48 actual routed entries at 8 slots/top-k 6.

## Current Blocker

Need graph-safe actual-route routed FFN execution. The next useful change is
not more input-packer gating; it is making TurboMind grouped GEMM skip inactive
fixed-capacity route rows or adding a compact graph-safe routed executor with
static upper bounds and device actual counts.
