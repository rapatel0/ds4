# TEMP Status Report 432

## Current Focus

TP/EP only. The current rank-major direction is to remove device-0 full-hidden
staging and keep tensors in rank-major/rank-local layout unless a true
collective is mathematically required.

## What Changed

Added a graph-replay-safe rank-major FFN half-input parity audit in
`tools/ds4-v100-tp-ep-full-layer-smoke.cu`.

The audit runs only when:

```text
--routed-ffn-rank-major-input-parity-gate
```

is enabled, so it has no production overhead by default.

Added a route-total-aware routed input compare so fixed-capacity graph route
planning compares only active route rows from `d_route_totals`.

## V100 Result

Build passed on `gpu-01`:

```text
CUDA_ARCH=sm_70 make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
```

Run artifact:

```text
/localpool/ds4/workspace/logs/sprint432-rank-major-parity/alllayers-combined-slots4-limited.stdout
```

Shape:

```text
4 slots
256K context
43 layers
1 decode step/layer
persistent CUDA graph replay
fixed-capacity graph route plan
rank-major shared + route FFN inputs
```

Topline:

```text
result: PASS
route audit: 43/43 clean
rank-major half-input totals: 43/43 clean
projected_slot_step_tok_s: 20.780357
checksum: 8339404968
```

The throughput number is diagnostic only because parity audit is enabled and
the routed FFN executor still uses fixed-capacity work.

## Interpretation

Rank-major FFN input layout is validated under graph replay:

- `shared_gate`: zero mismatches
- `shared_up`: zero mismatches
- active `route_a`: zero mismatches
- route metadata: zero missing selected experts, zero weight mismatches, zero
  invalid slots

The bottleneck has moved below input layout. The next useful work is the
actual-route routed FFN executor, not more parity work.

## Next Tasks

1. Add an opt-in actual-route routed FFN executor path.
2. Keep graph launch shape static but gate or compact inactive route rows on
   device.
3. Avoid invoking TurboMind/CUTLASS for inactive fixed-capacity route rows.
4. Validate 4-slot/256K graph replay correctness.
5. Validate 8-slot/256K graph replay performance against fixed-capacity.
6. Only then move back to 28/32-slot serving and MTP.
