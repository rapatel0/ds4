# TEMP Status Report 011

Date: 2026-05-23

## Sprint 197 Result

The runtime profile now reports routed-FFN materialization liveness. This was
validated on the V100 pod with the persistent Sprint 181 appliance pack.

Representative profile for GPU0:

| Field | Value |
|---|---:|
| calls | `48` |
| route_expanded_a_calls | `0` |
| compact_a_calls | `48` |
| mid_half_calls | `48` |
| down_routes_calls | `0` |
| down_reduce_epilogue_calls | `48` |
| avg_scratch_bytes | `36384` |
| avg_mid_half_bytes | `24576` |
| gate_up_pct | `46.65%` |
| down_pct | `24.44%` |

## Interpretation

The current `fused6_reduce` path already does two important things correctly:

- it avoids route-expanded activation staging;
- it elides `down_routes` with the down-reduce epilogue.

The remaining materialized boundary is `mid_half`, and it appears on every
routed FFN call. For the six-route decode shape, however, this is only `24 KiB`
per call. That means a pure scratch-memory reduction is unlikely to move the
production 16-slot/256K serving number by itself.

## Next Direction

The next production-serving sprint should target execution-boundary cost, not
just buffer liveness:

- persistent/tile-level gate/up plus down executor;
- or a larger fused routed-FFN kernel that consumes the gated activation tile
  before writing full `mid_half`;
- or TP4 only for larger batched/prefill shapes where Sprint 196 showed the
  doubling collective wins.

This narrows the high-throughput path: the next routed-FFN work has to change
GEMM/launch structure, not just remove a small intermediate allocation.
