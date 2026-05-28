# TEMP Status Report 014

Date: 2026-05-23

## Sprint 200 Result

Sprint 200 did not add a runtime executor. It added the missing exact-shape
TurboMind bench and triggered the stop condition for the easy six-route kernel
cut-ins.

Focused V100 bench, compact six-route production shape:

| Metric | Value |
|---|---:|
| generic gated-SiLU | `0.0946 ms` |
| fixed `m16_6` gated-SiLU probe | `0.1196 ms` |
| generic down projection | `0.0512 ms` |
| F32 output clear only | `0.0022 ms` |
| six-route down-reduce with clear | `0.0650 ms` |
| down-reduce relative error | `2.0515e-04` |
| down-reduce bad values | `0/4096` |

## Interpretation

The promoted Sprint 199 path is not accidentally using the slow fixed `m16_6`
gate/up probe; `fused6_reduce` falls through to the generic TurboMind
gated-SiLU path and then uses the six-route down-reduce epilogue.

The output clear is too small to chase by itself. A wrapper ABI that only hides
the clear and existing down-reduce call would not be a material kernel
improvement, and a race-safe in-kernel clear would require changing the atomic
route-reduce epilogue into a true per-column route reduction.

## Next Direction

Pivot to bounded full-layer TP4/EP or a real non-atomic route-reduce epilogue
rewrite. The full-layer TP4/EP path is more aligned with the throughput vision
because it can change the dense execution shape, not just trim a six-route
micro-boundary.
