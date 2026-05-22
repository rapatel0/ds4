# Sprint 173 Follow-Ups

Date: 2026-05-22

## Persistent TP/EP Boundary

- **What**: Build a persistent tensor/expert-parallel routed-FFN boundary that
  reuses the Sprint 173 descriptor/output-mode shape instead of bolting TP onto
  one layer after the fact. The boundary should keep peer ownership and partial
  output native to the execution topology, with overlap across a meaningful
  layer group.
- **Why**: `fused6` proved that removing route-expanded `a_half` alone is not a
  throughput lever. Prior TP primitives showed positive math/copy speedups, but
  the one-layer overlay regressed because copy/sync work was attached outside
  the persistent boundary.
- **Severity**: Critical.
- **Suggested sprint**: Sprint 174.
- **Files**: `ds4_cuda.cu`, `ds4_v100_layer_execute.*`,
  `ds4_v100_scheduler.*`, `tests/cuda_v100_tp_routed_ffn_smoke.c`.

## Larger Routed-FFN Fusion

- **What**: If TP/EP is not pursued next, implement a larger routed-FFN boundary
  that also removes or fuses `mid_half` and `down_routes`, not just `a_half`.
- **Why**: Sprint 173 liveness logs show `mid_half` and `down_routes` are still
  materialized. The served regression indicates indexed activation alone adds
  overhead without removing enough traffic or launch work.
- **Severity**: Important.
- **Suggested sprint**: Sprint 174 or later, behind the TP/EP decision.
- **Files**: `ds4_cuda.cu`,
  `kernels/turbomind/ggml-turbomind/ggml-turbomind-ds4-probe.cu`,
  `kernels/turbomind/ggml-turbomind/api.cc`,
  `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h`.

## In-Kernel F32-to-F16 Activation Tile

- **What**: Add a TurboMind ABI/probe that reads F32 activation rows and casts to
  F16 inside the A-tile load, eliminating even the compact per-token `a_half`
  staging.
- **Why**: This was the Sprint 173 stretch slice. It should not be attempted as
  another isolated wrapper experiment unless it is part of a larger fused or
  TP/EP boundary, because the cheaper un-expanded staging variant regressed.
- **Severity**: Important.
- **Suggested sprint**: Future, only as part of larger boundary work.
- **Files**: `ds4_cuda.cu`,
  `kernels/turbomind/ggml-turbomind/ggml-turbomind-ds4-probe.cu`,
  `kernels/turbomind/ggml-turbomind/api.cc`.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Persistent TP/EP boundary | Critical | Sprint 174 | `ds4_cuda.cu`, `ds4_v100_layer_execute.*`, `ds4_v100_scheduler.*` |
| Larger routed-FFN fusion | Important | Sprint 174+ | `ds4_cuda.cu`, TurboMind probe/API |
| In-kernel F32-to-F16 activation tile | Important | Future | `ds4_cuda.cu`, TurboMind probe/API |
