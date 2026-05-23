# Sprint 206 - Six-Route FFN Persistent-Boundary Gate

Date: 2026-05-23
Status: Completed

## Objective

Pivot from the rejected TP4 decode branch back to the production six-route
routed-FFN path and measure the exact gate/up -> down-reduce sequence as one
boundary.

## Rationale

Sprint 205 rejected another TP4 decode collective variant. The production
serving stack is currently `fused6_reduce + graph`, so the next useful question
is whether the remaining routed-FFN gap is still launch/boundary overhead or
the actual MXFP4 GEMM work.

The six-route path already:

- consumes compact activation rows rather than expanded route rows;
- keeps MXFP4 weights packed;
- computes in the TurboMind/CUTLASS path rather than persistent F16 mirrors;
- elides `down_routes` through the down-reduce epilogue.

That leaves the gate/up and down GEMM bodies as the likely bottleneck. This
sprint adds a focused gate that times the full six-route routed FFN boundary
directly.

## Scope

1. Extend `test_ggml_turbomind_grouped_gate_up_fusion` with a full
   `gated_silu -> down_reduce` sequence timer for `total_routes=6`.
2. Add optional CUDA graph replay for that sequence through
   `DS4_TURBOMIND_GATE_UP_GRAPH_FFN`.
3. Build and run on the V100 pod against the current TurboMind library.
4. Record whether graph/persistent replay is still a material lever for the
   exact six-route FFN boundary.

## Non-Goals

- No production scheduler integration.
- No new monolithic gate/up/down kernel yet.
- No renewed TP4 decode integration.

## Definition Of Done

- [x] Sprint plan exists.
- [x] The benchmark builds on the V100 pod.
- [x] The six-route compact case passes correctness.
- [x] The full FFN sequence reports normal and graph replay timings.
- [x] Evidence is copied to
      `logs/from-cluster/sprint206-ffn-sequence/`.
- [x] Vision/status documents are updated.
- [x] Changes are committed.

## Implementation

`test_grouped_gate_up_fusion.cpp` now allocates a separate `d_ffn_reduce`
output and measures:

```text
gated MXFP4 gate/up:
  d_A -> d_gated

six-route MXFP4 down-reduce:
  d_gated -> d_ffn_reduce
```

For the same boundary it optionally captures:

```text
gated gate/up + output clear + down-reduce
```

into a CUDA graph and replays it on a non-blocking stream. This is not the final
kernel, but it tells us whether the existing boundary is still mostly launch
overhead.

## Validation

V100 build passed:

```text
cmake --build build/turbomind-v100 --target test_ggml_turbomind_grouped_gate_up_fusion -j80
```

Exported symbols were present:

```text
ggml_turbomind_ds4_mxfp4_down_6_m16_reduce
ggml_turbomind_ds4_mxfp4_gated_silu_6
```

Compact six-route run:

| Run | Iters | Gated | Down Reduce | Sum | Full Sequence | Graph Sequence | Graph Speedup | Correctness |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| smoke | 100 | `0.0947 ms` | `0.0651 ms` | `0.1598 ms` | `0.1597 ms` | `0.1573 ms` | `1.016x` | PASS |
| repeat | 500 | `0.0945 ms` | `0.0599 ms` | `0.1544 ms` | `0.1459 ms` | `0.1389 ms` | `1.050x` | PASS |

Evidence:

```text
logs/from-cluster/sprint206-ffn-sequence/
```

## Decision

The existing six-route boundary is not meaningfully launch-bound after graph
promotion. Graph replay of the exact gate/up -> down-reduce sequence improves
only `1.016x-1.050x` in the focused benchmark.

This supports the current working theory: the missing lever is not route
loading, output clear, host dispatch, or another graph wrapper. The remaining
gap is inside the MXFP4 GEMM bodies and the handoff between them. The next
sprint should implement or prototype a true monolithic/software-pipelined
routed-FFN kernel or a CUTLASS/TurboMind kernel variant that fuses the gated
activation tile into the down projection without round-tripping through the
current separate grouped GEMM boundary.
