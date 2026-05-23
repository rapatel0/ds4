# TEMP Status Report 020

Date: 2026-05-23

## Current Topline

Production best remains the Sprint 199 promoted stack:

```text
16-slot / 256K served:
  fused6_reduce + graph
  generated tok/s:     67.886268
  continuation tok/s:  66.825545
  correctness:         16/16
```

Sprint 206 did not change production defaults. It added a focused V100
benchmark for the exact six-route routed-FFN sequence that production is using.

## What Sprint 206 Tested

The new benchmark path measures:

```text
MXFP4 gated gate/up -> MXFP4 down-reduce
```

for the compact six-route shape:

```text
total_routes=6
active_experts=6
max_routes_per_expert=1
hidden=4096
mid=2048
```

It also captures that same sequence into a CUDA graph and replays it on a
non-blocking stream.

## Results

| Run | Iters | Gated | Down Reduce | Full Sequence | Graph Sequence | Graph Speedup |
|---|---:|---:|---:|---:|---:|---:|
| smoke | 100 | `0.0947 ms` | `0.0651 ms` | `0.1597 ms` | `0.1573 ms` | `1.016x` |
| repeat | 500 | `0.0945 ms` | `0.0599 ms` | `0.1459 ms` | `0.1389 ms` | `1.050x` |

Both runs passed correctness.

## Interpretation

This makes a loading/dispatch-only explanation less likely for the current
six-route routed-FFN bottleneck. The exact boundary already graphs cleanly, and
graph replay only gives a small local improvement.

The remaining work is therefore inside the MXFP4 compute path:

- gate/up kernel body;
- down kernel body;
- the global-memory `d_gated`/`mid_half` handoff between them;
- tile scheduling and software pipelining inside a real fused kernel.

## Tensor Parallel Status

TP4 is paused for production decode:

- compute-only routed FFN split was attractive (`2.35x-3.64x` in Sprint 202);
- resident decode boundaries were not good enough at the production 96-route
  shape (`0.860x-0.896x` in Sprints 204-205);
- TP4 remains plausible for larger batches/prefill or a broader full-layer
  topology, but not as the next decode integration.

## Next Practical Step

Implement a true monolithic/software-pipelined routed-FFN kernel prototype that
keeps the gated activation tile local to the kernel boundary and feeds the down
projection without relying on the current separate grouped GEMM handoff.
