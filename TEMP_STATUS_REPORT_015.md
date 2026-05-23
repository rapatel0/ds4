# TEMP Status Report 015 - Sprint 201 TP4 Boundary Proxy

Date: 2026-05-23

## Topline

Sprint 201 added and validated a bounded full-layer TP4 boundary proxy. The
result does not make TP4 a low-batch latency fix, but it keeps TP4/EP plausible
as a broad full-layer topology for higher active-token counts.

Current promoted served baseline remains Sprint 199:

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| `fused6_reduce + graph` | 256K | 16 | `67.886268` | `66.825545` | `16/16` |

## New Sprint 201 Data

Tool:

```text
tools/ds4-v100-tp4-layer-proxy
```

Build:

```text
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp4-layer-proxy
```

V100 results on devices `0,1,2,3`:

| Case | Boundary avg | Per layer | Per collective | Effective wire GB/s | Overhead-only tok/s | Verify |
|---|---:|---:|---:|---:|---:|---|
| doubling, 16 tokens, 43 layers, 4 collectives/layer | `24.414061 ms` | `0.567769 ms` | `0.141942 ms` | `14.775` | `655.360` | ok |
| root, 16 tokens, 43 layers, 4 collectives/layer | `22.113369 ms` | `0.514264 ms` | `0.128566 ms` | `12.234` | `723.544` | ok |
| doubling, 64 tokens, 43 layers, 4 collectives/layer | `34.830881 ms` | `0.810020 ms` | `0.202505 ms` | `41.424` | `1837.450` | ok |
| doubling, 128 tokens, 43 layers, 4 collectives/layer | `51.026125 ms` | `1.186654 ms` | `0.296664 ms` | `56.553` | `2508.519` | ok |
| doubling, 16 tokens, local-op repeats 16 | `23.904603 ms` | `0.555921 ms` | `0.138980 ms` | `15.090` | `669.327` | ok |

Evidence:

```text
logs/from-cluster/sprint201-tp4-layer-proxy/
```

## Interpretation

At 16 active tokens, TP4 boundary overhead alone is already `22-24 ms` for the
full 43-layer path. That is an overhead-only ceiling of roughly `655-724 tok/s`
before attention, expert GEMMs, dense/shared FFN, KV work, logits, and runtime
scheduling are counted.

That does not reject TP4, but it rejects partial TP. If we only tensor-parallel
the routed FFN and copy full hidden state in/out at layer boundaries, we pay too
much communication for too little execution-shape change.

The larger-token cases are more encouraging:

- `64` active tokens: `1837` overhead-only tok/s.
- `128` active tokens: `2509` overhead-only tok/s.

This points TP4 toward high-batch throughput or prefill, and possibly toward a
full-layer TP4/EP runtime slice where dense attention, shared dense paths, and
routed experts all stay native to the TP boundary.

## Decision

Do not expand routed-only TP overlays.

The next TP sprint should be a bounded full-layer TP4/EP runtime slice. If that
cannot keep dense and routed compute inside the TP boundary, return to the
other serious branch: a true persistent fused routed-FFN kernel with
CUTLASS/TurboMind-style software pipelining.
