# Sprint 201 - Bounded TP4 Layer-Boundary Prototype

Date: 2026-05-23
Status: Completed

## Objective

Pivot from routed-only kernel wrappers to a bounded full-layer tensor-parallel
measurement that answers whether TP4/EP is worth turning into production
runtime code.

The sprint target is not another routed-FFN overlay. It is an executable
four-GPU layer-boundary proxy for the communication shape a real TP4 full-layer
runtime would pay:

```text
participants = 4 V100s
hidden = 4096
active tokens = 16 by default
layers = 43 by default
collectives/layer = 4 by default
payload dtype = F32 hidden-state boundary for conservative measurement
```

## Rationale

Sprint 200 stopped the six-route persistent-kernel branch. The exact production
six-route fixed gate/up probe was slower than the generic TurboMind path
(`0.1196 ms` versus about `0.0946 ms`), and the output clear was only
`0.0022 ms`. That rejects a small clear-fusion ABI as a material serving lever.

The latest evidence says the next useful question is broader:

- routed-only TP2 exists and is correct, but the overlay preserves the wrong
  per-layer boundary and regressed served throughput;
- TP4 collectives are correct, but prior tests measured only isolated hidden
  all-reduces;
- full-layer TP4/EP could still be useful because it changes the whole layer
  execution shape, not just the routed FFN;
- the practical-serving goal needs hundreds of tok/s, so a TP4 path must be
  rejected quickly if its boundary overhead alone is too large.

## Scope

1. Add a CUDA tool that runs a TP4 layer-boundary proxy across four V100s.
2. Support `root` and `doubling` collective algorithms.
3. Support configurable `tokens`, `layers`, `collectives-per-layer`, and
   resident local-op repeats.
4. Report:
   - full boundary latency;
   - per-layer latency;
   - per-collective latency;
   - effective wire GB/s;
   - overhead-only token/s for the active token count.
5. Validate cross-device output equality after the full boundary sequence.
6. Build and run the tool on the V100 pod for the 16-token/43-layer default and
   at least one larger active-token shape.

## Non-Goals

- No production TP4 scheduler integration in this sprint.
- No NCCL dependency.
- No model-quality or served HTTP benchmark from this proxy alone.
- No claim that the proxy includes DS4 GEMM compute; it measures the boundary
  cost that a full-layer TP4 implementation must amortize.

## Implementation Map

| File | Work |
|---|---|
| `tools/ds4-v100-tp4-layer-proxy.cu` | New CUDA executable for repeated TP4 hidden collectives with resident GPU work between boundaries. |
| `Makefile` | Add CUDA build and clean targets for the new tool. |
| `logs/from-cluster/sprint201-tp4-layer-proxy/` | Store V100 build/run evidence. |
| `TEMP_STATUS_REPORT_015.md` | Record the current status and decision from Sprint 201. |
| `docs/sprints/VISION.md` | Update practical-serving direction based on measured TP4 boundary cost. |
| `docs/sprints/STATUS.md` and `docs/sprints/EXPERIMENT-STATUS.md` | Add the sprint result and topline numbers. |

## Definition Of Done

- [x] Sprint plan exists and is aligned with the Sprint 200 stop condition.
- [x] `tools/ds4-v100-tp4-layer-proxy.cu` exists.
- [x] `Makefile` can build `tools/ds4-v100-tp4-layer-proxy` on CUDA hosts.
- [x] V100 build passes with `CUDA_ARCH=sm_70`.
- [x] V100 run passes cross-device verification for the 16-token default.
- [x] V100 run records at least one larger active-token shape.
- [x] Evidence is copied into `logs/from-cluster/sprint201-tp4-layer-proxy/`.
- [x] Status, vision, experiment, and TEMP report documents are updated.
- [x] Changes are committed.

## Decision Gate

If the boundary-only 16-token/43-layer TP4 proxy is already too large to leave
room for DS4 compute, do not start production TP4 integration yet. Either:

- narrow TP4 to a high-batch/prefill-only path, or
- return to a true persistent fused routed-FFN kernel with CUTLASS-style
  software pipelining.

If the boundary overhead is small enough, the next sprint should implement a
bounded full-layer TP4 runtime prototype over a small layer span.

## Implementation

Added `tools/ds4-v100-tp4-layer-proxy.cu`, a four-GPU CUDA executable that
runs repeated hidden-state collectives across a full DS4 layer count. It keeps
two resident buffers per participant, alternates all-reduce inputs and outputs,
normalizes after each collective, optionally performs local resident arithmetic,
and verifies final cross-device equality.

The Makefile now builds the tool on CUDA hosts as:

```text
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp4-layer-proxy
```

## Validation

V100 build passed on `llm/llamacpp-build-8gpu`.

Measured on devices `0,1,2,3`:

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

## Decision

The TP4 communication boundary is not automatically disqualifying, but it is
too expensive to justify another routed-only overlay.

At the `16` active-token / `256K` target shape, the full 43-layer TP4 boundary
alone costs about `22-24 ms`. That implies an overhead-only ceiling of roughly
`655-724 tok/s` before any attention, routed expert, shared FFN, output, KV, or
runtime scheduling work. This can still support a several-hundred tok/s target
if full-layer TP materially improves compute utilization, but it cannot carry a
small partial overlay.

At larger active-token shapes, the envelope improves sharply: `64` active tokens
measure `1837 tok/s` overhead-only, and `128` active tokens measure
`2509 tok/s`. TP4 is therefore more plausible as a full-layer high-batch or
prefill/throughput topology than as a low-batch decode latency fix.

Next sprint should implement one bounded full-layer TP4/EP runtime slice only
if it keeps dense and routed work inside the TP boundary. Otherwise, return to
a true persistent fused routed-FFN kernel with CUTLASS/TurboMind-style
software pipelining.
