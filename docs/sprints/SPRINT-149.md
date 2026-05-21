# Sprint 149 - TP Split And P2P Topology Probe

Date: 2026-05-21

## Objective

Quantify whether tensor-parallel routed-FFN splitting is worth a production
scheduler rewrite after the fused-kernel stage-count probe failed to move
served throughput materially.

## Changes

- Extended the TurboMind gate/up fusion benchmark with
  `DS4_TURBOMIND_GATE_UP_TP_SPLIT=1`.
- The new mode slices the DS4 FFN middle dimension from `2048` into two
  `1024` halves using the same generated MXFP4 fixtures.
- It measures:
  - full-rank generic gated gate/up plus down time;
  - half-0 and half-1 gate/up plus down time;
  - sequential two-half time on one GPU;
  - ideal 2-way TP compute speedup before communication;
  - F16/F32 hidden reduce payload size.
- Added `test_ggml_turbomind_p2p_reduce_proxy`, a CUDA-runtime peer-copy
  payload benchmark for the same hidden-vector sizes.
- Ran a targeted NCU check on `m128` versus `m128_s4` gate/up to verify the
  stage-4 result.

## Stage-4 NCU Check

One profiled 768-route gate/up launch showed no material hardware-counter
change:

| Mode | Kernel time | SM throughput | DRAM throughput | HMMA instructions |
|---|---:|---:|---:|---:|
| `m128` | `779.14 us` | `35.38%` | `13.06%` | `50,331,648` |
| `m128_s4` | `780.70 us` | `35.45%` | `13.00%` | `50,331,648` |

This confirms that the stage-4 variant should remain explicit opt-in.

## TP Compute Proxy

Compact routed benchmark, 6 active groups:

| Shape | Full routed FFN | Half 0 | Half 1 | Ideal 2-way compute | Ideal speedup | Sequential split |
|---|---:|---:|---:|---:|---:|---:|
| 768 routes | `0.9821 ms` | `0.5284 ms` | `0.5284 ms` | `0.5284 ms` | `1.858x` | `0.930x` |
| 1536 routes | `1.3134 ms` | `0.8918 ms` | `0.8945 ms` | `0.8945 ms` | `1.468x` | `0.735x` |

The 2-way split is promising as a 1.5-2x-class compute lever, but not as a
direct path from `~61 tok/s` to `1k+ tok/s`.

## P2P Payload Proxy

Measured with `routes * hidden` payloads where `hidden=4096`:

| Payload | NV2 copy | NV1 copy | SYS copy |
|---|---:|---:|---:|
| 6 MiB | `0.1317 ms` | `0.2608 ms` | `0.6386 ms` |
| 12 MiB | `0.2615-0.2618 ms` | `0.5203-0.5207 ms` | `1.2904-1.3138 ms` |
| 24 MiB | `0.5210 ms` | `1.0388-1.0391 ms` | `2.5683 ms` |

Placement matters. A 2-way TP prototype should pair GPUs over NV2 links where
possible, for example `0-3`, `0-4`, or `4-7` from the observed topology.

## Decision

TP/EP is now justified as a prototype, not as an assumed redesign. The next
production-relevant experiment should be a bounded 2-GPU routed-FFN TP path
for one stage or a scheduler-level microbenchmark that overlaps the half-FFN
compute with NV2 payload exchange. Avoid 8-way EP first: the current compact
served shape has only 6 active expert groups, so 8-way expert parallelism is
likely underfilled before it helps.

## Artifacts

- `logs/from-cluster/sprint149-ncu-gate-up-metrics/`
- `logs/from-cluster/sprint149-tp-split-probe/`
- `logs/from-cluster/sprint149-p2p-reduce-proxy/`
