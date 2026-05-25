# Sprint 371: TP/EP Full Active-Slot Characterization

## Overview

Run the full active-slot matrix using the Sprint 370 driver at the target
serving shape:

- `32` configured slots
- `256K` context
- valid long-context chat start position
- `32` generated tokens/request
- active request cases `1,4,8,16,32`

## Goal

Determine whether current TP/EP serving throughput and GPU utilization scale
with active request count before implementing active-slot compaction or deeper
kernel/state fusion.

## Definition Of Done

- Full matrix completes on the V100 pod.
- Aggregate matrix TSV/JSON are copied into `logs/from-cluster`.
- Results record HTTP success count, coalesced batch size, server/client tok/s,
  compressed-KV timing, and GPU utilization.
- Docs/status are updated with the interpretation.
- Artifacts are committed.

## Decision Rule

- If server decode throughput and utilization scale materially from `1` to
  `32` active requests, prioritize kernel/fusion work inside the existing
  32-slot execution path.
- If they remain mostly flat, prioritize active-slot compaction or scheduler
  changes that avoid paying full fixed-slot cost at low/moderate occupancy.

## Outcome

The full V100 matrix completed.

Command shape:

```text
--requests-cases 1,4,8,16,32
--tokens 32
--ctx 262144
--slots 32
--position 262080
--gpu-sample-interval-ms 500
--hc-current-stream-sync
--tool none
```

Results:

| Active requests | HTTP 200 | Coalesced batch | Client tok/s | Server wall tok/s | Server decode tok/s | Avg GPU util | Max GPU util |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 1 | 1.584552 | 82.049572 | 98.230713 | 10.264286% | 40% |
| 4 | 4/4 | 4 | 6.430512 | 83.852456 | 99.991505 | 10.200000% | 40% |
| 8 | 8/8 | 8 | 12.450978 | 81.926628 | 97.865480 | 9.958333% | 40% |
| 16 | 16/16 | 16 | 24.557272 | 81.630760 | 97.446076 | 9.840278% | 39% |
| 32 | 32/32 | 32 | 50.694229 | 82.530166 | 98.768134 | 10.317857% | 41% |

Artifacts:

- Cluster:
  `/workspace/logs/sprint371-active-slot-matrix-full`
- Local:
  `logs/from-cluster/sprint371-active-slot-matrix-full`

Interpretation:

Coalescing works and client aggregate tok/s scales with active request count
because the fixed TP/EP batch cost is amortized over more returned tokens.
However, server-side wall/decode tok/s, compressed-KV time, and GPU
utilization are effectively flat from `1` to `32` active requests. The current
runtime is paying nearly the same 32-slot/layer-step cost even at low active
occupancy, and the full 32-slot topline is still limited by kernel/state
fragmentation and rank imbalance rather than insufficient active requests.

Decision:

- Active-slot compaction is useful for practical low/moderate occupancy cost
  and latency.
- It will not materially improve the full 32-slot aggregate topline.
- The next performance sprint should target the full-occupancy bottleneck:
  compressed/indexer dense projection, attention projection/state, and
  remaining GPU0-heavy staging/imbalance.
