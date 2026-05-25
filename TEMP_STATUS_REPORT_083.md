# TEMP Status Report 083

Date: 2026-05-25

## Current Focus

TP/EP active-slot characterization at the real 32-slot / 256K serving shape.

## Sprint 371 Result

Full matrix completed on the V100 pod:

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

| Active requests | HTTP 200 | Coalesced batch | Client tok/s | Server wall tok/s | Server decode tok/s | Avg GPU util | Max GPU util |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 1 | 1.584552 | 82.049572 | 98.230713 | 10.264286% | 40% |
| 4 | 4/4 | 4 | 6.430512 | 83.852456 | 99.991505 | 10.200000% | 40% |
| 8 | 8/8 | 8 | 12.450978 | 81.926628 | 97.865480 | 9.958333% | 40% |
| 16 | 16/16 | 16 | 24.557272 | 81.630760 | 97.446076 | 9.840278% | 39% |
| 32 | 32/32 | 32 | 50.694229 | 82.530166 | 98.768134 | 10.317857% | 41% |

Artifacts:

- Cluster: `/workspace/logs/sprint371-active-slot-matrix-full`
- Local: `logs/from-cluster/sprint371-active-slot-matrix-full`

## Interpretation

Coalescing is working. Client aggregate tok/s scales almost linearly with active
request count because the same fixed batch cost returns more tokens as more
slots are active.

The server-side decode rate and GPU utilization are flat. That means the
current runtime pays nearly the same 32-slot/layer-step cost even when only
one slot is active. Active-slot compaction would improve low-occupancy
efficiency and latency, but it will not materially improve the full 32-slot
aggregate topline.

## Next Best Work

For maximum 32-slot aggregate throughput, target the full-occupancy bottleneck:

- compressed/indexer dense projection
- attention projection/state
- compressed KV state/emit boundaries
- GPU0-heavy staging and rank imbalance

Active-slot compaction remains useful, but it is now a practical-serving
efficiency item rather than the main path to higher 32-slot tok/s.
