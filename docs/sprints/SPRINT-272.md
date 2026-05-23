# Sprint 272 - TP/EP Multi Copy Streams Probe

Date: 2026-05-23
Status: Complete

## Overview

Sprint 272 tests whether source-scheduled staged compose copies benefit from
multiple copy streams per source rank. Sprint 271 showed that peer-copy time
is the largest part of compose, so this is the last targeted compose-scheduler
probe before pivoting to TP/EP end-to-end serving.

## Implementation

`tools/ds4-v100-tp-ep-full-layer-smoke` now supports:

```text
--multi-copy-streams
```

The option creates per-destination copy streams for each rank and schedules
source-side peer copies on those streams. It remains opt-in until more serving
evidence exists.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint272-multi-copy-streams/cluster/single-copy-stream-16step.log`
- `logs/from-cluster/sprint272-multi-copy-streams/cluster/single-copy-stream-16step-summary.log`
- `logs/from-cluster/sprint272-multi-copy-streams/cluster/multi-copy-streams-16step.log`
- `logs/from-cluster/sprint272-multi-copy-streams/cluster/multi-copy-streams-16step-summary.log`
- `logs/from-cluster/sprint272-multi-copy-streams/cluster/multi-copy-streams-32step.log`
- `logs/from-cluster/sprint272-multi-copy-streams/cluster/multi-copy-streams-32step-summary.log`

Results:

| Metric | Single copy stream, 16 steps | Multi copy streams, 16 steps | Multi copy streams, 32 steps |
|---|---:|---:|---:|
| Passing invocations | 688 | 688 | 1376 |
| ms/token proxy | 39.288036 | 37.395624 | 36.911097 |
| Projected slot-step tok/s | 814.497321 | 855.715092 | 866.947964 |
| Sum EP/overlap ms | 287.470527 | 289.740083 | 536.041653 |
| Sum compose ms | 340.885269 | 308.370566 | 644.609467 |
| Compose reduce ms | 54.451149 | 52.888732 | 114.803546 |
| Compose copy ms | 248.331836 | 219.221398 | 452.180787 |
| Compose final ms | 38.102284 | 36.260436 | 77.625134 |
| Checksum | 8244145680 | 8244145680 | 8297177632 |
| Result | PASS | PASS | PASS |

## Decision

Multi-copy streams are promising: the 16-step A/B reduces copy time by `11.7%`
and improves projected scaffold throughput from `814.497321` to `855.715092`
slot-step tok/s. The 32-step opt-in topline is `36.911097 ms/token` proxy and
`866.947964` projected slot-step tok/s.

Do not continue compose micro-optimization immediately. The next sprint should
make the TP/EP system operational end-to-end enough to report generated and
continuation tok/s, then revisit kernel selection/fusion with serving data.
