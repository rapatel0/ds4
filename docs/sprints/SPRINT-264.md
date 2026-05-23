# Sprint 264 - TP/EP Source-Scheduled Staged Copies

Date: 2026-05-23
Status: Complete

## Overview

Sprint 264 optimizes the staged compose/all-to-all path. Sprint 263 showed
that direct remote reads are slower than explicit staging, so this sprint keeps
staging but changes copy scheduling. The prior path enqueued peer copies by
destination stream. The new source-scheduled path gives each source rank a copy
stream and enqueues that source GPU's outbound shards on its own stream.

This remains a scaffold gate, not generated-token serving throughput.

## Implementation

`RankState` now owns a `copy_stream`. The tool supports:

```text
--source-copy-schedule
--dest-copy-schedule
```

Source-copy scheduling is now the default. Destination-copy scheduling remains
available for diagnostics.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint264-source-copy-schedule-ab/cluster/dest-copy-50step.log`
- `logs/from-cluster/sprint264-source-copy-schedule-ab/cluster/dest-copy-50step-summary.log`
- `logs/from-cluster/sprint264-source-copy-schedule-ab/cluster/source-copy-50step.log`
- `logs/from-cluster/sprint264-source-copy-schedule-ab/cluster/source-copy-50step-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--warmup 2 --iters 5 --decode-steps 50
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes
--local-tp-runtime --shared-expert-bindings --overlap-ep-dense --all-layers
```

A/B result:

| Metric | Destination-scheduled copies | Source-scheduled copies |
|---|---:|---:|
| Passing layers | 43 / 43 | 43 / 43 |
| Source copy schedule | 0 | 1 |
| Sum decode ms/token | 38.072821 | 32.016315 |
| Projected slot-step tok/s | 840.494594 | 999.490407 |
| Sum EP/overlap ms | 12.612690 | 12.494587 |
| Sum dense ms | 0.000000 | 0.000000 |
| Sum compose ms | 25.452322 | 19.513090 |
| Wall ms | 14060.874811 | 13661.510783 |
| Checksum | 204721433 | 204721433 |
| Result | PASS | PASS |

## Decision

Promote source-scheduled staged copies as the default. This preserves the
staged-copy approach that beat direct remote reads, but improves compose time
by `23.3%` and projected scaffold throughput by `18.9%`. The next step should
either convert this scaffold into a serving loop or continue optimizing the
destination-side compose kernel.
