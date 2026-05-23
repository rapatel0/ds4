# Sprint 269 - TP/EP Continuous Token-Major Gate

Date: 2026-05-23
Status: Complete

## Overview

Sprint 269 runs longer token-major scaffold gates after the Sprint 267 shared
TP runtime promotion and Sprint 268 position advance. The purpose is to reduce
early-token noise and get a steadier view of the TP/EP path at the practical
target shape.

This is still scaffold throughput, not generated-token serving throughput.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint269-token-major-continuous/cluster/continuous-16step.log`
- `logs/from-cluster/sprint269-token-major-continuous/cluster/continuous-16step-summary.log`
- `logs/from-cluster/sprint269-token-major-continuous/cluster/continuous-32step.log`
- `logs/from-cluster/sprint269-token-major-continuous/cluster/continuous-32step-summary.log`

Command shape:

```text
--slots 32 --top-k 6
--warmup 0 --iters 1
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes
--shared-expert-bindings --overlap-ep-dense --source-copy-schedule
--token-major-all-layers --all-layers
```

The default token-major shared TP runtime and position advance are active.

Results:

| Metric | 16-step run | 32-step run |
|---|---:|---:|
| Start position | 2048 | 4096 |
| Token steps | 16 | 32 |
| Layer invocations | 688 | 1376 |
| Passing invocations | 688 | 1376 |
| Shared TP runtime | 1 | 1 |
| Sum decode ms | 642.201062 | 1257.287012 |
| ms/token proxy | 40.137566 | 39.290219 |
| Projected slot-step tok/s | 797.258102 | 814.452062 |
| Sum EP/overlap ms | 271.960352 | 514.766496 |
| Sum compose ms | 370.040118 | 742.079181 |
| Wall ms | 45723.864708 | 91515.672970 |
| Checksum | 8244145680 | 8297177632 |
| Result | PASS | PASS |

## Decision

The continuous token-major scaffold is stable and improves as startup effects
amortize: `814.452062` projected slot-step tok/s at `32` token steps. The
dominant measured stage is now compose/all-to-all, not routed expert compute:
the 32-step run spends `742.079181 ms` in compose versus `514.766496 ms` in
the overlapped EP stage.

Next work should either collapse the compose/all-to-all boundary or bridge the
token-major scaffold into generated/continuation serving so we can measure
actual tok/s while preserving this TP/EP schedule.
