# Sprint 262 - TP/EP FP16 EP Return Recheck

Date: 2026-05-23
Status: Complete

## Overview

Sprint 262 rechecks FP16 EP return after Sprint 261 changed the execution
shape. Earlier FP16-return experiments were rejected before resident expert
bindings and EP+dense overlap. Since compose/all-to-all is now dominant, it was
worth retesting whether halving the EP return payload helps.

This is a measurement sprint, not generated-token serving throughput.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint262-ep-return-fp16-overlap-ab/cluster/fp32-return-50step.log`
- `logs/from-cluster/sprint262-ep-return-fp16-overlap-ab/cluster/fp32-return-50step-summary.log`
- `logs/from-cluster/sprint262-ep-return-fp16-overlap-ab/cluster/fp16-return-50step.log`
- `logs/from-cluster/sprint262-ep-return-fp16-overlap-ab/cluster/fp16-return-50step-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--warmup 2 --iters 5 --decode-steps 50
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes
--local-tp-runtime --shared-expert-bindings --overlap-ep-dense --all-layers
```

A/B result:

| Metric | FP32 EP return | FP16 EP return |
|---|---:|---:|
| Passing layers | 43 / 43 | 43 / 43 |
| Sum decode ms/token | 38.470986 | 43.875315 |
| Projected slot-step tok/s | 831.795688 | 729.339500 |
| Sum EP/overlap ms | 12.853440 | 12.667618 |
| Sum dense ms | 0.000000 | 0.000000 |
| Sum compose ms | 25.608539 | 31.200853 |
| Wall ms | 13852.149469 | 14922.370777 |
| Checksum | 204721433 | 204721433 |
| Result | PASS | PASS |

## Decision

Keep FP32 EP return as the default. FP16 return reduces the EP/overlap bucket
slightly but makes compose substantially slower, so total projected throughput
regresses by `12.3%`. The next compose/all-to-all work should avoid standalone
cast-and-expand staging; it needs a fused or direct destination accumulation
approach.
