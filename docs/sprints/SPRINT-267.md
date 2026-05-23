# Sprint 267 - TP/EP Token-Major Shared TP Runtime

Date: 2026-05-23
Status: Complete

## Overview

Sprint 267 rechecks shared TP runtime in token-major serving order. Earlier
layer-major runs rejected shared TP runtime because summed decode timing
regressed. Token-major order has a different cost profile: the runtime and KV
state are reused across token steps, so TP/KV residency can reduce setup and
may improve the serving-order proxy.

This is still scaffold throughput, not generated-token serving throughput.

## Implementation

`tools/ds4-v100-tp-ep-full-layer-smoke` now defaults token-major all-layer
runs to shared TP runtime unless the caller explicitly passes:

```text
--local-tp-runtime
```

Layer-major all-layer runs keep the old default. `--share-tp-runtime` remains
available as an explicit override.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint267-token-major-shared-tp-runtime/cluster/local-tp-runtime-4step.log`
- `logs/from-cluster/sprint267-token-major-shared-tp-runtime/cluster/local-tp-runtime-4step-summary.log`
- `logs/from-cluster/sprint267-token-major-shared-tp-runtime/cluster/shared-tp-runtime-4step.log`
- `logs/from-cluster/sprint267-token-major-shared-tp-runtime/cluster/shared-tp-runtime-4step-summary.log`
- `logs/from-cluster/sprint267-token-major-shared-tp-runtime/cluster/default-token-major-1step.log`
- `logs/from-cluster/sprint267-token-major-shared-tp-runtime/cluster/default-token-major-1step-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--warmup 0 --iters 1 --decode-steps 4
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes
--shared-expert-bindings --overlap-ep-dense --source-copy-schedule
--token-major-all-layers --all-layers
```

A/B result:

| Metric | Local TP runtime | Shared TP runtime |
|---|---:|---:|
| Token steps | 4 | 4 |
| Layer invocations | 172 | 172 |
| Passing invocations | 172 | 172 |
| Shared TP runtime | 0 | 1 |
| Sum decode ms | 205.158196 | 191.609295 |
| ms/token proxy | 51.289549 | 47.902324 |
| Projected slot-step tok/s | 623.908781 | 668.026047 |
| Sum EP/overlap ms | 112.738584 | 96.796352 |
| Sum compose ms | 92.368003 | 94.759843 |
| Wall ms | 34880.753622 | 11661.323548 |
| Checksum | 296236348 | 296236348 |
| Result | PASS | PASS |

Default wiring check:

| Metric | Value |
|---|---:|
| Token steps | 1 |
| Layer invocations | 43 |
| Shared TP runtime | 1 |
| Sum decode ms | 83.942342 |
| Projected slot-step tok/s | 381.214048 |
| Wall ms | 3165.959421 |
| Checksum | 268612144 |
| Result | PASS |

## Decision

Promote shared TP runtime as the token-major all-layer default. It improves
the serving-order proxy by `7.1%` and cuts wall/setup time by about `66.6%`
with checksum preserved. Keep `--local-tp-runtime` for A/B diagnostics and
keep layer-major defaults unchanged.

Next work should reduce token-major compose/all-to-all and orchestration cost,
then bridge this scaffold into generated/continuation serving measurement.
