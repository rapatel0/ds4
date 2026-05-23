# Sprint 266 - TP/EP Shared Dense Ops Probe

Date: 2026-05-23
Status: Complete

## Overview

Sprint 266 tests whether token-major setup cost can be reduced by hoisting the
per-layer dense operation objects. The token-major scaffold from Sprint 265
still builds dense cuBLAS handles, dense input buffers, and dense output
buffers for each layer invocation. This sprint adds an opt-in all-layer dense
op cache and tests it in token-major order.

This is a measurement sprint, not generated-token serving throughput.

## Implementation

`tools/ds4-v100-tp-ep-full-layer-smoke` now supports:

```text
--shared-dense-ops
```

The option pre-creates the attention-output and shared-expert dense ops for
all 43 layers using the existing shared dense FP16 cache. It keeps cuBLAS
handles and input/output buffers resident across token-major layer invocations.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint266-shared-dense-ops-token-major/cluster/local-dense-ops-4step.log`
- `logs/from-cluster/sprint266-shared-dense-ops-token-major/cluster/local-dense-ops-4step-summary.log`
- `logs/from-cluster/sprint266-shared-dense-ops-token-major/cluster/shared-dense-ops-4step.log`
- `logs/from-cluster/sprint266-shared-dense-ops-token-major/cluster/shared-dense-ops-4step-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--warmup 0 --iters 1 --decode-steps 4
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes
--local-tp-runtime --shared-expert-bindings
--overlap-ep-dense --source-copy-schedule
--token-major-all-layers --all-layers
```

A/B result:

| Metric | Local dense ops | Shared dense ops |
|---|---:|---:|
| Token steps | 4 | 4 |
| Layer invocations | 172 | 172 |
| Passing invocations | 172 | 172 |
| Shared dense ops | 0 | 1 |
| Sum decode ms | 207.967921 | 224.343371 |
| ms/token proxy | 51.991980 | 56.085843 |
| Projected slot-step tok/s | 615.479538 | 570.553966 |
| Sum EP/overlap ms | 112.772654 | 117.749695 |
| Sum compose ms | 95.139249 | 106.524800 |
| Wall ms | 36149.389385 | 35519.067908 |
| Checksum | 296236348 | 296236348 |
| Result | PASS | PASS |

## Decision

Reject shared dense ops as the default. It slightly reduces wall time but
regresses token-major decode proxy by `7.3%`. Keep it as an opt-in diagnostic.
The next token-major residency work should target TP runtime/KV metadata or
reduce the per-layer orchestration cost without perturbing dense/compose
timing.
