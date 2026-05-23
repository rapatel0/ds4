# Sprint 265 - TP/EP Token-Major Scaffold

Date: 2026-05-23
Status: Complete

## Overview

Sprint 265 adds the first serving-order TP/EP scaffold. Previous all-layer
gates were layer-major: each layer ran many decode iterations before moving to
the next layer. That is useful for kernel timing, but a server decodes in
token-major order: for each generated token, run layers `0..42` once.

This remains a scaffold gate, not generated-token serving throughput. It does
not yet carry real hidden state or logits, but it exposes the scheduling order
and setup costs we must solve before production serving.

## Implementation

`tools/ds4-v100-tp-ep-full-layer-smoke` now supports:

```text
--token-major-all-layers
```

In this mode:

```text
for step in decode_steps:
  for layer in 0..42:
    run one TP/EP decode step for that layer
```

The path keeps the current resident defaults:

- shared dense FP16 cache
- shared TurboMind API
- shared rank buffers
- resident active MXFP4 expert bindings
- local per-layer TP runtime
- EP+dense overlap
- staged compose with source-scheduled peer copies

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint265-token-major-scaffold/cluster/token-major-4step.log`
- `logs/from-cluster/sprint265-token-major-scaffold/cluster/token-major-4step-summary.log`

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

Result:

| Metric | Value |
|---|---:|
| Token steps | 4 |
| Layer invocations | 172 |
| Passing invocations | 172 |
| Slots | 32 |
| Context | 256K |
| Sum decode ms | 195.360044 |
| ms/token proxy | 48.840011 |
| Projected slot-step tok/s | 655.200508 |
| Sum EP/overlap ms | 102.840750 |
| Sum dense ms | 0.000000 |
| Sum compose ms | 92.438684 |
| Wall ms | 35297.952033 |
| Checksum | 296236348 |
| Result | PASS |

## Decision

Keep token-major mode as the serving-order scaffold gate. It is slower than
the layer-major kernel proxy, as expected, but it is closer to the runtime that
must eventually produce generated and continuation tok/s. The next sprint
should reduce token-major wall/setup cost by making per-layer TP runtime/KV
state or per-layer row metadata resident in this mode.
