# Sprint 258 - TP/EP Shared Runtime Repeat Gate

Date: 2026-05-23
Status: Complete

## Overview

Sprint 258 repeats the shared TP runtime path with a longer decode loop. Sprint
257 proved correctness and reduced wall time, but its 10-step decode proxy
regressed versus Sprint 256. This sprint runs the same resident scaffold for
50 decode steps per layer to reduce short-run noise.

This is a measurement sprint, not generated-token serving throughput.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Reuse the TP/EP shared dense cache, shared TurboMind API, shared rank
  buffers, and shared TP runtime path.
- Run a longer all-layer decode-only gate on the V100 pod at `32` slots /
  `256K`.
- Decide whether Sprint 257's decode regression is likely noise.

## Non-Goals

- No code changes.
- No PP scheduler changes.
- No production server integration.
- No MTP.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint258-tp-ep-shared-runtime-repeat/cluster/all-layer-50step-shared-tp-runtime.log`
- `logs/from-cluster/sprint258-tp-ep-shared-runtime-repeat/cluster/all-layer-50step-shared-tp-runtime-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--warmup 2 --iters 5 --decode-steps 50
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes --all-layers
```

All-layer decode-only gate:

| Metric | Sprint 256 10-step | Sprint 257 10-step | Sprint 258 50-step |
|---|---:|---:|---:|
| Passing layers | 43 / 43 | 43 / 43 | 43 / 43 |
| Shared TP runtime | 0 | 1 | 1 |
| Sum decode ms/token | 43.895297 | 46.024692 | 45.672166 |
| Projected slot-step tok/s | 729.007483 | 695.278962 | 700.645557 |
| Sum EP ms | 11.781577 | 13.353038 | 13.353355 |
| Sum dense ms | 7.624789 | 7.838466 | 7.698405 |
| Sum compose ms | 24.481659 | 24.829097 | 24.614787 |
| Wall ms | 33978.379725 | 28437.257957 | 30289.004553 |
| Checksum | 204721433 | 204721433 | 204721433 |
| Result | PASS | PASS | PASS |

## Decision

The shared TP runtime is correct and materially reduces scaffold wall time, but
the decode proxy regression persists in the longer gate. Do not treat Sprint
257 as a decode-performance promotion yet. The next implementation step should
either isolate why EP timing rises when the TP runtime is shared, or keep
Sprint 256 as the decode-speed base while hoisting expert descriptor bindings.
