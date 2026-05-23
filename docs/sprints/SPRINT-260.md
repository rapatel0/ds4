# Sprint 260 - TP/EP Resident Expert Bindings

Date: 2026-05-23
Status: Complete

## Overview

Sprint 260 hoists the active MXFP4 expert bindings into an all-layer resident
cache. Before this sprint, each layer read TurboMind expert bytes from the pack,
allocated device buffers, built pointer tables, ran decode, and freed those
buffers. That is not a production appliance shape: expert weights should be
device resident.

This remains a scaffold gate, not generated-token serving throughput.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Keep the TP/EP codepath separate.
- Cache active gated and down expert bindings for all 43 layers across all 8
  V100s.
- Preserve a local per-layer expert-binding mode for A/B diagnostics.
- Validate at `32` slots / `256K`, local TP runtime, shared dense cache,
  shared TurboMind API, and shared rank buffers.

## Implementation

`tools/ds4-v100-tp-ep-full-layer-smoke` now supports:

```text
--shared-expert-bindings
--local-expert-bindings
```

The default is shared expert bindings. The resident cache loads the active six
local experts per GPU for each layer and tensor family:

```text
43 layers x 8 GPUs x 6 active local experts x {gate_up, down}
```

The V100 gate reports:

```text
tp_ep_all_layer_expert_bindings_shared layers 43 devices 8
  bytes 27594326016 bytes_per_gpu 3449290752 PASS
```

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint260-expert-bindings-ab/cluster/local-expert-50step.log`
- `logs/from-cluster/sprint260-expert-bindings-ab/cluster/local-expert-50step-summary.log`
- `logs/from-cluster/sprint260-expert-bindings-ab/cluster/shared-expert-50step.log`
- `logs/from-cluster/sprint260-expert-bindings-ab/cluster/shared-expert-50step-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--warmup 2 --iters 5 --decode-steps 50
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes --local-tp-runtime --all-layers
```

A/B result:

| Metric | Local expert bindings | Shared expert bindings |
|---|---:|---:|
| Passing layers | 43 / 43 | 43 / 43 |
| Shared expert bindings | 0 | 1 |
| Resident expert bytes/GPU | 0 | 3449290752 |
| Sum decode ms/token | 43.688136 | 44.131138 |
| Projected slot-step tok/s | 732.464293 | 725.111599 |
| Sum EP ms | 11.800559 | 12.207359 |
| Sum dense ms | 7.729216 | 7.554478 |
| Sum compose ms | 24.151182 | 24.363489 |
| Wall ms | 35770.339339 | 14338.419135 |
| Checksum | 204721433 | 204721433 |
| Result | PASS | PASS |

## Decision

Keep shared expert bindings as the default all-layer path because it matches the
production appliance requirement: expert weights are resident rather than
reloaded per layer. The decode proxy is slightly lower in this run, but the
checksum is stable and setup wall time drops by about `59.9%`. Next work should
move from scaffold residency toward a real serving loop or collapse the
EP/dense/compose boundary further.
