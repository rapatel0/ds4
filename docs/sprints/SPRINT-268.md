# Sprint 268 - TP/EP Token-Major Position Advance

Date: 2026-05-23
Status: Complete

## Overview

Sprint 268 makes the token-major scaffold more serving-shaped by advancing the
logical context position for each token step. Previous token-major runs
traversed all layers for multiple token steps but reused the same `position`
for every step. That was acceptable for a scheduler scaffold, but it was not a
good proxy for decode state progression.

This remains scaffold throughput, not generated-token serving throughput.

## Implementation

In `--token-major-all-layers` mode, the tool now sets:

```text
layer_position = start_position + token_step
```

for every layer invocation. `kv_slot` remains the sequence slot id. The
per-layer token-major item log now prints the effective `position`.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint268-token-major-position-advance/cluster/position-advance-4step.log`
- `logs/from-cluster/sprint268-token-major-position-advance/cluster/position-advance-4step-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--warmup 0 --iters 1 --decode-steps 4
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes
--shared-expert-bindings --overlap-ep-dense --source-copy-schedule
--token-major-all-layers --all-layers
```

The default token-major shared TP runtime from Sprint 267 is active.

Result:

| Metric | Value |
|---|---:|
| Token steps | 4 |
| Position range | 1024-1027 |
| Layer invocations | 172 |
| Passing invocations | 172 |
| Shared TP runtime | 1 |
| Sum decode ms | 183.081848 |
| ms/token proxy | 45.770462 |
| Projected slot-step tok/s | 699.140856 |
| Sum EP/overlap ms | 93.872406 |
| Sum compose ms | 89.157724 |
| Wall ms | 11799.119372 |
| Checksum | 296236348 |
| Result | PASS |

## Decision

Keep position advance as the token-major default. It better matches decode
state progression and preserves correctness. The measured proxy improves from
Sprint 267's shared-runtime `47.902324 ms/token` to `45.770462 ms/token`.

Next work should run a longer continuous token-major gate and then replace the
scaffold's synthetic hidden/checksum flow with generated/continuation serving
measurement.
