# Sprint 263 - TP/EP Direct Remote Compose Probe

Date: 2026-05-23
Status: Complete

## Overview

Sprint 263 tests whether the compose/all-to-all stage can avoid explicit
peer-copy staging. The current path copies each source rank's EP contribution
shard into destination-local staging buffers, then launches a compose kernel.
This sprint adds an opt-in direct remote compose mode where the compose kernel
reads source `d_ep_contrib_all` shards directly over peer memory.

This is a measurement sprint, not generated-token serving throughput.

## Implementation

`tools/ds4-v100-tp-ep-full-layer-smoke` now supports:

```text
--direct-remote-compose
```

The default remains staged peer copies. Direct remote compose is disabled for
FP16 EP return and is only used with the FP32 fused-sum compose path.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint263-direct-remote-compose-ab/cluster/staged-compose-50step.log`
- `logs/from-cluster/sprint263-direct-remote-compose-ab/cluster/staged-compose-50step-summary.log`
- `logs/from-cluster/sprint263-direct-remote-compose-ab/cluster/direct-compose-50step.log`
- `logs/from-cluster/sprint263-direct-remote-compose-ab/cluster/direct-compose-50step-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--warmup 2 --iters 5 --decode-steps 50
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes
--local-tp-runtime --shared-expert-bindings --overlap-ep-dense --all-layers
```

A/B result:

| Metric | Staged compose | Direct remote compose |
|---|---:|---:|
| Passing layers | 43 / 43 | 43 / 43 |
| Direct remote compose | 0 | 1 |
| Sum decode ms/token | 38.061178 | 50.437041 |
| Projected slot-step tok/s | 840.751688 | 634.454351 |
| Sum EP/overlap ms | 12.684805 | 12.655482 |
| Sum dense ms | 0.000000 | 0.000000 |
| Sum compose ms | 25.368965 | 37.776787 |
| Wall ms | 13959.169457 | 14450.961466 |
| Checksum | 204721433 | 204721433 |
| Result | PASS | PASS |

## Decision

Reject direct remote compose as the production path. It preserves correctness
but increases compose time by `48.9%` and regresses projected scaffold
throughput by `24.5%`. The V100/NVLink setup prefers explicit staged peer
copies over remote reads inside the compose kernel. Keep the option only as a
diagnostic. The next compose work should target better staged-copy scheduling
or a fused destination-side reduction after staged data arrives.
