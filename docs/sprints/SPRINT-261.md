# Sprint 261 - TP/EP EP-Dense Overlap

Date: 2026-05-23
Status: Complete

## Overview

Sprint 261 changes the decode scheduling shape inside the TP/EP scaffold.
Routed EP and dense tensor-core GEMMs are independent until next-hidden compose,
but the prior loop ran them serially on the same rank stream. This sprint adds
a separate dense stream per rank and an overlapped EP+dense mode.

This remains a scaffold gate, not generated-token serving throughput.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Keep the TP/EP codepath separate.
- Run dense cuBLAS GEMMs on a separate stream from routed EP.
- Add an A/B switch for serial versus overlapped scheduling.
- Validate at `32` slots / `256K` with resident expert bindings.

## Implementation

`RankState` now owns a `dense_stream`. Dense launches use that stream when
available. `run_decode_loop()` can now launch routed EP work and dense work
before the synchronization point, then compose after both are complete.

The tool supports:

```text
--overlap-ep-dense
--serial-ep-dense
```

Overlap is now the default. Serial mode remains available for diagnostics.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint261-ep-dense-overlap-ab/cluster/serial-50step.log`
- `logs/from-cluster/sprint261-ep-dense-overlap-ab/cluster/serial-50step-summary.log`
- `logs/from-cluster/sprint261-ep-dense-overlap-ab/cluster/overlap-50step.log`
- `logs/from-cluster/sprint261-ep-dense-overlap-ab/cluster/overlap-50step-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--warmup 2 --iters 5 --decode-steps 50
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes
--local-tp-runtime --shared-expert-bindings --all-layers
```

A/B result:

| Metric | Serial EP then dense | Overlapped EP+dense |
|---|---:|---:|
| Passing layers | 43 / 43 | 43 / 43 |
| Overlap EP+dense | 0 | 1 |
| Sum decode ms/token | 50.691201 | 37.822268 |
| Projected slot-step tok/s | 631.273270 | 846.062424 |
| Sum EP/overlap ms | 12.829893 | 12.611494 |
| Sum dense ms | 9.042506 | 0.000000 |
| Sum compose ms | 28.809735 | 25.202841 |
| Wall ms | 15373.979849 | 14077.073508 |
| Checksum | 204721433 | 204721433 |
| Result | PASS | PASS |

In overlap mode, the EP timing bucket includes the overlapped EP+dense wait;
the dense bucket is zero by construction. The total decode and checksum are
the decision fields.

## Decision

Promote EP+dense overlap as the default TP/EP scaffold schedule. It improves
projected scaffold throughput by `34.0%` in the same-binary A/B while
preserving the checksum. The next performance target is the compose/all-to-all
boundary, which is now the dominant remaining stage.
