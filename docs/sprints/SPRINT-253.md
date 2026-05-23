# Sprint 253 - TP/EP Decode-Only Harness Repair

Date: 2026-05-23
Status: Complete

## Overview

Sprint 252 exposed a harness-specific failure: the all-layer scaffold passed
with `--compose-next-hidden`, but the cleaner decode-only benchmark path failed
early with `invalid resource handle`. Sprint 253 repairs that decode-only path
so the TP/EP scaffold can be measured without the extra one-shot compose
validation stage.

This remains a scaffold gate, not generated-token serving throughput.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Fix the decode-only all-layer harness failure.
- Preserve strict descriptor validation as the default.
- Preserve shared dense cache and descriptor-bypass modes.
- Run the decode-only all-layer gate on the V100 pod at `32` slots / `256K`.

## Non-Goals

- No PP scheduler changes.
- No production server integration.
- No logits-equivalent output.
- No MTP.
- No TurboMind/API handle hoist yet.

## Design

The failure occurred after TurboMind repeat/timing probes and before the first
resident dense decode-loop setup kernel. The compose-enabled path happened to
run an extra dense setup path first, which avoided the stale CUDA error state.

`prepare_resident_f8_dense()` now drains the current device's CUDA error state
before launching local dense setup conversion kernels. This keeps the
decode-only path from tripping on a stale prior error while preserving explicit
checks after the setup kernels themselves.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | Decode-only setup repair |
| `docs/sprints/SPRINT-253.md` | Plan and evidence |
| `docs/sprints/STATUS.md` | Status update |
| `docs/sprints/VISION.md` | Outcome update |
| `logs/from-cluster/sprint253-tp-ep-decode-only-harness/` | V100 evidence |

## Definition Of Done

- [x] Implementation stays in separate TP/EP code.
- [x] No PP scheduler files are modified.
- [x] Tool builds on the V100 pod with `CUDA_ARCH=sm_70`.
- [x] Decode-only all-layer gate passes with shared dense cache and descriptor
      checks disabled.
- [x] Evidence records `descriptor_checks=0`.
- [x] Evidence is copied to
      `logs/from-cluster/sprint253-tp-ep-decode-only-harness/`.
- [x] Status and vision docs are updated.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint253-tp-ep-decode-only-harness/cluster/all-layer-10step-decode-only.log`
- `logs/from-cluster/sprint253-tp-ep-decode-only-harness/cluster/all-layer-10step-decode-only-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--warmup 2 --iters 5 --decode-steps 10
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --all-layers
```

10-step decode-only gate:

| Metric | Sprint 252 compose-enabled | Sprint 253 decode-only |
|---|---:|---:|
| Passing layers | 43 / 43 | 43 / 43 |
| Descriptor checks | 0 | 0 |
| Sum decode ms/token | 44.383590 | 44.035733 |
| Projected slot-step tok/s | 720.987187 | 726.682578 |
| Per-layer avg decode ms | 1.032177 | 1.024087 |
| Per-layer min decode ms | 1.000416 | 0.982757 |
| Per-layer max decode ms | 1.221799 | 1.155619 |
| Sum EP ms | 11.784128 | 11.804094 |
| Sum dense ms | 7.933067 | 7.744769 |
| Sum compose ms | 24.659710 | 24.482197 |
| Wall ms | 46990.435640 | 39951.007721 |
| Result | PASS | PASS |

Shared dense cache for the decode-only run:

| Metric | Value |
|---|---:|
| Dense rows | 4096 |
| Source bytes | 8640413696 |
| Cache bytes | 14451998720 |
| Cache build ms | 7887.606836 |
| Result | PASS |

## Decision

Decode-only all-layer scaffold measurement is restored. Use this mode as the
default lightweight TP/EP scaffold benchmark after strict descriptor validation
has passed.

The next sprint should stop fixing scaffold-only measurement issues and hoist
real runtime state: TurboMind/API handles, route buffers, expert bindings, and
event/stream lifecycle across the 43-layer loop.
