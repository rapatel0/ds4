# Sprint 256 - TP/EP Shared Rank Buffers

Date: 2026-05-23
Status: Complete

## Overview

Sprint 256 continues the resident TP/EP scaffold work by hoisting fixed rank
buffers across the all-layer loop. For a fixed `slots/top_k` run, route
offsets, route-to-slot maps, streams, timing events, route input buffers,
gated buffers, down buffers, and compose buffers are layer-invariant. They
should be allocated once and reused.

This remains a scaffold gate, not generated-token serving throughput.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Keep the TP/EP codepath separate.
- Share rank streams/events and core route buffers across `--all-layers`.
- Preserve the existing single-layer diagnostic path.
- Run the all-layer decode-only gate on the V100 pod at `32` slots / `256K`.

## Non-Goals

- No PP scheduler changes.
- No production server integration.
- No expert descriptor-binding hoist yet.
- No TP runtime/KV-state hoist yet.
- No MTP.

## Design

`tools/ds4-v100-tp-ep-full-layer-smoke` now has a `SharedRankBuffers` owner
used only by `--all-layers` mode. It initializes once:

```text
per GPU:
  stream
  start/mid/stop events
  route offsets
  route-to-slot map
  input activation buffer
  gated intermediate buffer
  down output buffer
```

The existing lazy compose buffers also persist across layers when shared rank
buffers are used. Per-layer packed expert descriptors are still loaded and
freed per layer.

The shared path records:

```text
tp_ep_all_layer_rank_buffers_shared devices 8 core_bytes 3933984 PASS
```

and the final scaffold line records `shared_rank_buffers=1`.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | Adds all-layer shared rank buffers |
| `docs/sprints/SPRINT-256.md` | Plan and evidence |
| `docs/sprints/STATUS.md` | Status update |
| `docs/sprints/VISION.md` | Outcome update |
| `logs/from-cluster/sprint256-tp-ep-shared-rank-buffers/` | V100 evidence |

## Definition Of Done

- [x] Implementation stays in separate TP/EP code.
- [x] No PP scheduler files are modified.
- [x] Tool builds on the V100 pod with `CUDA_ARCH=sm_70`.
- [x] Single-layer diagnostic path still uses local rank buffers.
- [x] `--all-layers` uses shared rank buffers.
- [x] All-layer decode-only gate passes with shared dense cache, shared
      TurboMind API, descriptor checks disabled, and predecode probes disabled.
- [x] Evidence records `shared_rank_buffers=1`.
- [x] Evidence is copied to
      `logs/from-cluster/sprint256-tp-ep-shared-rank-buffers/`.
- [x] Status and vision docs are updated.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint256-tp-ep-shared-rank-buffers/cluster/all-layer-10step-shared-rank-buffers.log`
- `logs/from-cluster/sprint256-tp-ep-shared-rank-buffers/cluster/all-layer-10step-shared-rank-buffers-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--warmup 2 --iters 5 --decode-steps 10
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes --all-layers
```

10-step all-layer decode-only gate:

| Metric | Sprint 255 | Sprint 256 |
|---|---:|---:|
| Passing layers | 43 / 43 | 43 / 43 |
| Descriptor checks | 0 | 0 |
| Predecode probes | 0 | 0 |
| Shared TurboMind API | 1 | 1 |
| Shared rank buffers | 0 | 1 |
| Sum decode ms/token | 43.957040 | 43.895297 |
| Projected slot-step tok/s | 727.983506 | 729.007483 |
| Sum EP ms | 11.808814 | 11.781577 |
| Sum dense ms | 7.613046 | 7.624789 |
| Sum compose ms | 24.529236 | 24.481659 |
| Wall ms | 35565.756621 | 33978.379725 |
| Checksum | 204721433 | 204721433 |
| Result | PASS | PASS |

The rank-buffer hoist cuts scaffold wall time by `4.46%` versus Sprint 255.
The unchanged checksum confirms the hoisted route buffers preserve decode
semantics.

## Decision

Keep shared rank buffers for all-layer TP/EP scaffold runs. The next residency
target should be the TP runtime/KV lifecycle or expert descriptor bindings;
those are now larger remaining setup sources than streams, events, and fixed
route buffers.
