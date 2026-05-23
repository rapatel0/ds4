# Sprint 255 - TP/EP Shared TurboMind API

Date: 2026-05-23
Status: Complete

## Overview

Sprint 255 starts converting the all-layer TP/EP scaffold from a sequence of
per-layer tool invocations into a resident runtime. Sprint 254 removed
benchmark-only validation probes, but the all-layer loop still opened,
initialized, shut down, and closed the TurboMind API once per transformer
layer. This sprint hoists that lifecycle across the full 43-layer loop.

This remains a scaffold gate, not generated-token serving throughput.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Keep the TP/EP codepath separate.
- Share one TurboMind dynamic library handle and API lifecycle across
  `--all-layers`.
- Preserve the existing single-layer path for focused diagnostics.
- Run the all-layer decode-only gate on the V100 pod at `32` slots / `256K`.

## Non-Goals

- No PP scheduler changes.
- No production server integration.
- No route-buffer or expert-binding hoist yet.
- No TP runtime-state hoist yet.
- No MTP.

## Design

`tools/ds4-v100-tp-ep-full-layer-smoke` now has a `SharedApi` owner used only
by `--all-layers` mode:

```text
all-layer mode:
  dlopen TurboMind once
  init all 8 devices once
  run all 43 layers with shared API handle
  shutdown once
  dlclose once

single-layer mode:
  preserve old local dlopen/init/shutdown path
```

The shared path records:

```text
tp_ep_all_layer_turbomind_api_shared devices 8 PASS
```

and the final scaffold line records `shared_api=1`.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | Adds all-layer shared TurboMind API lifecycle |
| `docs/sprints/SPRINT-255.md` | Plan and evidence |
| `docs/sprints/STATUS.md` | Status update |
| `docs/sprints/VISION.md` | Outcome update |
| `logs/from-cluster/sprint255-tp-ep-shared-api/` | V100 evidence |

## Definition Of Done

- [x] Implementation stays in separate TP/EP code.
- [x] No PP scheduler files are modified.
- [x] Tool builds on the V100 pod with `CUDA_ARCH=sm_70`.
- [x] Single-layer diagnostic path still uses local API lifecycle.
- [x] `--all-layers` uses shared TurboMind API lifecycle.
- [x] All-layer decode-only gate passes with shared dense cache, descriptor
      checks disabled, and predecode probes disabled.
- [x] Evidence records `shared_api=1`.
- [x] Evidence is copied to
      `logs/from-cluster/sprint255-tp-ep-shared-api/`.
- [x] Status and vision docs are updated.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint255-tp-ep-shared-api/cluster/all-layer-10step-shared-api.log`
- `logs/from-cluster/sprint255-tp-ep-shared-api/cluster/all-layer-10step-shared-api-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--warmup 2 --iters 5 --decode-steps 10
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes --all-layers
```

10-step all-layer decode-only gate:

| Metric | Sprint 254 | Sprint 255 |
|---|---:|---:|
| Passing layers | 43 / 43 | 43 / 43 |
| Descriptor checks | 0 | 0 |
| Predecode probes | 0 | 0 |
| Shared TurboMind API | 0 | 1 |
| Sum decode ms/token | 44.848746 | 43.957040 |
| Projected slot-step tok/s | 713.509362 | 727.983506 |
| Sum EP ms | 11.806264 | 11.808814 |
| Sum dense ms | 8.126768 | 7.613046 |
| Sum compose ms | 24.910200 | 24.529236 |
| Wall ms | 37819.503379 | 35565.756621 |
| Checksum | 204721433 | 204721433 |
| Result | PASS | PASS |

The shared API lifecycle cuts scaffold wall time by `5.96%` versus Sprint 254.
The summed decode proxy also improves slightly, but the stage split confirms
this is mainly residency cleanup rather than a new kernel breakthrough.

## Decision

Keep shared TurboMind API lifecycle for all-layer TP/EP scaffold runs. The next
residency work should hoist route buffers, streams/events, expert descriptor
bindings, and then the TP runtime/KV state. Those are the remaining setup
churn sources before this scaffold can become a serving loop.
