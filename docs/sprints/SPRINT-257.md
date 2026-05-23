# Sprint 257 - TP/EP Shared TP Runtime

Date: 2026-05-23
Status: Complete

## Overview

Sprint 257 hoists the TP runtime/KV allocator across the all-layer TP/EP
scaffold. Sprint 256 still opened and closed the `ds4_v100_tp_runtime` once
per layer, rebuilding the 256K KV/compression/scratch arenas repeatedly. This
sprint opens that runtime once and reuses it while preserving the per-layer
`dense_kv_slice(layer, slot, position)` call.

This remains a scaffold gate, not generated-token serving throughput.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Keep the TP/EP codepath separate.
- Share TP runtime/KV/compression/scratch arenas across `--all-layers`.
- Preserve single-layer diagnostics.
- Run the all-layer decode-only gate on the V100 pod at `32` slots / `256K`.

## Non-Goals

- No PP scheduler changes.
- No production server integration.
- No expert descriptor-binding hoist yet.
- No MTP.

## Design

`tools/ds4-v100-tp-ep-full-layer-smoke` now has a `SharedTpRuntime` owner used
only by `--all-layers` mode:

```text
all-layer mode:
  open TP runtime once
  allocate sharded KV/compression/scratch arenas once
  run dense_kv_slice(layer, slot, position) per layer
  close TP runtime once

single-layer mode:
  preserve old local TP runtime lifecycle
```

The shared path records:

```text
tp_ep_all_layer_tp_runtime_shared devices 8 slots 32 ctx 262144 ... PASS
```

and the final scaffold line records `shared_tp_runtime=1`.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | Adds all-layer shared TP runtime lifecycle |
| `docs/sprints/SPRINT-257.md` | Plan and evidence |
| `docs/sprints/STATUS.md` | Status update |
| `docs/sprints/VISION.md` | Outcome update |
| `logs/from-cluster/sprint257-tp-ep-shared-tp-runtime/` | V100 evidence |

## Definition Of Done

- [x] Implementation stays in separate TP/EP code.
- [x] No PP scheduler files are modified.
- [x] Tool builds on the V100 pod with `CUDA_ARCH=sm_70`.
- [x] Single-layer diagnostic path still uses local TP runtime lifecycle.
- [x] `--all-layers` uses shared TP runtime lifecycle.
- [x] All-layer decode-only gate passes with shared dense cache, shared
      TurboMind API, shared rank buffers, descriptor checks disabled, and
      predecode probes disabled.
- [x] Evidence records `shared_tp_runtime=1`.
- [x] Evidence is copied to
      `logs/from-cluster/sprint257-tp-ep-shared-tp-runtime/`.
- [x] Status and vision docs are updated.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint257-tp-ep-shared-tp-runtime/cluster/all-layer-10step-shared-tp-runtime.log`
- `logs/from-cluster/sprint257-tp-ep-shared-tp-runtime/cluster/all-layer-10step-shared-tp-runtime-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--warmup 2 --iters 5 --decode-steps 10
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes --all-layers
```

Shared runtime allocation:

| Metric | Value |
|---|---:|
| KV bytes/GPU | 3707940864 |
| Compression-state bytes/GPU | 1803550720 |
| Scratch bytes/GPU | 1610612736 |
| Runtime total bytes/GPU | 7122628608 |

10-step all-layer decode-only gate:

| Metric | Sprint 256 | Sprint 257 |
|---|---:|---:|
| Passing layers | 43 / 43 | 43 / 43 |
| Descriptor checks | 0 | 0 |
| Predecode probes | 0 | 0 |
| Shared TurboMind API | 1 | 1 |
| Shared rank buffers | 1 | 1 |
| Shared TP runtime | 0 | 1 |
| Sum decode ms/token | 43.895297 | 46.024692 |
| Projected slot-step tok/s | 729.007483 | 695.278962 |
| Sum EP ms | 11.781577 | 13.353038 |
| Sum dense ms | 7.624789 | 7.838466 |
| Sum compose ms | 24.481659 | 24.829097 |
| Wall ms | 33978.379725 | 28437.257957 |
| Checksum | 204721433 | 204721433 |
| Result | PASS | PASS |

The runtime hoist cuts scaffold wall time by `16.31%` versus Sprint 256, but
the decode proxy regresses by `4.85%`, mostly in EP timing. Treat the hoist as
correct residency progress, not yet a decode-speed promotion. A repeat gate is
needed before deciding whether the decode regression is noise, allocator side
effect, or runtime-state interaction.

## Decision

Keep shared TP runtime as the resident scaffold path because it removes a large
class of non-serving setup and preserves checksum. The next sprint should run a
repeat/longer decode gate and then either keep this path as the base or fix the
EP timing regression before hoisting expert descriptor bindings.
