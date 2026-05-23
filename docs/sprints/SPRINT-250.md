# Sprint 250 - TP/EP All-Layer Scaffold Gate

Date: 2026-05-23
Status: Complete

## Overview

Sprint 249 made the TP/EP full-layer smoke layer-parametric. Sprint 250 adds an
`--all-layers` mode that executes layers `0..42` inside one process and emits
per-layer plus aggregate TP/EP scaffold metrics at the target practical-serving
shape: `32` slots, `256K` context, TP8/EP8, MTP off.

This is still a scaffold gate, not production serving. The current mode
reuses the layer runner and therefore still rebuilds per-layer runtime/cache
state inside the process. The value is that the TP/EP path now has a single
43-layer correctness and timing gate instead of shell-orchestrated layer
probes.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Add an `--all-layers` mode to the TP/EP full-layer smoke.
- Run all 43 transformer layers in one process.
- Emit one compact `tp_ep_all_layer_item` line per layer.
- Emit one aggregate `tp_ep_all_layer_scaffold` line with pass count, summed
  decode proxy, stage sums, wall time, and checksum.
- Validate the 43-layer gate on the V100 pod at `32` slots / `256K`.

## Non-Goals

- No PP scheduler changes.
- No generic PP/TP scheduler abstraction.
- No production server integration.
- No logits-equivalent full model output.
- No final generated-token throughput claim.
- No MTP.

## Design

`tools/ds4-v100-tp-ep-full-layer-smoke` now supports:

```text
--all-layers
```

In this mode the tool:

1. parses the normal command-line options once,
2. iterates layers `0..42`,
3. runs the existing layer-parametric TP/EP layer scaffold for each layer,
4. prints a compact per-layer result row, and
5. prints an aggregate scaffold row.

The aggregate projected slot-step throughput is computed from the sum of the
per-layer decode-loop `ms_per_step` values:

```text
projected_slot_step_tok_s = slots * 1000 / sum_decode_ms_per_token
```

That is a scaffold estimate. It excludes true all-layer hidden recurrence,
logits, sampling, server queueing, and real runtime reuse. It also includes
only the currently modeled EP, dense compose, peer return, and compose stages.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | Adds `--all-layers` mode and aggregate reporting |
| `docs/sprints/SPRINT-250.md` | Plan and evidence |
| `docs/sprints/STATUS.md` | Status update |
| `docs/sprints/VISION.md` | Outcome update |
| `logs/from-cluster/sprint250-tp-ep-all-layer-scaffold/` | V100 evidence |

## Definition Of Done

- [x] Implementation stays in separate TP/EP code.
- [x] No PP scheduler files are modified.
- [x] Tool builds on the V100 pod with `CUDA_ARCH=sm_70`.
- [x] `--all-layers` runs all 43 transformer layers in one process.
- [x] Short 2-step all-layer gate passes.
- [x] 10-step all-layer gate passes.
- [x] Evidence records per-layer rows and an aggregate scaffold row.
- [x] Evidence is copied to
      `logs/from-cluster/sprint250-tp-ep-all-layer-scaffold/`.
- [x] Status and vision docs are updated.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Pack:
`/workspace/packs/ds4-appliance-full-tm-gated-s181`

Contract:
`/workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv`

Logs:

- `logs/from-cluster/sprint250-tp-ep-all-layer-scaffold/cluster/all-layer-short.log`
- `logs/from-cluster/sprint250-tp-ep-all-layer-scaffold/cluster/all-layer-summary.log`
- `logs/from-cluster/sprint250-tp-ep-all-layer-scaffold/cluster/all-layer-10step.log`
- `logs/from-cluster/sprint250-tp-ep-all-layer-scaffold/cluster/all-layer-10step-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--compose-next-hidden --fuse-compose-sum
--dense-f16-cublas-compose --dense-f16-cache-compose
--all-layers
```

Short gate:

| Metric | Value |
|---|---:|
| Decode steps per layer | 2 |
| Passing layers | 43 / 43 |
| Sum decode ms/token | 53.936133 |
| Projected slot-step tok/s | 593.294300 |
| Sum EP ms | 12.766764 |
| Sum dense ms | 11.362042 |
| Sum compose ms | 29.798677 |
| Wall ms | 101999.984776 |
| Result | PASS |

10-step gate:

| Metric | Value |
|---|---:|
| Decode steps per layer | 10 |
| Passing layers | 43 / 43 |
| Sum decode ms/token | 45.356852 |
| Projected slot-step tok/s | 705.516343 |
| Per-layer avg decode ms | 1.054810 |
| Per-layer min decode ms | 0.995387 |
| Per-layer max decode ms | 1.451088 |
| Sum EP ms | 12.009343 |
| Sum dense ms | 8.064360 |
| Sum compose ms | 25.277469 |
| Wall ms | 91879.358460 |
| Checksum | 6174401222 |
| Result | PASS |

The 10-step worst layer was layer `4` at `1.451088 ms/step`; the fastest was
layer `22` at `0.995387 ms/step`.

## Decision

The separate TP/EP path now has a 43-layer scaffold gate. This is a meaningful
step beyond representative layer probes, but it also exposes the next real
engineering task: the all-layer path must become a resident runtime loop
instead of recreating per-layer state. The next sprint should move cache,
runtime arenas, TurboMind handles, route buffers, and hidden shards outside the
per-layer runner so the 43-layer wall time and projected decode proxy reflect a
serving-shaped loop.
