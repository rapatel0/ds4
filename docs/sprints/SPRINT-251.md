# Sprint 251 - TP/EP Shared Dense Cache Residency

Date: 2026-05-23
Status: Complete

## Overview

Sprint 250 added a one-process all-layer TP/EP scaffold, but each layer still
rebuilt its dense FP16 cache. Sprint 251 hoists that dense cache out of the
per-layer runner in `--all-layers` mode: the tool now materializes the full
dense contract once, then reuses that shared cache for all 43 layers.

This keeps the source pack quantized. The FP16 cache is a runtime execution
cache for V100 tensor-core dense paths, not a source-format change.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Let the contract parser load all layers for shared-cache setup.
- Materialize one all-layer dense FP16 cache in `--all-layers` mode.
- Reuse that cache for every per-layer resident decode-loop call.
- Preserve single-layer behavior.
- Compare the 10-step all-layer gate against Sprint 250.

## Non-Goals

- No PP scheduler changes.
- No production server integration.
- No logits-equivalent full model output.
- No generated-token throughput claim.
- No MTP.
- No EP expert cache hoist yet.
- No shared TP runtime/TurboMind handle hoist yet.

## Design

`parse_contract(path, layer, ...)` now accepts `layer < 0` as "all layers."
Single-layer callers still pass a concrete layer and receive the same filtered
rows as before.

In `--all-layers` mode, when `--dense-f16-cache-compose` is enabled:

```text
parse full contract once
prepare_dense_f16_cache(full_rows) once
for layer in 0..42:
  parse layer rows
  run layer scaffold with shared cache pointer
free shared cache once
```

The per-layer runner still owns the rest of its runtime state. That is now the
next optimization target.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | All-layer shared dense-cache path |
| `docs/sprints/SPRINT-251.md` | Plan and evidence |
| `docs/sprints/STATUS.md` | Status update |
| `docs/sprints/VISION.md` | Outcome update |
| `logs/from-cluster/sprint251-tp-ep-shared-dense-cache/` | V100 evidence |

## Definition Of Done

- [x] Implementation stays in separate TP/EP code.
- [x] No PP scheduler files are modified.
- [x] Tool builds on the V100 pod with `CUDA_ARCH=sm_70`.
- [x] Shared all-layer dense cache materializes once.
- [x] All-layer 10-step gate passes 43/43 layers at `32` slots / `256K`.
- [x] Evidence records cache size/build time, per-layer rows, and aggregate
      scaffold metrics.
- [x] Evidence is copied to
      `logs/from-cluster/sprint251-tp-ep-shared-dense-cache/`.
- [x] Status and vision docs are updated.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Pack:
`/workspace/packs/ds4-appliance-full-tm-gated-s181`

Contract:
`/workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv`

Logs:

- `logs/from-cluster/sprint251-tp-ep-shared-dense-cache/cluster/all-layer-10step-shared-cache.log`
- `logs/from-cluster/sprint251-tp-ep-shared-dense-cache/cluster/all-layer-10step-shared-cache-summary.log`

Shared dense cache:

| Metric | Value |
|---|---:|
| Dense rows | 4096 |
| Source bytes | 8640413696 |
| Cache bytes | 14451998720 |
| Cache aligned bytes | 14451998720 |
| Max temp bytes | 132382720 |
| Cache build ms | 7772.591153 |
| Result | PASS |

10-step all-layer gate:

| Metric | Sprint 250 per-layer cache | Sprint 251 shared cache |
|---|---:|---:|
| Passing layers | 43 / 43 | 43 / 43 |
| Sum decode ms/token | 45.356852 | 43.753529 |
| Projected slot-step tok/s | 705.516343 | 731.369579 |
| Per-layer avg decode ms | 1.054810 | 1.017524 |
| Per-layer min decode ms | 0.995387 | 0.981924 |
| Per-layer max decode ms | 1.451088 | 1.054658 |
| Sum EP ms | 12.009343 | 11.837140 |
| Sum dense ms | 8.064360 | 7.613544 |
| Sum compose ms | 25.277469 | 24.296721 |
| Wall ms | 91879.358460 | 74382.064295 |
| Result | PASS | PASS |

The shared-cache gate improves wall time by about `19.0%` and the scaffold
decode proxy by about `3.7%`. The wall-time gain is the important result: dense
cache state no longer gets rebuilt per layer. The decode proxy remains
compose/synchronization dominated.

## Decision

Shared dense cache residency is promoted for the TP/EP all-layer scaffold.
The remaining scaffold artifacts are now clearer:

- per-layer TP runtime open/close,
- per-layer TurboMind init/shutdown,
- per-layer expert packed-byte binding,
- per-layer route/input buffer allocation,
- no true hidden-shard recurrence across layers.

The next sprint should hoist TurboMind/API and rank buffers out of the
per-layer runner, then move toward a real resident all-layer loop.
