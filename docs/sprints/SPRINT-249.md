# Sprint 249 - TP/EP Layer-Parametric Resident Loop

Date: 2026-05-23
Status: Complete

## Overview

Sprint 248 proved the descriptor-selected dense execution table across all
transformer layers. Sprint 249 moves the resident TP/EP full-layer smoke off
hardcoded layer `2` tensor names and validates representative DS4 layer
families at the target practical-serving shape: `32` slots, `256K` context,
TP8/EP8, MTP off.

This is still not a production server and not generated-token throughput. It is
a resident TP/EP layer scaffold gate that combines cache-backed FP16 dense
compose, real TurboMind MXFP4 EP experts, sharded KV row selection, and
next-hidden composition for multiple layer schedules.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Make the TP/EP full-layer smoke select composition tensors from
  `--layer N` instead of hardcoded layer `2`.
- Preserve the dense FP16 cache-backed composition path.
- Select DS4 compressed/indexer KV behavior from the layer ratio:
  - layers `0-1`: SWA-only, no compressed/indexer rows expected
  - even layers `2..42`: ratio `4`, indexer KV present
  - odd layers `3..41`: ratio `128`, no indexer KV
- Validate representative layers `0`, `1`, `2`, `3`, and `42` on the V100 pod.
- Record decode-loop stage timing and final pass/fail for each layer.

## Non-Goals

- No PP scheduler changes.
- No generic PP/TP scheduler abstraction.
- No production server integration.
- No all-43-layer hidden-state recurrence yet.
- No MTP.
- No generated-token throughput claim.

## Design

The full-layer smoke now derives layer-local tensor names from the requested
layer:

```text
blk.<layer>.attn_output_b.weight
blk.<layer>.ffn_down_shexp.weight
```

KV row selection also follows the DS4 layer compression schedule instead of
assuming every layer has ratio-4 indexer KV:

```text
layer 0-1       -> ratio 0, no compressed/indexer row expected
layer even >= 2 -> ratio 4, indexer row expected
layer odd       -> ratio 128, no indexer row expected
```

The final scaffold pass condition now treats compressed-state rows as required
only for non-SWA layers. This fixed the prior false failure on layers `0` and
`1`, where `comp_rows=0` is correct.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | Layer-parametric TP/EP full-layer smoke |
| `docs/sprints/SPRINT-249.md` | Plan and evidence |
| `docs/sprints/STATUS.md` | Status update |
| `docs/sprints/VISION.md` | Outcome update |
| `logs/from-cluster/sprint249-tp-ep-layer-parametric/` | V100 evidence |

## Definition Of Done

- [x] Implementation stays in separate TP/EP code.
- [x] No PP scheduler files are modified.
- [x] Tool builds on the V100 pod with `CUDA_ARCH=sm_70`.
- [x] SWA-only layers `0` and `1` pass without requiring compression rows.
- [x] Ratio-4 layers `2` and `42` pass with indexer KV selected.
- [x] Ratio-128 layer `3` passes without indexer KV.
- [x] Evidence records decode ms/step, slot-step tok/s, EP/dense/compose stage
      timings, KV ratio, and final PASS.
- [x] Evidence is copied to
      `logs/from-cluster/sprint249-tp-ep-layer-parametric/clean/`.
- [x] Status and vision docs are updated.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Pack:
`/workspace/packs/ds4-appliance-full-tm-gated-s181`

Contract:
`/workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv`

Logs:

- `logs/from-cluster/sprint249-tp-ep-layer-parametric/clean/summary.log`
- `logs/from-cluster/sprint249-tp-ep-layer-parametric/clean/layer-0-cache-f16-compose-10steps.log`
- `logs/from-cluster/sprint249-tp-ep-layer-parametric/clean/layer-1-cache-f16-compose-10steps.log`
- `logs/from-cluster/sprint249-tp-ep-layer-parametric/clean/layer-2-cache-f16-compose-10steps.log`
- `logs/from-cluster/sprint249-tp-ep-layer-parametric/clean/layer-3-cache-f16-compose-10steps.log`
- `logs/from-cluster/sprint249-tp-ep-layer-parametric/clean/layer-42-cache-f16-compose-10steps.log`

Command shape:

```text
--slots 32 --top-k 6 --ctx 262144 implied by runtime
--compose-next-hidden --decode-steps 10
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
```

Representative layer results:

| Layer | DS4 ratio | Dense rows | KV rows | Comp rows | Decode ms/step | Slot-step tok/s | EP ms/step | Dense ms/step | Compose ms/step | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 64 | 8 | 0 | 1.031754 | 31015.142077 | 0.297864 | 0.172395 | 0.561374 | PASS |
| 1 | 0 | 64 | 8 | 0 | 1.181511 | 27083.969701 | 0.323040 | 0.214812 | 0.643509 | PASS |
| 2 | 4 | 112 | 16 | 8 | 0.999333 | 32021.345429 | 0.270880 | 0.177647 | 0.550679 | PASS |
| 3 | 128 | 80 | 8 | 8 | 1.032074 | 31005.534682 | 0.279415 | 0.181605 | 0.570920 | PASS |
| 42 | 4 | 112 | 16 | 8 | 1.006666 | 31788.100522 | 0.277434 | 0.177258 | 0.551839 | PASS |

All five runs reported:

```text
decode_pass 1
compose_pass 1
repeat_bad 0
repeat_nan 0
finite_bad 0
tp_ep_full_layer_scaffold ... PASS
```

## Decision

The TP/EP full-layer scaffold is now layer-parametric for representative DS4
layer families. The earlier layer-2-only hardcoding is no longer the next
blocker.

The next sprint should build the first resident all-layer TP/EP loop in a
separate TP/EP tool or runtime path. That loop should iterate layer descriptors
without per-layer process startup, maintain hidden shards across layers, and
report an all-layer decode-loop proxy at `32` slots / `256K` before any server
integration.
