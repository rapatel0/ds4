# Sprint 248 - TP/EP All-Layer Dense Execution Table

Date: 2026-05-23
Status: Complete

## Overview

Sprint 247 proved that the representative layer-2 decode loop can consume
cache-backed FP16 dense weights. Sprint 248 lifts that from two hardcoded
composition tensors to a descriptor-selected dense execution table across the
transformer layers.

The workbench still is not a full serving runtime. It isolates the dense
execution side of the TP/EP path: build the dense FP16 cache from the contract,
group dense rows by layer/tensor/GPU ownership, then execute cache-backed
FP16/cuBLAS GEMMs for every complete layer dense group.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Extend the TP/EP dense-cache smoke with an `--execute-table` mode.
- Build descriptor-selected dense groups from the TP/EP contract.
- Exclude embedding/output rows from execution-table timing while keeping the
  cache loader able to materialize the full dense contract.
- Run a layer-2 table gate and an all-layer table gate on the V100 pod.
- Report group counts, GEMM counts, FLOPs, timing, TFLOP/s, checksum, and
  nonfinite counts.

## Non-Goals

- No PP scheduler edits.
- No `ds4_v100_scheduler.*` changes.
- No serving integration.
- No EP or KV work inside the dense table timing.
- No MTP.
- No final generated-token throughput claim.

## Design

The dense table is derived from contract rows:

```text
dense_tp rows -> FP16 cache arena
layer >= 0 dense rows -> group by (layer, tensor_id)
complete group == one row per TP rank
for each group:
  fill deterministic FP16 activation [slots x cols]
  run cublasGemmEx(W_cache^T, X) on every GPU
  optional output checksum/nonfinite pass
```

This gives a descriptor-selected all-layer dense path without relying on the
layer-2 hardcoded composition names.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-dense-cache-smoke.cu` | `--execute-table` dense execution-table workbench |
| `docs/sprints/SPRINT-248.md` | Plan and evidence |
| `docs/sprints/STATUS.md` | Status update |
| `docs/sprints/VISION.md` | Outcome update |
| `logs/from-cluster/sprint248-tp-ep-dense-table/` | V100 evidence |

## Definition Of Done

- [x] Implementation stays in a separate TP/EP tool.
- [x] No PP scheduler files are modified.
- [x] Tool builds on the V100 pod with `CUDA_ARCH=sm_70`.
- [x] Layer-2 dense execution-table gate passes.
- [x] All-layer dense execution-table gate passes.
- [x] Evidence records group/GEMM counts, timing, TFLOP/s, checksum, and
      nonfinite count.
- [x] Evidence is copied to
      `logs/from-cluster/sprint248-tp-ep-dense-table/`.
- [x] Status and vision docs are updated.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Pack:
`/workspace/packs/ds4-appliance-full-tm-gated-s181`

Contract:
`/workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv`

Logs:

- `logs/from-cluster/sprint248-tp-ep-dense-table/layer2-dense-table-execute.log`
- `logs/from-cluster/sprint248-tp-ep-dense-table/all-layer-dense-table-execute.log`

Layer-2 table:

| Metric | Value |
|---|---:|
| Groups | 14 |
| GEMMs per iteration | 112 |
| Timed iterations | 10 |
| ms/iteration | 1.384323 |
| Dense table TFLOP/s | 6.992914 |
| Nonfinite outputs | 0 |
| Result | PASS |

All-layer table:

| Metric | Value |
|---|---:|
| Groups | 510 |
| GEMMs per iteration | 4080 |
| Timed iterations | 5 |
| FLOPs per iteration | 394684006400 |
| ms/iteration | 51.003671 |
| Dense table TFLOP/s | 7.738345 |
| Nonfinite outputs | 0 |
| Result | PASS |

Full cache materialization remains intact:

```text
tp_ep_dense_cache_smoke layer all rows 4096 source_gib 8.047012
cache_aligned_gib 13.459473 PASS
```

## Decision

The TP/EP path now has a descriptor-selected dense execution table across all
transformer layers. This removes the hardcoded layer-2 dense selection as the
next blocker for all-layer work.

The measured dense-only all-layer table is `51.003671 ms` per 32-slot pass in
this workbench. That is useful but not sufficient for serving: the next sprint
must compose this table with EP routed experts, KV/update, and layer-to-layer
hidden state flow in a resident all-layer loop so the bottleneck moves from
isolated dense timing to full TP/EP decode timing.
