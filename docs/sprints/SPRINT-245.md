# Sprint 245 - TP/EP Dense FP16 Cache Admission Gate

Date: 2026-05-23
Status: Complete

## Overview

Sprint 244 proved that the representative TP/EP dense stage gets much faster
when packed F8 dense weights are expanded once into resident FP16 and executed
with cuBLAS FP16 Tensor Core GEMM. Sprint 245 answers the immediate appliance
question: can the target `32` slot / `256K` TP/EP topology afford that runtime
cache without overfilling 32 GiB V100s?

This sprint does not promote FP16 as the source format. The source pack remains
quantized. The new contract accounting estimates a runtime choice where dense
F8 TP tensors are materialized as FP16 execution weights on startup. It reports
both a conservative keep-both case and the practical serving case where the
runtime FP16 cache replaces the cacheable dense source tensors in VRAM.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Extend the separate TP/EP pack contract with dense FP16 cache admission
  accounting.
- Use real production-pack metadata, not hand estimates.
- Report per-GPU:
  - base TP/EP memory;
  - cacheable F8 dense packed bytes;
  - F8-to-FP16 runtime cache bytes;
  - BF16-to-FP16 shadow bytes for V100 execution;
  - keep-packed total;
  - replace-source total;
  - remaining headroom versus 32 GiB.
- Validate against the V100 pack at `32` slots / `256K` / F8 KV.

## Non-Goals

- No PP scheduler edits.
- No `ds4_v100_scheduler.*` changes.
- No server/API integration.
- No MTP.
- No claim that FP16 dense cache is the final optimized kernel path.
- No end-to-end generated tok/s claim.

## Design

For every `dense_tp` row in the TP/EP pack contract:

```text
source shape: [cols x rows]
TP shard rows: ceil(rows / 8)
packed shard bytes: contract bytes_estimate
runtime FP16 shard bytes: cols * TP shard rows * 2
```

The contract records two memory interpretations:

```text
keep-packed:
  base resident source pack + dense FP16 runtime cache

replace-source:
  base resident source pack
  - cacheable dense source bytes
  + dense FP16 runtime cache bytes
```

The replace-source case is the useful appliance target because the TP/EP
runtime can load dense execution weights into an FP16 arena at startup while
keeping the quantized source pack as the disk/offline artifact.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-pack-contract.c` | Dense FP16 cache admission accounting |
| `docs/sprints/SPRINT-245.md` | Plan and evidence |
| `docs/sprints/STATUS.md` | Status update |
| `docs/sprints/VISION.md` | Outcome update |
| `logs/from-cluster/sprint245-tp-ep-dense-f16-cache-contract/` | V100 evidence |

## Definition Of Done

- [x] Implementation stays in the separate TP/EP contract tool.
- [x] No PP scheduler files are modified.
- [x] The tool builds locally.
- [x] The tool builds on the V100 pod.
- [x] The real production pack contract runs at `32` slots / `256K` / F8 KV.
- [x] Evidence includes base memory and dense-FP16 replace-source memory.
- [x] Evidence is copied to
      `logs/from-cluster/sprint245-tp-ep-dense-f16-cache-contract/`.
- [x] Status and vision docs are updated with the decision.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Pack:
`/workspace/packs/ds4-appliance-full-tm-gated-s181`

Command shape:

```text
slots=32
ctx=262144
KV=f8_e4m3_b128
reserve=2.0 GiB/GPU
scratch=1.5 GiB/GPU
topology=PP1/TP8/EP8
MTP=off
```

Logs:

- `logs/from-cluster/sprint245-tp-ep-dense-f16-cache-contract/tp-ep-dense-f16-cache-contract.log`
- `logs/from-cluster/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-memory-summary.tsv`
- `logs/from-cluster/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.md`

The contract emits `4096` dense TP rows, `2920` F8 dense rows eligible for
FP16 cache, `1176` BF16 dense rows eligible for FP16 shadow, `5496`
replicated control rows, `688` EP expert rows, and `840` KV/state rows.

Per GPU at the target shape:

| Metric | GiB |
|---|---:|
| Base total, including reserve | 27.024 |
| F8 packed dense eligible bytes | 0.687 |
| F8-to-FP16 runtime cache | 1.364 |
| BF16 packed shadowable bytes | 0.319 |
| BF16-to-FP16 shadow | 0.319 |
| Keep-packed total | 28.707 |
| Replace-source total | 27.701 |
| Replace-source headroom vs 32 GiB | 4.299 |

All eight GPUs are balanced with the same totals.

## Decision

Dense FP16 runtime caching is memory-admissible for the target
`32` slot / `256K` TP/EP topology if cacheable dense source tensors are
replaced in VRAM by execution-format FP16 weights. The resulting estimate is
`27.701 GiB` per GPU including the existing `2.0 GiB` reserve, leaving
`4.299 GiB` of physical headroom.

This changes the next implementation priority. Instead of trying another
small per-tile HMMA kernel immediately, the next sprint should add a TP/EP
dense-cache loader/runtime path for all dense tensors, then benchmark the
resident 43-layer path. If memory behavior and correctness hold, packed
low-bit dense kernels can be optimized against the FP16/cuBLAS ceiling with a
working fallback already in place.
