# Sprint 246 - TP/EP Dense FP16 Cache Runtime Smoke

Date: 2026-05-23
Status: Complete

## Overview

Sprint 245 showed that a dense FP16 runtime cache is memory-admissible at the
target `32` slot / `256K` TP/EP shape. Sprint 246 turns that accounting into a
real V100 allocation and conversion smoke. The new tool reads the TP/EP pack
contract, allocates one dense FP16 cache arena per GPU, stages packed F8/BF16
dense shards through a temporary GPU buffer, converts them on device, and
keeps the FP16 cache resident until validation completes.

This remains a TP/EP-only runtime path. It does not touch the PP scheduler and
does not change the source pack format.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Add a new TP/EP dense-cache tool rather than extending PP abstractions.
- Allocate one FP16 dense cache arena per GPU from contract metadata.
- Convert both dense source dtypes needed on V100:
  - `f8_e4m3_b128` to FP16;
  - `bf16` to FP16.
- Validate both a layer-2 subset and the full dense contract on the V100 pod.
- Report actual V100 free-memory movement, source/cache bytes, conversion
  timing, checksums, and nonfinite counts.

## Non-Goals

- No PP scheduler edits.
- No `ds4_v100_scheduler.*` changes.
- No serving integration.
- No all-layer decode benchmark yet.
- No custom packed low-bit dense kernel promotion.
- No MTP.

## Design

The loader uses the TP/EP contract as the ownership source:

```text
for each dense_tp row:
  parse dtype, shape, owning_gpu, shard_index, source offset
  compute rows_per_gpu from bytes_estimate and source dtype
  reserve aligned space in that GPU's dense FP16 arena

for each GPU:
  cudaMalloc(dense_fp16_arena)
  cudaMalloc(max_source_row_temp)

for each dense_tp row:
  read packed source shard from pack file
  H2D into temp buffer
  convert temp -> dense_fp16_arena[offset] on the owning GPU
  checksum FP16 bits and count nonfinite values
```

The runtime cache is an execution-format arena. The source quantized pack
remains the offline artifact; a production runtime can decide whether to keep
or discard source dense bytes in VRAM after the cache is built.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-dense-cache-smoke.cu` | TP/EP dense FP16 cache allocation/conversion smoke |
| `Makefile` | CUDA build target and Darwin fallback |
| `docs/sprints/SPRINT-246.md` | Plan and evidence |
| `docs/sprints/STATUS.md` | Status update |
| `docs/sprints/VISION.md` | Outcome update |
| `logs/from-cluster/sprint246-tp-ep-dense-cache-runtime/` | V100 evidence |

## Definition Of Done

- [x] New implementation is in separate TP/EP files.
- [x] No PP scheduler files are modified.
- [x] The CUDA tool builds on the V100 pod with `CUDA_ARCH=sm_70`.
- [x] Layer-2 dense cache smoke passes.
- [x] Full-contract dense cache smoke passes.
- [x] Evidence records per-GPU memory and conversion statistics.
- [x] Evidence is copied to
      `logs/from-cluster/sprint246-tp-ep-dense-cache-runtime/`.
- [x] Status and vision docs are updated.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Pack:
`/workspace/packs/ds4-appliance-full-tm-gated-s181`

Contract:
`/workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv`

Logs:

- `logs/from-cluster/sprint246-tp-ep-dense-cache-runtime/layer2-dense-cache-smoke.log`
- `logs/from-cluster/sprint246-tp-ep-dense-cache-runtime/all-dense-cache-smoke.log`

Layer-2 subset:

| Metric | Value |
|---|---:|
| Total dense rows | 112 |
| Aggregate source bytes | 0.151901 GiB |
| Aggregate FP16 cache | 0.281738 GiB |
| Result | PASS |

Full dense contract:

| Metric | Per GPU |
|---|---:|
| Dense rows | 512 |
| F8 rows | 365 |
| BF16 rows | 147 |
| Source bytes | 1.005877 GiB |
| F8 source bytes | 0.687212 GiB |
| BF16 source bytes | 0.318665 GiB |
| FP16 cache arena | 1.682434 GiB |
| Max temporary source buffer | 126.250 MiB |
| Free before | 31.428 GiB |
| Free after cache+temp allocation | 29.618 GiB |
| Free after temp free | 29.743 GiB |
| Nonfinite count | 0 |

Aggregate full-contract result:

```text
tp_ep_dense_cache_smoke layer all rows 4096 source_gib 8.047012
cache_aligned_gib 13.459473 PASS
```

## Decision

The dense FP16 runtime cache is no longer just a spreadsheet estimate. The
full dense contract can be materialized on all eight V100s as FP16 arenas with
nonzero checksums and zero nonfinite values.

Next work should wire this arena into the TP/EP resident layer execution path
instead of repeatedly using two ad hoc resident dense tensors. Once the layer
path can select dense tensors from the cache, benchmark a resident all-layer
loop at `32` slots / `256K`, then decide whether custom packed low-bit dense
kernels are still needed immediately or can be optimized behind the FP16
runtime fallback.
