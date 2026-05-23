# Sprint 247 - TP/EP Dense Cache Compose Integration

Date: 2026-05-23
Status: Complete

## Overview

Sprint 246 materialized all dense TP rows into FP16 arenas, but the resident
layer loop still built private FP16 copies for the two dense tensors it used.
Sprint 247 wires the dense cache arena into the representative TP/EP decode
loop so dense execution can use cache pointers instead of per-op private
weight buffers.

This is still a bounded layer-2 resident loop, not serving. The purpose is to
prove the execution path can look up descriptor-owned dense weights from a
runtime cache arena and use them in the tensor-core dense path.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Add `--dense-f16-cache-compose` to the separate TP/EP full-layer smoke.
- Require `--dense-f16-cublas-compose` when cache-backed dense execution is
  selected.
- Build a layer-local dense FP16 cache arena from the TP/EP contract.
- Use cache pointers for the resident decode loop's two F8 composition
  tensors:
  - `blk.2.attn_output_b.weight`
  - `blk.2.ffn_down_shexp.weight`
- Run same-binary A/B/C at `32` slots / `256K` / `50` resident steps:
  - scalar dense;
  - private FP16/cuBLAS dense;
  - cache-backed FP16/cuBLAS dense.

## Non-Goals

- No PP scheduler edits.
- No `ds4_v100_scheduler.*` changes.
- No server/API integration.
- No all-layer decode yet.
- No MTP.
- No custom packed low-bit dense promotion.

## Design

The new option builds a `DenseF16Cache` from the already parsed layer contract:

```text
contract dense_tp rows -> per-GPU FP16 cache arena
source F8/BF16 shard -> temp GPU buffer -> FP16 cache arena[offset]
resident dense op -> lookup tensor_id + gpu -> __half* cache pointer
cublasGemmEx(cache_weight, activation_fp16) -> FP32 output shard
```

The resident dense op still owns its activation/output buffers and cuBLAS
handle. It does not own the cached weight pointer, so teardown avoids freeing
cache-backed weight memory twice.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | Dense cache option and decode-loop lookup |
| `docs/sprints/SPRINT-247.md` | Plan and evidence |
| `docs/sprints/STATUS.md` | Status update |
| `docs/sprints/VISION.md` | Outcome update |
| `logs/from-cluster/sprint247-tp-ep-dense-cache-compose/` | V100 evidence |

## Definition Of Done

- [x] Implementation stays in the separate TP/EP full-layer smoke.
- [x] No PP scheduler files are modified.
- [x] Tool builds on the V100 pod with `CUDA_ARCH=sm_70`.
- [x] Scalar dense control passes.
- [x] Private FP16/cuBLAS dense passes.
- [x] Cache-backed FP16/cuBLAS dense passes.
- [x] Evidence records `dense_f16_cache=1` in the decode-loop line.
- [x] Evidence is copied to
      `logs/from-cluster/sprint247-tp-ep-dense-cache-compose/`.
- [x] Status and vision docs are updated.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Command shape:

```text
slots=32
ctx=262144
top_k=6
layer=2
decode_steps=50
MTP=off
fuse_compose_sum=on
dense_compute_all=on
```

Logs:

- `logs/from-cluster/sprint247-tp-ep-dense-cache-compose/scalar-dense-fused-compose-50steps.log`
- `logs/from-cluster/sprint247-tp-ep-dense-cache-compose/private-f16-cublas-fused-compose-50steps.log`
- `logs/from-cluster/sprint247-tp-ep-dense-cache-compose/cache-f16-cublas-fused-compose-50steps.log`

| Mode | F16/cuBLAS | Cache | ms/step | Slot-step tok/s | EP ms/step | Dense ms/step | Compose ms/step | Checksum | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Scalar dense | 0 | 0 | 1.642514 | 19482.326340 | 0.312723 | 0.754608 | 0.575106 | 2382924023 | PASS |
| Private FP16/cuBLAS | 1 | 0 | 1.056807 | 30279.894858 | 0.311265 | 0.180614 | 0.564823 | 2515001 | PASS |
| Cache-backed FP16/cuBLAS | 1 | 1 | 1.015128 | 31523.122614 | 0.274181 | 0.180588 | 0.560272 | 2515001 | PASS |

The cache-backed run emits:

```text
tp_ep_dense_f16_cache layer 2 rows 112 source_bytes 163102720
cache_bytes 302514176 cache_aligned_bytes 302514176 max_temp_bytes 4227072 PASS
```

## Decision

The TP/EP decode loop can now execute dense tensor-core GEMMs from a runtime
FP16 cache arena. The cache-backed path preserves the private FP16 checksum and
slightly improves the 50-step layer-loop metric in this run, but the main
decision is architectural: cache lookup is wired into execution and can replace
per-op weight materialization.

Next work should lift this from the two layer-2 composition tensors to a
descriptor-selected dense execution table usable by every layer in a resident
all-layer loop.
