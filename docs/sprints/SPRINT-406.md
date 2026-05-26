# Sprint 406: Compact Compressed-KV State Layout

Date: 2026-05-26

## Overview

Sprint 405 proved that lazy diagnostic output-head residency is not enough to
admit HC-current NCCL at the target `32` slot / `256K` TP/EP shape. The NCCL
candidate now fails later, before first-token completion, with CUDA OOM while
allocating attention compressed-KV state for layer 5.

The next low-risk reclaim is the attention compressed-KV state layout. The
current allocation uses the worst-case geometry for every ratio layer:

```text
slots * kCompStateRowsMax(128) * kCompWidthMax(1024) * sizeof(float)
```

That is correct but wasteful. DS4 layers need exact geometries:

| Layer ratio | Needed rows | Needed width | Current rows | Current width |
|---:|---:|---:|---:|---:|
| 4 | 8 | 1024 | 128 | 1024 |
| 128 | 128 | 512 | 128 | 1024 |

This sprint changes the resident attention compressed-KV state allocation and
kernel strides to use those exact dimensions.

## Constraints

- TP/EP only. No PP/layer-split work.
- No dtype change and no lossy quantization.
- Default behavior should improve because the old layout was a padded memory
  layout, not an accuracy requirement.
- Preserve first-token correctness for the non-NCCL target shape.
- Re-test HC-current NCCL + lazy output-head at `32` slots / `256K`.

## Implementation

Files:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`

Add helper functions:

- `attn_comp_state_rows_for_ratio(ratio)`
- `attn_comp_state_width_for_ratio(ratio)`

Use them for:

- attention compressed-KV state allocation
- attention compressed-KV state zeroing
- `compressor_store_slots_kernel`
- `compressor_pool_*_emit_slots_kernel`
- ratio-4 shift
- compressed reference-diff readback

Indexer compressed state already uses its compact `8 x 256` layout and should
not be changed.

## Expected Memory Impact

At `32` slots:

- ratio-4 attention state per layer/rank drops from about `32 MiB` to `2 MiB`
  across KV + score.
- ratio-128 attention state per layer/rank drops from about `32 MiB` to
  `16 MiB` across KV + score.
- Across all ratio layers, this should reclaim hundreds of MiB per GPU, enough
  to retest the HC-current NCCL admission failure from Sprint 405.

## Validation

Local:

```text
git diff --check
```

V100:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Run direct target-shape probes:

1. non-NCCL control, `32` slots / `256K`, lazy output-head on
2. HC-current NCCL + lazy output-head, `32` slots / `256K`

Record:

- return code
- first token
- generated and continuation decode tok/s
- `after_hc_controls` min free
- lazy output-head min free when reached
- OOM/failure site if still failing

Artifacts:

- `logs/from-cluster/sprint406-compact-kv-state/lazy-control/`
- `logs/from-cluster/sprint406-compact-kv-state/lazy-hc-nccl/`

Results:

| Case | Return | First token | Generated decode tok/s | Continuation decode tok/s | Min free VRAM | Key checkpoint |
|---|---:|---:|---:|---:|---:|---|
| lazy control | 0 | 54639 | 78.447208 | 77.220402 | 1018 MiB | `after_hc_controls=1880 MiB`, `after_lazy_output_head=1018 MiB` |
| lazy + HC-current NCCL | 0 | 54639 | 89.952595 | 100.096637 | 386 MiB | `after_hc_controls=1248 MiB`, `after_lazy_output_head=386 MiB` |

Sprint 405 comparison:

- Non-NCCL lazy output-head peak improved from `68 MiB` free to `1018 MiB`
  free.
- HC-current NCCL moved from CUDA OOM at compressed-KV state allocation on
  layer 5 to full first-token completion with token `54639`.

The conservative `1536 MiB` NCCL reserve still fails after lazy output-head:
all eight GPUs are below threshold, with GPU0 at `386 MiB`. This means the
NCCL path is now operational at the target shape, but it is not production
admitted with the current reserve.

## Definition of Done

- Exact attention compressed-KV state dimensions are implemented.
- Local checks pass.
- V100 build passes.
- Target non-NCCL control preserves first token.
- Target HC-current NCCL is rerun and recorded.
- Sprint doc, status, vision, temporary report, and cluster artifacts are
  committed explicitly.

## Decision Gate

Promote the compact state layout unless it changes first token or introduces a
new correctness failure. If HC-current NCCL still fails, use the updated VRAM
checkpoint and failure site to choose the next memory reclaim target.

## Decision

Promote the exact attention compressed-KV state layout.

This is a semantic-preserving memory layout correction: ratio-4 layers use
`8 x 1024` state and ratio-128 layers use `128 x 512` state instead of the old
universal `128 x 1024` padding. It preserves first-token correctness and
turns the target-shape HC-current NCCL run from an OOM into a completed
diagnostic.

Do not promote HC-current NCCL as production default yet. The next sprint
should make the output-head path lazy/resident-compatible for HTTP serving and
continue reclaiming peak VRAM so the target shape clears the `1536 MiB` NCCL
reserve.
