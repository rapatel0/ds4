---
sprint: 364
title: TP/EP Direct Compressed Input Fill
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 364 - TP/EP Direct Compressed Input Fill

## Overview

Sprint 363 showed that wider scalar emitted-row fusion is not the next
material lever. The remaining compressed-KV cost is upstream: before the
compressor/indexer dense projections, every rank receives a full f32
attention-normalized current vector, then separate kernels convert that vector
into half inputs for the attention compressor and ratio-4 indexer dense ops.

This sprint tests an opt-in direct-fill path:

```text
current attn_normed on rank 0 -> per-rank d_current_full copy -> half input fill
```

becomes:

```text
current attn_normed on rank 0 -> half input fill kernels directly read source
```

The expected benefit is reduced staging traffic and one less full f32
copy/read path before compressed dense projection. The risk is that remote
peer reads from rank-0 memory may underperform the explicit peer copy, as the
older full-current peer-gather experiment did.

This is TP/EP-only. No PP/layer-split work. No MTP.

## Implementation

1. Add `--true-ds4-compressed-kv-direct-input-fill-gate`.
2. In `run_true_ds4_compressed_kv_projection_gate`, when the gate is enabled:
   - skip the per-rank `d_current_full` copy from `hc->d_attn_normed`,
   - feed the existing half-fill kernels from `hc->d_attn_normed` directly,
   - apply the same direct source to the later ratio-4 indexer input fill.
3. Expose the gate through:
   - `tools/ds4-v100-run-appliance.sh`,
   - `deploy/v100/ds4-v100-appliance.env.example`,
   - `tools/ds4-v100-tp-ep-profile.py`.
4. Keep it default-off unless V100 direct A/B shows a material win.

## Verification

- Local syntax checks pass.
- V100 `sm_70` build passes.
- V100 direct 32-step A/B at `32` slots / `256K` / `position=262112`:
  - control: production pool+norm default,
  - candidate: direct compressed input fill plus production pool+norm default.
- Both variants preserve finite output head and first selected token.
- Compare generated decode tok/s, wall tok/s, compressed-KV sum, and
  `attn_input_fill_ms` / `indexer_input_fill_ms`.

## Definition of Done

- [x] The gate compiles for `sm_70`.
- [x] The gate is reachable from direct smoke, launcher, and profile harness.
- [x] V100 direct A/B passes correctness invariants.
- [x] Results are summarized in this sprint doc, `STATUS.md`, and `VISION.md`.
- [x] A temp status report is written.
- [x] Artifacts are committed.

## Outcome

Implemented the direct compressed input fill gate:

- `--true-ds4-compressed-kv-direct-input-fill-gate`,
- `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DIRECT_INPUT_FILL=1`,
- `tools/ds4-v100-tp-ep-profile.py --direct-compressed-input-fill`.

The gate skips the per-rank `d_current_full` staging copy in the compressed
projection path and feeds existing half-fill kernels directly from
`hc->d_attn_normed`.

V100 build passed:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

One-token emitted-row same-build A/B at `32` slots / `256K` /
`position=262143`:

| Variant | First token | Finite bad | Decode tok/s | Wall tok/s | Compressed-KV sum | Attn input fill | Indexer input fill |
|---|---:|---:|---:|---:|---:|---:|---:|
| pool+norm default | 54639 | 0 | 81.339455 | 21.006518 | 126.724613 ms | 12.587939 ms | 3.754212 ms |
| direct input fill | 54639 | 0 | 60.573442 | 19.295478 | 260.365841 ms | 84.142732 ms | 65.145245 ms |

## Decision

Reject direct remote compressed input fill. It is legal and correct, but
peer-read conversion is much slower than the explicit per-rank current copy
plus local half-fill kernels.

Per the stop-on-material-no-improvement rule, this sprint stopped at the
one-token emitted-row A/B rather than spending additional cluster time on a
full 32-step run. The regression is already large and localized:

```text
compressed-KV sum: 126.724613 -> 260.365841 ms
attn input fill:  12.587939  -> 84.142732 ms
index input fill: 3.754212   -> 65.145245 ms
```

The next optimization should preserve local per-rank reads. Better candidates:

1. reduce the number of local half-fill launches without remote reads,
2. concatenate/merge compressor dense projections where memory allows, or
3. improve the dense projection kernels themselves.

Artifacts:

```text
logs/from-cluster/sprint364-direct-compressed-input-fill-smoke/
```
