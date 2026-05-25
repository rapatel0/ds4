---
sprint: 355
title: TP/EP Fused Compressed Pool Norm
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 355 - TP/EP Fused Compressed Pool Norm

## Overview

Sprint 354 showed that fusing only compressed-row RoPE and F16 rounding is too
narrow to matter. The larger state/emit boundary still launches a pool kernel
that writes an unnormalized emitted row, followed by a normalization kernel
that reads the same row back and writes it again.

This sprint tests an opt-in fused pool+normalization kernel. The fused kernel
computes the compressor softmax pool for one slot in shared memory, performs
the row RMS normalization in the same block, and writes only the normalized
emitted row. RoPE and F16 rounding remain separate so the semantic surface is
smaller than a full emit rewrite.

No PP/layer-split work. No MTP. No default semantic change unless V100 evidence
justifies promotion.

## Implementation

1. Add `--true-ds4-compressed-kv-fused-pool-norm-gate`.
2. Add a fused emitted-row pool+normalization CUDA kernel for `head_dim <= 512`.
3. Use the fused kernel for emitted attention compressed rows and ratio-4
   indexer compressed rows when the gate is enabled.
4. Add `--fused-compressed-pool-norm` to the direct profile harness.
5. Report fused pool+norm selection in compressed-KV rows and profiler
   summaries.

## Verification

- Local syntax checks pass.
- V100 `sm_70` build passes.
- V100 direct baseline passes at `32` slots / `256K` / emitted-row
  `position=262143`.
- V100 direct fused pool+norm candidate passes at the same shape.
- Compare selected token, generated decode tok/s, and compressed-KV internal
  timings.

## Definition of Done

- [x] Fused pool+norm gate is implemented and defaults off.
- [x] Direct profiler flag is implemented.
- [x] V100 build passes.
- [x] V100 baseline and fused candidate A/B runs pass.
- [x] Docs/status/temp report are updated with the decision.
- [x] Artifacts are copied into `logs/from-cluster/`.
- [x] Sprint artifacts are committed.

## Outcome

Implemented `--true-ds4-compressed-kv-fused-pool-norm-gate` in
`tools/ds4-v100-tp-ep-full-layer-smoke.cu` and
`--fused-compressed-pool-norm` in `tools/ds4-v100-tp-ep-profile.py`.

The fused kernel uses one CUDA block per slot, computes the compressor pooled
row into shared memory, performs the RMS normalization in the same block, and
writes only the normalized row. RoPE and F16 rounding remain separate.

V100 same-binary A/B, `32` slots / `256K`, emitted-row `position=262143`,
one decode step:

| Variant | Fused layers | Decode tok/s | Decode ms | Pre-EP compressed-KV ms | Compressed-KV sum ms | First token |
|---|---:|---:|---:|---:|---:|---:|
| control | `0` | `81.189757` | `394.138390` | `131.016911` | `130.510967` | `54639` |
| fused pool+norm | `41` | `81.687107` | `391.738686` | `128.201681` | `127.736989` | `54639` |

State/emit timers:

| Variant | Attention state/emit ms | Indexer state/emit ms |
|---|---:|---:|
| control | `24.395683` | `9.045088` |
| fused pool+norm | `22.192089` | `8.495251` |

## Decision

Keep opt-in pending repeat/combination testing. This is the first larger
emitted-row fusion with a clear stage-level win: compressed-KV sum improves by
`2.773978 ms`, attention state/emit improves by `2.203594 ms`, and indexer
state/emit improves by `0.549837 ms`. The topline decode improvement is still
small (`+0.61%`) in a single one-token emitted-row run, so it should not be
promoted as a default yet.

Next sprint should test fused pool+norm together with fused input fill, then
repeat the winning candidate to decide whether to promote the gate for the
direct TP/EP profile path.

Artifacts:

```text
logs/from-cluster/sprint355-fused-compressed-pool-norm/cluster/
```
