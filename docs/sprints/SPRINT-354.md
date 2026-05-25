---
sprint: 354
title: TP/EP Fused Compressed RoPE Round
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 354 - TP/EP Fused Compressed RoPE Round

## Overview

Sprint 352 showed attention/indexer state+emit is a larger remaining
compressed-KV cost than typed stores or input fill. Sprint 353's fused input
fill was correct but too small to promote.

This sprint tests a narrow state/emit fusion: when compressed-row RoPE is
enabled, combine the emitted-row RoPE pass and the following F16-rounding pass
into one kernel. This removes one kernel launch and one row pass for every
emitted attention compressed row and ratio-4 indexer row.

No PP/layer-split work. No MTP. No default semantic change unless V100 evidence
justifies promotion.

## Implementation

1. Add `--true-ds4-compressed-kv-fused-rope-round-gate`.
2. Add a fused compressed-row RoPE+F16-round CUDA kernel.
3. Use the fused kernel only when compressed-row RoPE is active.
4. Add `--fused-compressed-rope-round` to the direct profile harness.
5. Report fused RoPE+round selection in compressed-KV rows and profiler
   summaries.

## Verification

- Local syntax checks pass.
- V100 `sm_70` build passes.
- V100 direct baseline passes at `32` slots / `256K` / emitted-row
  `position=262143`.
- V100 direct fused RoPE+round candidate passes at the same shape.
- Compare generated decode tok/s and compressed-KV internal timings.

## Definition of Done

- [x] Fused RoPE+round gate is implemented and defaults off.
- [x] Direct profiler flag is implemented.
- [x] V100 build passes.
- [x] V100 baseline and fused candidate A/B runs pass.
- [x] Docs/status/temp report are updated with the decision.
- [x] Artifacts are copied into `logs/from-cluster/`.
- [x] Sprint artifacts are committed.

## Outcome

Implemented `--true-ds4-compressed-kv-fused-rope-round-gate` in
`tools/ds4-v100-tp-ep-full-layer-smoke.cu` and
`--fused-compressed-rope-round` in `tools/ds4-v100-tp-ep-profile.py`.

The fused kernel applies compressed-row RoPE and the following F16 rounding in
one pass when RoPE is active. It is selected only for emitted compressed rows.

V100 same-binary A/B, `32` slots / `256K`, emitted-row `position=262143`,
one decode step:

| Variant | Fused layers | Decode tok/s | Decode ms | Pre-EP compressed-KV ms | Compressed-KV sum ms | First token |
|---|---:|---:|---:|---:|---:|---:|
| control | `0` | `79.810167` | `400.951423` | `130.520098` | `130.000098` | `54639` |
| fused RoPE+round | `41` | `79.344207` | `403.306067` | `130.382524` | `129.883234` | `54639` |

State/emit timers:

| Variant | Attention state/emit ms | Indexer state/emit ms |
|---|---:|---:|
| control | `24.680003` | `9.003129` |
| fused RoPE+round | `24.352357` | `8.880899` |

## Decision

Reject promotion. The fused kernel is correct and slightly reduces state/emit
subtimers, but total decode regresses within noise and the compressed-KV stage
is effectively flat. Removing this individual launch/pass is not the material
lever.

Keep the gate opt-in for diagnostics only. The next state/emit work should
target larger boundaries, especially pooling+normalization or store+pooling,
rather than only fusing RoPE with rounding.

Artifacts:

```text
logs/from-cluster/sprint354-fused-compressed-rope-round/cluster/
```
