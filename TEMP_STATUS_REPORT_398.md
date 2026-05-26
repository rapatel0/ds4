# TEMP Status Report 398

Date: 2026-05-26

## Current Focus

TP/EP only. No PP/layer-split work.

This pass tested whether the HC-current post-router fill/pack tail could be
made faster by replacing per-rank peer copies, dense input fills, half input
fills, and route packing with one fused kernel per rank.

## Implemented

- Added `--tp-hc-current-input-fused-fill-pack-gate`.
- Added launcher env:
  `DS4_V100_TP_EP_HC_CURRENT_INPUT_FUSED_FILL_PACK`.
- Added profile harness flag:
  `--hc-current-fused-fill-pack`.
- Added a default-off fused kernel for the production unscaled HC-current path.
- Preserved dense/current and routed/ffn-normed input semantics separately.
- Left reference scaled route-pack, graph-event, and peer-gather paths on the
  existing implementation.

## Results

Build on V100: PASS.

Direct same-binary A/B at `32` slots / `256K` / `position=262080` /
model-router compact-MoE / route-plan async upload:

| Mode | First token | Generated decode tok/s | Continuation decode tok/s | HC fill/pack ms | HC-current ms | Total decode ms | VRAM failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 54639 | 87.759480 | 97.163622 | 28.140415 | 599.389235 | 729.265944 | 0 |
| fused fill/pack | 54639 | 64.310075 | 67.874579 | 320.439853 | 869.514336 | 995.178434 | 0 |

Artifacts:

- `logs/from-cluster/sprint398-hc-fused-fill-pack/direct-control/`
- `logs/from-cluster/sprint398-hc-fused-fill-pack/direct-candidate/`

## Decision

Do not promote. Keep the fused fill/pack gate default-off as a diagnostic.

The direct remote-load fusion is correct but materially slower. The current
explicit peer-copy plus local-read staging is better for this boundary on the
V100 topology. Future HC-current work should either preserve local staging or
fuse farther downstream so staged data is consumed locally without introducing
remote load pressure.
