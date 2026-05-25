# TEMP Status Report 384

Date: 2026-05-25

## Current Focus

TP/EP only. Sprint 384 measured the quality-preserving serving path with
model-router routes and compact MoE decode enabled.

## V100 Artifact

```text
/workspace/logs/sprint384-real-router-matrix/
```

## Real-Router Matrix

`32` configured slots, `256K` context, `position=262080`, `32` generated chat
tokens/request, GPU sampling at `500 ms`, VRAM reserve `64 MiB`.

| Active requests | HTTP 200 | Client tok/s | Server decode tok/s | Avg GPU util | Max GPU util | Max memory | Min free VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 1.254734 | 80.934514 | 8.657051% | 42.0% | 32418 MiB | 1754 MiB |
| 4 | 4/4 | 4.987325 | 81.231383 | 8.475000% | 41.0% | 32418 MiB | 1754 MiB |
| 8 | 8/8 | 9.654914 | 79.547736 | 8.400000% | 42.0% | 32418 MiB | 1754 MiB |
| 16 | 16/16 | 18.535369 | 76.816196 | 7.916667% | 41.0% | 32418 MiB | 1754 MiB |
| 32 | 32/32 | 38.554075 | 81.505160 | 8.547222% | 40.0% | 32418 MiB | 1754 MiB |

All cases had `vram_failures=0`.

## Interpretation

This is the baseline that matters for DS4 intelligence. The Sprint 383
launcher-default matrix was useful, but it did not enable model-router routes.

Compared with Sprint 383:

- `32`-request client tok/s drops from `43.853691` to `38.554075`.
- Server decode drops from roughly `92-98` tok/s to roughly `77-82` tok/s.
- Average GPU utilization remains below `9%`.
- Min free VRAM remains stable at `1754 MiB`.
- GPU0 remains the busiest card.
- The additional real-router cost appears in the HC-current FFN/router stage,
  around `85-88 ms` per all-layer decode step.

## Next Best Work

Optimize the real-router path first. The next sprint should target GPU0-heavy
router / HC-current staging and launch fragmentation, then A/B against this
Sprint 384 matrix.
