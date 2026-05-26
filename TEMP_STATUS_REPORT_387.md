# TEMP Status Report 387

Date: 2026-05-25

## Current Focus

TP/EP only. Sprint 387 tested whether the remaining real-router
dense/select cost can be reduced by replacing the custom FP32 router dense
kernel with cuBLAS SGEMM.

## Change Tested

Added a default-off router cuBLAS path:

- Binary flag: `--router-cublas-gate`
- Profile flag: `--router-cublas`
- Launcher env: `DS4_V100_TP_EP_ROUTER_CUBLAS=1`

The gate computes router logits with persistent GPU0 cuBLAS:

```text
C[experts x slots] = W[experts x hidden] * X[hidden x slots]
```

The top-k/hash selection kernel, D2H route readback, and packed compact route
upload remain unchanged.

## V100 Results

Build: passed for `tools/ds4-v100-tp-ep-full-layer-smoke` at `sm_70`.

Same-binary direct A/B:

| Metric | Control | cuBLAS |
|---|---:|---:|
| First token | `54639` | `54639` |
| Generated decode tok/s | `76.179292` | `79.718036` |
| Total decode ms | `420.061664` | `401.414808` |
| Router dense/select ms | `33.591907` | `18.815270` |
| Route upload ms | `10.190194` | `10.691757` |

Same-binary HTTP `32` request chat A/B:

| Metric | Control | cuBLAS |
|---|---:|---:|
| HTTP 200 | `32/32` | `32/32` |
| First token | `83484` | `83484` |
| Client generated tok/s | `44.579314` | `41.769369` |
| Server decode tok/s | `94.952767` | `95.944290` |
| Router dense/select ms | `27.752540` | `4.959189` |
| Route upload ms | `6.742906` | `7.002523` |
| Avg GPU util | `9.081081%` | `8.212500%` |
| VRAM failures | `0` | `0` |

Artifacts:

- Direct control: `/workspace/logs/sprint387-router-cublas/direct-control/`
- Direct cuBLAS: `/workspace/logs/sprint387-router-cublas/direct/`
- HTTP control: `/workspace/logs/sprint387-router-cublas/http32-control/`
- HTTP cuBLAS: `/workspace/logs/sprint387-router-cublas/http32/`

## Interpretation

cuBLAS is a strong local router dense replacement, but it is not a serving
promotion yet. The direct path improves, and the HTTP router dense/select
timer falls sharply, but the full HTTP client topline regresses and server
decode improves by only about `1%`. Keep the gate diagnostic-only. The next
performance sprint should fuse or reschedule a broader boundary so that
router savings are not hidden by HC-current/input staging and neighboring
attention/KV work.
