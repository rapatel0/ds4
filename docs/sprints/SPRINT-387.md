# Sprint 387: Router Dense cuBLAS Gate

## Overview

Test whether the model-router dense projection should use cuBLAS instead of
the current custom FP32 row kernel.

After Sprint 386, compact route upload is no longer the main real-router
metadata cost. The remaining measured router substage is dense/select:
`27.787626 ms` per all-layer HTTP `32` request decode step. The dense part is
currently launched as one custom block per `(expert, slot)` row, which is
simple but may underuse the V100 compared with a batched SGEMM over the
`256 x 4096 x slots` router shape.

## Scope

- Add a default-off `--router-cublas-gate`.
- Create a persistent GPU0 cuBLAS handle in the shared HC-control runtime.
- When the gate is enabled, compute router logits with `cublasSgemm`:
  `C[experts x slots] = W[experts x hidden] * X[hidden x slots]`.
- Keep the existing top-k/hash selection kernel and route-plan path unchanged.
- Validate direct and HTTP serving-shaped real-router runs on gpu-01.

## Out Of Scope

- No PP/layer-split work.
- No FP16/INT8 router conversion yet.
- No GPU-side route-plan construction.
- No MTP changes.

## Definition Of Done

- Local checks pass.
- V100 build passes.
- Direct real-router A/B preserves first token and records router dense/select
  timing.
- HTTP `32` request real-router check passes and records client/server tok/s.
- Decision is documented: promote, keep diagnostic-only, or reject.

## Risks

- cuBLAS SGEMM may be slower at `N=32` because launch/library overhead can
  dominate the small router matrix.
- The existing custom kernel and cuBLAS may produce slightly different FP32
  accumulation order. First-token parity is the hard gate.
- If the selected expert route changes, the gate must remain diagnostic-only
  until a narrower parity harness is added.

## Outcome

Complete.

Implemented a default-off router dense cuBLAS gate:

- Binary flag: `--router-cublas-gate`
- Profile flag: `--router-cublas`
- Launcher env: `DS4_V100_TP_EP_ROUTER_CUBLAS=1`

The gate creates a persistent GPU0 cuBLAS handle in the shared HC-control
runtime and computes router logits with:

```text
C[experts x slots] = W[experts x hidden] * X[hidden x slots]
```

using `cublasSgemm`. The existing router top-k/hash kernel and compact route
plan remain unchanged.

Same-binary direct A/B at `32` slots / `256K` / `position=262080` /
`1` generated token:

| Metric | Control | cuBLAS |
|---|---:|---:|
| First token | `54639` | `54639` |
| Generated decode tok/s | `76.179292` | `79.718036` |
| Total decode ms | `420.061664` | `401.414808` |
| Router dense/select ms | `33.591907` | `18.815270` |
| Route upload ms | `10.190194` | `10.691757` |

Same-binary HTTP `32` request chat A/B at `32` slots / `256K` /
`position=262080` / `32` generated tokens per request:

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

Decision: keep `--router-cublas-gate` default-off. It proves the custom
router dense kernel is locally replaceable and substantially reduces the
router dense/select timer, but the full HTTP client topline regressed in the
same-binary serving A/B and server decode improved by only about `1%`.
The next promotion candidate should target a broader fusion/scheduling
boundary where the saved router time is not hidden by neighboring stages.
