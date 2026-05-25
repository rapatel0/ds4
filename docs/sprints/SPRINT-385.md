# Sprint 385: Real-Router Stage Split

## Overview

Split the broad real-router `sum_hc_current_ffn_router_ms` timer into
actionable substage timers.

Sprint 384 established the quality-preserving serving baseline:
`--model-router-routes --compact-moe-decode` at `32` slots / `256K` reaches
about `81.5` server decode tok/s and spends roughly `85-88 ms` per all-layer
decode step in the HC-current FFN/router bucket. That bucket is too broad to
optimize safely.

## Scope

- Add substage timers for the real-router portion of
  `run_shared_hc_current_input`:
  - FFN RMSNorm,
  - router dense + top-k select,
  - selected/weight D2H readback,
  - CPU route-plan construction + H2D route-table upload.
- Emit the new fields on `tp_ep_token_major_item` and
  `tp_ep_token_major_scaffold`.
- Parse the new fields in `tools/ds4-v100-tp-ep-profile.py`.
- Remove the unused legacy single-route index H2D uploads when
  `--compact-moe-decode` is active, because the compact compose kernel uses
  multi-route indices/counts instead.
- Run V100 validation on the real-router path and record the dominant substage.

## Out Of Scope

- No PP/layer-split work.
- No kernel replacement yet.
- No MTP work.
- No E5M2 promotion.

## Definition Of Done

- Local syntax checks pass.
- V100 build passes for `tools/ds4-v100-tp-ep-full-layer-smoke`.
- A real-router V100 profile emits and parses:
  `sum_hc_current_ffn_norm_ms`,
  `sum_hc_current_router_select_ms`,
  `sum_hc_current_router_d2h_ms`, and
  `sum_hc_current_route_upload_ms`.
- The sprint outcome names the next optimization target based on measured
  substage cost.

## Risks

- Host-side timing includes synchronization side effects and launch latency;
  that is acceptable here because those are exactly the suspected bottlenecks.
- The split may show that router dense/select dominates, in which case the
  next sprint should be a real kernel/topology change rather than route-plan
  upload cleanup.

## Outcome

Implemented and validated.

Code changes:

- Added real-router substage timers:
  `ffn_norm`, `router_select`, `router_d2h`, and `route_upload`.
- Emitted the new fields in both per-layer `tp_ep_token_major_item` rows and
  aggregate `tp_ep_token_major_scaffold` rows.
- Parsed the new aggregate fields in
  `tools/ds4-v100-tp-ep-profile.py`.
- Removed legacy single-route-index table construction/upload when
  `--compact-moe-decode` is enabled, because compact compose consumes
  `route_indices_by_slot` and `route_count_by_slot`, not the single-route
  `route_index_by_slot`.

V100 artifacts:

```text
/workspace/logs/sprint385-router-stage-split/direct/
/workspace/logs/sprint385-router-stage-split/direct-after-skip-index/
/workspace/logs/sprint385-router-stage-split/http32/
```

Build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 \
  tools/ds4-v100-tp-ep-full-layer-smoke
```

passed on gpu-01.

Direct real-router profile, `32` slots / `256K` / `position=262080` /
`1` generated token:

| Metric | Before skip | After skip |
|---|---:|---:|
| first token | 54639 | 54639 |
| generated decode tok/s | 67.804166 | 68.544741 |
| total decode ms | 471.947400 | 466.848362 |
| HC-current input ms | 409.436758 | 403.145143 |
| FFN/router bucket ms | 96.053932 | 79.750084 |
| FFN norm ms | 0.949144 | 0.930949 |
| router dense/select ms | 33.479468 | 33.475698 |
| router D2H ms | 1.268358 | 1.263678 |
| route upload ms | 60.356962 | 44.079759 |

HTTP real-router `32` active requests / `32` generated chat tokens:

| Metric | Sprint 384 baseline | Sprint 385 |
|---|---:|---:|
| HTTP 200 | 32/32 | 32/32 |
| first token | 83484 | 83484 |
| client tok/s | 38.554075 | 42.427324 |
| server decode tok/s | 81.505160 | 85.792845 |
| avg GPU util | 8.547222% | 9.051282% |
| max memory | 32418 MiB | 32418 MiB |
| min free VRAM | 1754 MiB | 1754 MiB |
| VRAM failures | 0 | 0 |
| FFN/router bucket ms | 84.827882 | 68.406763 |
| route upload ms | not split | 38.837019 |
| router dense/select ms | not split | 27.758786 |

## Decision

Keep the compact-path legacy route-index upload removal.

It preserves first-token parity and improves the real-router serving case by
about `10.0%` client tok/s and `5.3%` server decode tok/s at the target
`32` request / `32` slot / `256K` shape. It also proves the broad router
bucket is mainly route upload plus router dense/select, not D2H or RMSNorm.

The next sprint should continue on this same path:

1. reduce route upload further by avoiding repeated tiny H2D copies for
   multi-route indices/counts, or
2. replace the GPU0 F32 router dense/select path with a more appropriate
   low-precision/tensor-core implementation or distributed router projection.
