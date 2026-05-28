# TEMP Status Report 385

Date: 2026-05-25

## Current Focus

TP/EP only. Sprint 385 split and reduced the real-router HC-current
FFN/router bucket.

## V100 Artifacts

```text
/workspace/logs/sprint385-router-stage-split/direct/
/workspace/logs/sprint385-router-stage-split/direct-after-skip-index/
/workspace/logs/sprint385-router-stage-split/http32/
```

## What Changed

- Added substage timing:
  `sum_hc_current_ffn_norm_ms`,
  `sum_hc_current_router_select_ms`,
  `sum_hc_current_router_d2h_ms`,
  `sum_hc_current_route_upload_ms`.
- Parsed those fields into profile `summary.json`.
- Removed unused legacy single-route-index H2D uploads on the compact-MoE
  path.

## Results

Direct real-router `32` slot / `256K` / `1` token:

| Metric | Before | After |
|---|---:|---:|
| first token | 54639 | 54639 |
| generated decode tok/s | 67.804166 | 68.544741 |
| total decode ms | 471.947400 | 466.848362 |
| FFN/router bucket ms | 96.053932 | 79.750084 |
| route upload ms | 60.356962 | 44.079759 |
| router dense/select ms | 33.479468 | 33.475698 |
| router D2H ms | 1.268358 | 1.263678 |
| FFN norm ms | 0.949144 | 0.930949 |

HTTP real-router `32` active requests / `32` generated chat tokens:

| Metric | Sprint 384 | Sprint 385 |
|---|---:|---:|
| HTTP 200 | 32/32 | 32/32 |
| first token | 83484 | 83484 |
| client tok/s | 38.554075 | 42.427324 |
| server decode tok/s | 81.505160 | 85.792845 |
| avg GPU util | 8.547222% | 9.051282% |
| max memory | 32418 MiB | 32418 MiB |
| min free VRAM | 1754 MiB | 1754 MiB |

## Interpretation

The split proved the router bottleneck is not D2H readback or FFN RMSNorm.
The two real targets are:

- route table upload: still `38.837019 ms` in the HTTP `32` case,
- router dense/select: `27.758786 ms` in the same case.

The route-index cleanup is worth keeping because it improves the real-router
serving path without changing tokens or memory footprint.

## Next Best Work

Continue optimizing real-router serving:

1. further reduce route-table upload by collapsing or device-building
   multi-route indices/counts, then
2. attack router dense/select with a lower-precision/tensor-core or distributed
   projection path.
