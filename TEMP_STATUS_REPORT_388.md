# TEMP Status Report 388

Date: 2026-05-25

## Current Focus

TP/EP only. Sprint 388 tested a broader real-router boundary: moving compact
route-plan construction from CPU to GPU.

## Change Tested

Added a default-off GPU route planner:

- Binary flag: `--gpu-route-plan-gate`
- Profile flag: `--gpu-route-plan`
- Launcher env: `DS4_V100_TP_EP_GPU_ROUTE_PLAN=1`

The gate copies GPU0 router selected/weight tables to each rank, builds expert
offsets, route slots, route weights, and compact compose maps on-device, then
copies only small route totals/offset summaries back to host for TurboMind
route count metadata.

## V100 Results

Build: passed for `tools/ds4-v100-tp-ep-full-layer-smoke` at `sm_70`.

Direct:

| Metric | Host planner | GPU planner |
|---|---:|---:|
| First token | `54639` | `54639` |
| Generated decode tok/s | `76.179292` | `65.263520` |
| Total decode ms | `420.061664` | `490.319858` |
| Router D2H ms | `1.295762` | `0.000000` |
| Route-plan/upload ms | `10.190194` | `20.049537` |

Direct with router cuBLAS:

| Metric | cuBLAS host planner | cuBLAS + GPU planner |
|---|---:|---:|
| First token | `54639` | `54639` |
| Generated decode tok/s | `79.718036` | `79.338091` |
| Total decode ms | `401.414808` | `403.337155` |
| Router dense/select ms | `18.815270` | `13.603933` |
| Route-plan/upload ms | `10.691757` | `16.046394` |

HTTP `32` request chat:

| Metric | Host planner | GPU planner |
|---|---:|---:|
| HTTP 200 | `32/32` | `32/32` |
| First token | `83484` | `83484` |
| Client generated tok/s | `44.579314` | `39.283698` |
| Server decode tok/s | `94.952767` | `87.652515` |
| Router D2H ms | `1.034725` | `0.000000` |
| Route-plan/upload ms | `6.742906` | `14.474102` |
| Avg GPU util | `9.081081%` | `8.363095%` |
| VRAM failures | `0` | `0` |

Artifacts:

- Direct GPU planner: `/workspace/logs/sprint388-gpu-route-plan/direct/`
- Direct cuBLAS + GPU planner:
  `/workspace/logs/sprint388-gpu-route-plan/direct-router-cublas/`
- HTTP GPU planner: `/workspace/logs/sprint388-gpu-route-plan/http32/`

## Interpretation

Correctness is clean, but performance is worse. The gate removes router D2H,
but the replacement cost is higher: selected/weight P2P replication to all
ranks, several tiny route-planning kernels, synchronization, and route-total
readback. Keep the gate diagnostic-only/rejected. The next viable route-boundary
attempt should fuse route planning with expert dispatch/compose or eliminate
per-layer host interaction more completely.
