# Sprint 388: GPU Compact Route Planner Gate

## Overview

Move compact-MoE route-plan construction off the CPU for the real-router
TP/EP path.

Sprints 386 and 387 showed that isolated H2D copy reduction and isolated
router dense replacement reduce local timers, but full serving throughput
does not improve enough unless the broader router/route-plan boundary is
collapsed. The current path still copies selected experts and weights back to
host, constructs offsets/route slots/compact compose maps on CPU, then uploads
route metadata back to every GPU.

## Scope

- Add a default-off `--gpu-route-plan-gate`.
- Replicate the GPU0 router selected/weight tables to all ranks.
- Build expert offsets, route slots, route weights, and compact compose
  indices/counts on-device from the selected experts.
- Preserve deterministic host route ordering by computing each route index
  from prior selected entries, not from nondeterministic atomics.
- Keep the existing host route planner as the default and fallback.
- Validate direct and serving-shaped real-router A/B on gpu-01.

## Out Of Scope

- No PP/layer-split work.
- No router dense kernel promotion.
- No MTP changes.
- No expert TP integration.

## Definition Of Done

- Local checks pass.
- V100 build passes.
- Direct real-router A/B preserves first token and records route upload /
  router timing.
- HTTP `32` request real-router A/B passes and records client/server tok/s if
  direct parity holds.
- Decision is documented: promote, keep diagnostic-only, or reject.

## Risks

- On-device route construction may preserve tokens but shift small floating
  addition order if route indices differ from the CPU planner.
- Copying selected/weight tables to all ranks plus small device kernels may
  be slower than the current packed host upload.
- If the main cost is synchronization around route planning rather than CPU
  construction or H2D metadata, this gate may not improve the topline.

## Outcome

Complete.

Implemented a default-off GPU compact route planner:

- Binary flag: `--gpu-route-plan-gate`
- Profile flag: `--gpu-route-plan`
- Launcher env: `DS4_V100_TP_EP_GPU_ROUTE_PLAN=1`

The gate copies GPU0 router selected/weight tables to each rank, builds
per-rank expert offsets, route slots, route weights, and compact compose maps
on-device, and copies only the small route totals/offset summary back to host
so the TurboMind route count metadata remains correct. Route indices are
computed deterministically from prior selected entries so the route order
matches the CPU planner.

Direct validation at `32` slots / `256K` / `position=262080` /
`1` generated token:

| Metric | Host planner | GPU planner |
|---|---:|---:|
| First token | `54639` | `54639` |
| Generated decode tok/s | `76.179292` | `65.263520` |
| Total decode ms | `420.061664` | `490.319858` |
| Router dense/select ms | `33.591907` | `33.750231` |
| Router D2H ms | `1.295762` | `0.000000` |
| Route-plan/upload ms | `10.190194` | `20.049537` |

Combined direct validation with `--router-cublas-gate`:

| Metric | cuBLAS host planner | cuBLAS + GPU planner |
|---|---:|---:|
| First token | `54639` | `54639` |
| Generated decode tok/s | `79.718036` | `79.338091` |
| Total decode ms | `401.414808` | `403.337155` |
| Router dense/select ms | `18.815270` | `13.603933` |
| Router D2H ms | `1.355017` | `0.000000` |
| Route-plan/upload ms | `10.691757` | `16.046394` |

HTTP `32` request chat validation at `32` slots / `256K` /
`position=262080` / `32` generated tokens per request:

| Metric | Host planner | GPU planner |
|---|---:|---:|
| HTTP 200 | `32/32` | `32/32` |
| First token | `83484` | `83484` |
| Client generated tok/s | `44.579314` | `39.283698` |
| Server decode tok/s | `94.952767` | `87.652515` |
| Router dense/select ms | `27.752540` | `27.804929` |
| Router D2H ms | `1.034725` | `0.000000` |
| Route-plan/upload ms | `6.742906` | `14.474102` |
| Avg GPU util | `9.081081%` | `8.363095%` |
| VRAM failures | `0` | `0` |

Artifacts:

- Direct GPU planner: `/workspace/logs/sprint388-gpu-route-plan/direct/`
- Direct cuBLAS + GPU planner:
  `/workspace/logs/sprint388-gpu-route-plan/direct-router-cublas/`
- HTTP GPU planner: `/workspace/logs/sprint388-gpu-route-plan/http32/`

Decision: reject as a default. Keep the gate only as a diagnostic if useful.
Correctness is good, but the naive GPU route planner replaces one D2H and
packed H2D path with P2P replication, several small kernels, stream
synchronization, and host route-total readback. That costs more than it saves.
The next route-boundary attempt should avoid per-layer host involvement
entirely or fuse route planning with expert dispatch/compose rather than
moving the existing planner structure kernel-for-kernel onto the GPUs.
