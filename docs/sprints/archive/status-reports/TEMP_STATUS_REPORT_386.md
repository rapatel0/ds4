# TEMP Status Report 386

Date: 2026-05-25

## Current Focus

TP/EP only. Sprint 386 targets the real-router compact-MoE route-plan upload
boundary at the production target shape: `32` slots, `256K` context, model
router routing, compact-MoE decode, and VRAM admission enabled.

## Change Tested

The compact route metadata is now packed into one contiguous device allocation
per rank:

- `d_route_indices_by_slot[src]` and `d_route_count_by_slot[src]` point into
  the packed buffer.
- Model-router compact-MoE route planning builds one packed host buffer.
- The runtime performs one H2D copy per destination GPU instead of separate
  per-source route-index/count uploads.
- The legacy single-route index table remains for non-compact compose.

## V100 Results

Build: passed for `tools/ds4-v100-tp-ep-full-layer-smoke` at `sm_70`.

Direct profile:

| Metric | Sprint 385 | Sprint 386 |
|---|---:|---:|
| First token | `54639` | `54639` |
| Generated decode tok/s | `68.544741` | `74.838601` |
| Total decode ms | `466.848362` | `427.586829` |
| Route upload ms | `44.079759` | `10.241125` |
| Router dense/select ms | `33.475698` | `33.466414` |

HTTP `32` request chat profile:

| Metric | Sprint 385 | Sprint 386 |
|---|---:|---:|
| HTTP 200 | `32/32` | `32/32` |
| First token | `83484` | `83484` |
| Client generated tok/s | `42.427324` | `40.302457` |
| Server decode tok/s | `85.792845` | `91.778174` |
| Route upload ms | `38.837019` | `6.796221` |
| Router dense/select ms | `27.758786` | `27.787626` |
| Avg GPU util | `9.051282%` | `8.378049%` |
| VRAM failures | `0` | `0` |
| Min free VRAM | `1754 MiB` | `1756 MiB` |

Artifacts:

- Direct: `/workspace/logs/sprint386-packed-route-upload/direct/`
- HTTP `32`: `/workspace/logs/sprint386-packed-route-upload/http32/`

## Interpretation

Packed route upload is correct and materially reduces the measured upload
timer. It improves server-side decode but does not improve the single-run HTTP
client topline, so the remaining practical bottleneck is no longer H2D copy
count for route tables. The next useful TP/EP work should target router
dense/select and the broader GPU0-heavy HC-current/input staging path, with
repeat HTTP A/B only if the client-side variance matters for a promotion call.
