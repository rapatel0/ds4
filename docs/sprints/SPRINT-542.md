# Sprint 542 - C1 Route Padding Decision

Date: 2026-05-29

## Goal

Decide whether the next C1 step should reduce graph suffix fixed-capacity route
padding, and make future profile summaries retain enough route-shape evidence
to support that decision.

## Starting Evidence

Sprint 540 promoted graph suffix replay after a warmed selected-token gate:

- Eager control:
  `/workspace/s540-warmed-graph-artifacts/none-s540-eager32x64-p262080`
- Graph candidate:
  `/workspace/s540-warmed-graph-artifacts/none-s540-graph32x64-compose-stable-p262080-serverargs-h2180dc1d`
- Request window improved `99.446247s -> 90.181067s`.
- All `32` generated token sequences and decode-step checksums matched eager.

The remaining concern is the graph-only fixed-capacity post-attention route
geometry used to make persistent suffix replay position-stable.

## Route-Padding Audit

The Sprint 540 graph artifact emitted one compact-route stats line per layer.
Those lines are not enough to prove route-shape behavior across all generated
tokens, but they quantify the initial graph envelope:

- `43` route-stat records, `43` layers.
- Actual routes per layer: `192` (`32` slots x top-6).
- Fixed graph geometry: `192` rows per rank, or `1536` routed rows/layer.
- Actual route rows over padded rows at full cap: `12.5%`.
- Max per-rank route pressure:
  - min `32`
  - p50 `64`
  - p95 `96`
  - max `132`
- Rank count with nonzero routes:
  - min `2`
  - p50 `5`
  - max `7`
- Compact return bytes were `3,145,728` vs all-destination bytes
  `4,194,304` per logged layer (`75.0%`).

Static cap sweeps over those logged shapes would overflow:

| Per-rank cap | Logged overflow records | Actual/padded rows |
|---:|---:|---:|
| `32` | `37/43` | `75.0%` |
| `64` | `8/43` | `37.5%` |
| `96` | `1/43` | `25.0%` |
| `128` | `1/43` | `18.75%` |
| `160` | `0/43` | `15.0%` |
| `192` | `0/43` | `12.5%` |

## Prior Static-Cap Evidence

Do not resurrect the old static-cap path as a promotion shortcut:

- Sprint 434 rejected static rank caps: overflow-free cap runs changed final
  checksums.
- Sprint 436 rejected executor-only static caps: token changed despite keeping
  transfer/compose envelope full.
- Sprint 437 rejected compose-only static caps: cap16 improved the proxy but
  changed the selected token.

Those results show the current TurboMind grouped-GEMM / compact-compose host
shapes are semantically shape-sensitive. A safe reduction must keep the
host-visible graph shape fixed and move inactivity/masking inside the
full-shape executor/compose implementation.

## Implementation

Updated `tools/ds4-v100-tp-ep-profile.py` so future summaries aggregate
`tp_ep_compact_moe_route_stats` instead of retaining only the last logged line.

New summary keys include:

- `compact_moe_route_stat_records`
- `compact_moe_layers_seen`
- `compact_moe_unique_route_shapes`
- `compact_moe_shape_repeat_ratio`
- `compact_moe_total_routes_min/max`
- `compact_moe_max_rank_routes_min/p50/p95/max`
- `compact_moe_nonzero_rank_count_min/p50/max`
- `compact_moe_compact_over_all_dest_pct`

No engine path, launcher default, or runtime flag changed.

## Validation

- Local parser fixture over two sample `tp_ep_compact_moe_route_stats` lines:
  PASS.
- `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py`: PASS.

No appliance rebuild was required because this sprint changed only profile
summary tooling and docs.

## Decision

Keep Sprint 540 graph suffix replay promoted.

Do not tune C1 by lowering static route caps. The next performance-code sprint
should either:

1. implement a full-shape device-masked post-attention routed executor/compose
   path that preserves graph-visible launch/copy shapes while skipping inactive
   rows internally, or
2. move to A5/A6 fusion if the masked full-shape path is too large for a
   single sprint.

Future warmed graph runs should use the new aggregate route summaries and still
isolate startup, include meaningful warmup, control stochastic settings, and
report request-window metrics.
