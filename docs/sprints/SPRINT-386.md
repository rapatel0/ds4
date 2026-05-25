# Sprint 386: Packed Compact Route Upload

## Overview

Collapse compact-MoE route-plan uploads from many tiny per-source tables into
one packed H2D upload per destination GPU.

Sprint 385 proved route upload remains the largest real-router substage after
removing the unused legacy single-route index table. The HTTP `32` request
case still spends `38.837019 ms` per all-layer decode step uploading compact
route metadata.

## Scope

- Add a contiguous compact route-plan buffer per rank.
- Point `d_route_indices_by_slot[src]` and `d_route_count_by_slot[src]` into
  that packed buffer.
- In model-router compact-MoE route planning, build one packed host buffer and
  copy it once per destination rank instead of copying every source table
  separately.
- Preserve the legacy single-route index table for non-compact compose.
- Validate direct and serving-shaped real-router runs on gpu-01.

## Out Of Scope

- No PP/layer-split work.
- No router dense/select kernel replacement.
- No MTP changes.
- No E5M2 promotion.

## Definition Of Done

- Local checks pass.
- V100 build passes.
- Direct real-router profile preserves first token and reduces or explains
  route-upload timing.
- HTTP `32` request real-router check passes and records client/server tok/s.

## Risks

- Device pointers now alias a packed allocation; cleanup must avoid freeing
  sub-pointers separately.
- If route upload is dominated by `cudaSetDevice` or host-side vector
  construction rather than H2D copy count, the packed upload may not move the
  total much. That result is still useful.

## Outcome

Complete.

V100 build passed for `tools/ds4-v100-tp-ep-full-layer-smoke`.

Direct real-router validation at `32` slots / `256K` /
`position=262080` / `1` generated token passed with first token `54639`,
matching Sprint 385. Packed compact route upload reduced the direct
all-layer route-upload timer from `44.079759 ms` to `10.241125 ms`.
Generated decode improved from `68.544741` to `74.838601` tok/s, while
router dense/select stayed flat (`33.475698 ms` to `33.466414 ms`).

HTTP serving validation at `32` concurrent chat requests / `32` slots /
`256K` / `position=262080` / `32` generated tokens per request passed with
`32/32` HTTP 200 responses and first token `83484`, matching Sprint 385.
The server-side decode rate improved from `85.792845` to `91.778174` tok/s,
and route upload dropped from `38.837019 ms` to `6.796221 ms`. Client-side
generated throughput moved from `42.427324` to `40.302457` tok/s, so the
result is a clear server decode/stage win but not a full HTTP client topline
win in this single run. VRAM admission stayed clean with `vram_failures=0`,
`vram_min_free_mib=1756`, and `vram_max_used_mib=30737`.

Artifacts:

- Direct: `/workspace/logs/sprint386-packed-route-upload/direct/`
- HTTP `32`: `/workspace/logs/sprint386-packed-route-upload/http32/`

Decision: keep the packed compact route plan in the compact-MoE path. It
removes a measured H2D-copy-count bottleneck without changing tokens, and it
makes the remaining real-router boundary primarily router dense/select plus
the broader HC-current/input staging path.
