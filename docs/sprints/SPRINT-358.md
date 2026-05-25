---
sprint: 358
title: TP/EP Amortized Selected-Token Fusion A/B
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 358 - TP/EP Amortized Selected-Token Fusion A/B

## Overview

Sprint 357 proved that emitted-row selected-token HTTP profiling works and that
fused input-fill + pool-norm reduces parsed compressed-KV stage time. The
client tok/s metric was not useful because each request generated only one
token, so HTTP orchestration dominated the wall measurement.

This sprint runs a longer selected-token HTTP A/B through the same resident
TP/EP serving path. The goal is to decide whether the compressed-fusion gates
produce a serving-visible topline improvement when request overhead is
amortized across more decode steps.

No PP/layer-split work. No MTP. No default promotion unless the longer V100
A/B proves a material serving win.

## Implementation

1. Copy the current profile harness to the V100 workspace.
2. Run selected-token HTTP control at `32` slots / `256K` /
   `position=262143` with more than one generated token/request.
3. Run selected-token HTTP fused input-fill + pool-norm at the same shape.
4. Compare HTTP success, emitted-row coverage, client tok/s, server parsed
   decode/serving metrics when available, and compressed-KV stage totals.
5. Decide whether to promote, keep opt-in, or reject the fused combination.

## Execution Note

The first attempt used `position=262143` with `32` generated tokens. That is
not a valid long-run shape at `ctx=262144`: the first decode succeeds, then
the next position is outside the configured context. The server returned
`tp_ep_decode_failed` with:

```text
tp_runtime_dense_kv_slice_failed slot 7 position is outside configured context
```

The valid amortized A/B therefore starts at `position=262112`, leaving exactly
32 decode positions while still reaching the emitted-row boundary.

## Verification

- Local profile harness syntax check passes.
- V100 profile harness syntax check passes.
- V100 selected-token control run returns all HTTP 200 responses.
- V100 selected-token fused run returns all HTTP 200 responses.
- Both runs exercise emitted compressed rows.
- Artifacts are copied into `logs/from-cluster/`.

## Definition of Done

- [x] V100 longer selected-token HTTP control run completes.
- [x] V100 longer selected-token HTTP fused run completes.
- [x] Results are summarized in this sprint doc.
- [x] `docs/sprints/STATUS.md` and `docs/sprints/VISION.md` are updated.
- [x] A temp status report is written.
- [x] Artifacts are committed.

## Outcome

V100 selected-token HTTP runs at `32` slots / `256K`, `position=262112`,
`32` tokens/request, `32` concurrent requests:

| Variant | HTTP 200 | Client tok/s | Emitted layers | Fused input layers | Fused pool layers | Compressed-KV sum ms | Scaffold decode tok/s | First token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | `32/32` | `71.818394` | `187` | `0` | `0` | `3506.921796` | `98.772310` | `109328` |
| input-fill + pool-norm | `32/32` | `72.297469` | `187` | `671` | `187` | `3509.986423` | `98.505291` | `109328` |
| pool-norm only | `32/32` | `73.052883` | `187` | `0` | `187` | `3474.878472` | `97.552747` | `109328` |

The longer run changes the interpretation from Sprint 357:

- The combined input-fill + pool-norm gates are not promotable. Client tok/s is
  slightly higher, but compressed-KV total and the scaffold decode proxy both
  regress.
- Pool-norm alone remains the better candidate. It improves client wall tok/s
  by `+1.72%` and reduces aggregate compressed-KV sum by `32.043324 ms`, but
  the scaffold decode proxy regresses from `98.772310` to `97.552747` tok/s.

## Decision

Do not promote any compressed-fusion default from this sprint.

Keep fused input-fill diagnostic-only; it is not helping the longer serving
shape. Keep fused pool-norm opt-in and worth revisiting, but only promote it
after a repeated longer HTTP A/B or a direct multi-step decode A/B resolves the
client-vs-scaffold disagreement.

The next sprint should return to implementation rather than wrapper work:
either add a direct multi-step selected-position benchmark that preserves
emitted-row coverage without HTTP overhead, or start fusing the remaining
compressed state/emit boundary that still dominates the compressed-KV stage.

Artifacts:

```text
logs/from-cluster/sprint358-amortized-selected-token-fusions/cluster/
logs/from-cluster/sprint358-amortized-selected-token-fusions-valid/cluster/
```
