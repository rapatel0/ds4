---
sprint: 359
title: TP/EP Direct Multi-Step Pool-Norm Confirmation
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 359 - TP/EP Direct Multi-Step Pool-Norm Confirmation

## Overview

Sprint 358 found a mixed signal for fused compressed pool+norm on the longer
selected-token HTTP path: client wall tok/s improved, aggregate compressed-KV
sum improved, but the parsed scaffold decode proxy regressed. Before doing
more kernel work or promoting any default, that disagreement needs a direct
non-HTTP multi-step A/B.

This sprint runs the direct token-major profile path at the same valid
long-context shape: `32` slots, `256K` context, `position=262112`, and `32`
decode steps. The key comparison is control versus fused pool-norm only. The
combined input-fill + pool-norm gate was already rejected by Sprint 358 and is
not the focus.

No PP/layer-split work. No MTP. No default promotion unless the direct
multi-step result is clearly positive.

## Implementation

1. Run direct token-major control at `32` slots / `256K` /
   `position=262112` / `32` decode steps.
2. Run direct token-major fused pool-norm only at the same shape.
3. Compare generated decode tok/s, compressed-KV stage sum, compressed
   state/emit subtimers, selected token, and finite output-head metadata.
4. Decide whether fused pool-norm is a promotion candidate, remains opt-in, or
   should be rejected.

## Verification

- V100 direct control run returns `rc=0`.
- V100 direct fused pool-norm run returns `rc=0`.
- Both runs preserve output-head finite status.
- Both runs preserve first selected token.
- Artifacts are copied into `logs/from-cluster/`.

## Definition of Done

- [x] Direct control run completes.
- [x] Direct fused pool-norm run completes.
- [x] Results are summarized in this sprint doc.
- [x] `docs/sprints/STATUS.md` and `docs/sprints/VISION.md` are updated.
- [x] A temp status report is written.
- [x] Artifacts are committed.

## Outcome

V100 direct token-major A/B at `32` slots / `256K`, `position=262112`, and
`32` decode steps:

| Variant | Return code | Decode tok/s | Wall tok/s | Emitted layers | Fused pool layers | Compressed-KV sum ms | Pre-EP compressed-KV ms | First token | Finite bad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | `0` | `95.851552` | `74.814127` | `188` | `0` | `3521.094409` | `3533.823377` | `98751` | `0` |
| pool-norm | `0` | `97.619138` | `76.140370` | `188` | `188` | `3458.469603` | `3470.514540` | `98751` | `0` |

Fused pool-norm improved:

- generated decode tok/s by `+1.84%`
- generated wall tok/s by `+1.77%`
- parsed compressed-KV sum by `62.624806 ms`
- pre-EP compressed-KV stage by `63.308837 ms`

The output-head first token stayed fixed at `98751`, and both runs reported
finite output head (`finite_bad=0`).

## Decision

Promote fused compressed pool+norm as the TP/EP serving default.

Changed defaults:

- `tools/ds4-v100-run-appliance.sh` now defaults
  `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM=1`.
- `deploy/v100/ds4-v100-appliance.env.example` now documents the same default.

Kept default-off:

- `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_INPUT_FILL=0`
- `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ROPE_ROUND=0`

Launcher proof:

```text
DS4_V100_SERVE_MODE=tp-ep ... tools/ds4-v100-run-appliance.sh --print-command
```

includes:

```text
--true-ds4-compressed-kv-fused-pool-norm-gate
```

Artifacts:

```text
logs/from-cluster/sprint359-direct-pool-norm-multistep/
```
