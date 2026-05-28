# TEMP Status Report 061 - HC Current Stream Sync

Date: 2026-05-25

## Current Focus

TP/EP serving optimization. Sprint 347 identified HC-current/input staging as
the dominant direct timer, and Sprint 348 rejected naive all-rank peer gather.

## Change

Added and promoted:

```text
--tp-hc-current-input-stream-sync-gate
DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC=1
tools/ds4-v100-tp-ep-profile.py --hc-current-stream-sync
```

The gate keeps the existing layout but runs central GPU0 HC-current control
kernels on rank 0's stream and replaces selected GPU0 device-wide barriers
with stream-scoped synchronization.

## V100 Results

Direct 32-slot / 256K / 2-step A/B:

| Case | Generated tok/s decode | Continuation tok/s decode | Sum decode ms | HC-current ms |
|---|---:|---:|---:|---:|
| Control | `74.841520` | `91.326928` | `855.140307` | `711.608991` |
| Stream sync | `81.190638` | `93.199784` | `788.268223` | `647.492171` |

HTTP 32-request / 32-slot / 256K / 2-token A/B:

| Case | HTTP 200 | Server generated tok/s | Server generated tok/s decode | Server continuation tok/s |
|---|---:|---:|---:|---:|
| Control | `32/32` | `82.573137` | `97.500352` | `82.452144` |
| Stream sync | `32/32` | `83.813937` | `98.859925` | `84.429119` |

## Decision

Promoted as default in `tools/ds4-v100-run-appliance.sh`:

```text
DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC=1
```

This is not a large topline change, but it is directionally correct and
reduces the measured hot-stage timer. Next target: fuse or bypass the HC
control/fill chain itself.

## Artifacts

```text
logs/from-cluster/sprint349-hc-stream-sync/cluster/
```
