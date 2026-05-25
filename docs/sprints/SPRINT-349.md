---
sprint: 349
title: TP/EP HC Current Stream-Scoped Barriers
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 349 - TP/EP HC Current Stream-Scoped Barriers

## Overview

Sprint 348 showed that naive all-rank peer gathering is not the right
optimization for `run_shared_hc_current_input`. The next narrower hypothesis
is synchronization scope. The HC-current path launches central control work on
GPU0 and uses several `cudaDeviceSynchronize()` barriers. That can wait for
unrelated GPU0 work and contributes to the low-utilization launch/sync
fragmentation seen in Sprint 347.

This sprint adds an opt-in stream-scoped version of those central GPU0
barriers. It keeps the existing data layout and semantics but launches GPU0
control kernels on rank 0's stream and uses stream synchronization where a
device-wide barrier is not required.

No PP/layer-split work. No MTP.

## Implementation

1. Add `--tp-hc-current-input-stream-sync-gate`.
2. In `run_shared_hc_current_input`, route central GPU0 gather, HC norm,
   HC split, FFN norm, and router kernels through rank 0's stream when the gate
   is enabled.
3. Replace the corresponding GPU0 `cudaDeviceSynchronize()` calls with
   `cudaStreamSynchronize(rank0.stream)` under the gate.
4. Expose the gate through the appliance launcher and direct profiler harness.

## Verification

Local:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
```

V100:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
direct-token-major control, 32 slots, 256K, 2 decode steps
direct-token-major stream-sync candidate, same shape
```

## Definition of Done

- [x] Stream-sync gate is implemented in the TP/EP binary.
- [x] Launcher and profiler harness can enable the gate.
- [x] V100 build passes.
- [x] Direct control and candidate both pass with finite output head.
- [x] Candidate is promoted or rejected with measured evidence.
- [x] Status/vision/temp report are updated.
- [x] Cluster artifacts are copied into `logs/from-cluster/`.
- [x] Sprint artifacts are committed.

## Outcome

Implemented:

```text
--tp-hc-current-input-stream-sync-gate
DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC=1
tools/ds4-v100-tp-ep-profile.py --hc-current-stream-sync
```

The gate routes the central GPU0 HC-current control kernels through rank 0's
stream and replaces the matching device-wide GPU0 barriers with stream-scoped
barriers. The data layout is unchanged.

V100 build passed:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Direct 32-slot / 256K / 2-step A/B:

| Case | Generated tok/s decode | Continuation tok/s decode | Sum decode ms | HC-current ms | Output finite |
|---|---:|---:|---:|---:|---:|
| Control | `74.841520` | `91.326928` | `855.140307` | `711.608991` | `0` bad |
| Stream sync | `81.190638` | `93.199784` | `788.268223` | `647.492171` | `0` bad |

HTTP 32-request / 32-slot / 256K / 2-token A/B:

| Case | HTTP 200 | Server generated tok/s | Server generated tok/s decode | Server continuation tok/s |
|---|---:|---:|---:|---:|
| Control | `32/32` | `82.573137` | `97.500352` | `82.452144` |
| Stream sync | `32/32` | `83.813937` | `98.859925` | `84.429119` |

## Decision

Promote stream-scoped HC-current barriers as the TP/EP launcher default:

```text
DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC=1
```

The direct path shows the intended stage improvement:

```text
HC-current ms: 711.608991 -> 647.492171
sum decode ms: 855.140307 -> 788.268223
```

The HTTP improvement is smaller but positive, and correctness held in both
direct and HTTP modes. The next optimization should target the actual HC
control computation/fill chain, since synchronization scope alone only moves a
small part of the serving topline.

Artifacts:

```text
logs/from-cluster/sprint349-hc-stream-sync/cluster/
```
