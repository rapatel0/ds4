---
sprint: 366
title: TP/EP Compressed Dense Event Dependencies
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 366 - TP/EP Compressed Dense Event Dependencies

## Overview

Sprints 364 and 365 showed that compressed-KV input-fill micro-fusions are too
small or regress when they change locality. The larger boundary still visible
in the 32-slot / 256K emitted-row runs is the compressed projection group:

```text
local staged current -> half input fill
host synchronize streams
attention compressor dense
host synchronize dense streams
gather/state/emit
ratio-4 indexer fill
host synchronize streams
indexer dense group
host synchronize dense streams
gather/state/score
```

This sprint keeps the TP/EP data layout and all math unchanged, but replaces
the input-fill-to-dense host barriers with CUDA stream-event dependencies for
the compressed attention and ratio-4 indexer dense groups.

This is TP/EP-only. No PP/layer-split work. No MTP.

## Implementation

1. Add `--true-ds4-compressed-kv-dense-event-wait-gate`.
2. After compressed attention input-fill kernels are enqueued on each rank's
   local stream, record a per-rank event and make that rank's dense stream wait
   on it instead of synchronizing the stream on the host.
3. Apply the same pattern to the ratio-4 indexer input-fill group.
4. Keep all downstream dense-stream synchronizes before stats/gather, so the
   output and correctness semantics remain unchanged.
5. Expose the gate through:
   - `tools/ds4-v100-run-appliance.sh`,
   - `deploy/v100/ds4-v100-appliance.env.example`,
   - `tools/ds4-v100-tp-ep-profile.py`.
6. Keep default off unless V100 HTTP evidence supports promotion.

## Verification

- Local syntax checks pass.
- V100 `sm_70` build passes.
- V100 direct 32-step A/B at `32` slots / `256K` / `position=262112`:
  - control: production launcher defaults,
  - candidate: compressed dense event waits.
- V100 selected-token HTTP A/B at the same shape.
- Both variants preserve finite output head and first selected token.
- Compare generated decode tok/s, client tok/s, compressed-KV sum, and the
  compressed dense/input-fill stage timings.

## Definition of Done

- [x] The event-wait gate compiles for `sm_70`.
- [x] The gate is reachable from direct smoke, launcher, and profile harness.
- [x] V100 direct A/B passes correctness invariants.
- [x] V100 selected-token HTTP A/B passes correctness invariants.
- [x] Results are summarized in this sprint doc, `STATUS.md`, and `VISION.md`.
- [x] A temp status report is written.
- [x] Artifacts are committed.

## Outcome

Implemented compressed dense event dependencies:

- `--true-ds4-compressed-kv-dense-event-wait-gate`,
- `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DENSE_EVENT_WAIT=1`,
- `tools/ds4-v100-tp-ep-profile.py --compressed-dense-event-wait`,
- `tools/ds4-v100-tp-ep-profile.py --disable-compressed-dense-event-wait`.

The gate records a per-rank event after compressed attention/indexer input
fill work on the local stream, then makes the rank's dense stream wait on that
event. This removes the host-side stream synchronizes before the compressed
dense launches while leaving downstream dense completion, gather, state/emit,
typed-KV, and output semantics unchanged.

V100 build passed:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Direct 32-step A/B at `32` slots / `256K` / `position=262112`:

| Variant | First token | Finite bad | Decode tok/s | Wall tok/s | Compressed-KV sum | Attn input fill | Indexer input fill | Event rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 98751 | 0 | 96.214306 | 75.215206 | 3431.137744 ms | 395.649216 ms | 124.015780 ms | 0 |
| dense event wait | 98751 | 0 | 99.093248 | 76.897975 | 3127.236790 ms | 120.937977 ms | 53.476868 ms | 1312 |

Selected-token HTTP A/B at `32` slots / `256K` / `position=262112`:

| Variant | HTTP 200 | First token | Finite bad | Client tok/s | Compressed-KV sum | Attn input fill | Indexer input fill | Event rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 32/32 | 109328 | 0 | 71.833757 | 3437.636456 ms | 405.363361 ms | 129.927227 ms | 0 |
| dense event wait | 32/32 | 109328 | 0 | 74.432464 | 3137.755187 ms | 123.787805 ms | 52.812849 ms | 1312 |

Launcher/profile default proof at `1` request / `1` token /
`position=262143`:

| Variant | HTTP 200 | Client tok/s | Compressed-KV sum | Event rows | Fused pool rows |
|---|---:|---:|---:|---:|---:|
| default | 1/1 | 0.575538 | 108.734830 ms | 39 | 39 |
| explicit disable | 1/1 | 0.559920 | 120.919675 ms | 0 | 39 |

Local validation:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
```

## Decision

Promote compressed dense event waits as the TP/EP launcher default. This is a
serving-visible win at the target long-context 32-slot shape:

```text
direct decode tok/s: 96.214306 -> 99.093248
direct wall tok/s:   75.215206 -> 76.897975
HTTP client tok/s:   71.833757 -> 74.432464
compressed-KV sum:   3437.636456 -> 3137.755187 ms in HTTP
```

This does not change dtype or tensor layout. It removes unnecessary
host-side barriers between existing local fill work and the dense streams.
The next optimization can build on this by targeting the remaining dense
projection/state cost rather than the removed input-fill barrier.

Artifacts:

```text
logs/from-cluster/sprint366-compressed-dense-event-wait/
logs/from-cluster/sprint366-compressed-dense-event-wait-http/
logs/from-cluster/sprint366-compressed-dense-event-wait-default-proof/
```
