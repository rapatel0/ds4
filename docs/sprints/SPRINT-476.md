# Sprint 476: TP/EP Direct Peer-Copy No-SYS Guard

## Overview

Sprint 475 made NCCL collectives avoid SYS by default. This sprint extends the
same rule to explicit direct peer copies in the TP/EP appliance. NCCL ring
selection does not cover calls that go through the runtime's
`ds4_peer_copy_async` wrapper, so those copies need their own accounting and
guardrail.

No PP/layer-split work is in scope.

## Goal

Make direct SYS peer copies visible and rejectable in the TP/EP serving path.
The production target is:

```text
NCCL collectives: no SYS
direct peer copies: no silent SYS
```

If direct SYS copies still exist, this sprint should produce exact evidence:
which copy class, source/destination pair, byte count, and next implementation
target.

## Implementation Tasks

1. Verify the existing `ds4_peer_copy_async` accounting covers all direct
   `cudaMemcpyPeerAsync` sites in the TP/EP serving binary.
2. Ensure launcher/profile controls can enable accounting and rejection:
   - `DS4_V100_TP_EP_PEER_ACCOUNTING`
   - `DS4_V100_TP_EP_PEER_REJECT_SYS`
   - `--tp-peer-accounting`
   - `--tp-peer-reject-sys`
3. Run a target-shape smoke with accounting enabled but rejection disabled to
   discover whether direct peer copies still use SYS.
4. If `sys_ops == 0`, promote rejection as the default.
5. If `sys_ops > 0`, keep rejection diagnostic-only and identify the first
   concrete hot copy site to replace with NCCL or a topology-aware route.

## Definition of Done

- Profile artifacts include parsed `peer_copy_*` summary fields.
- A V100 artifact proves either:
  - direct peer copies have `sys_ops=0`, or
  - direct peer copies still use SYS and the first offending edge is recorded.
- The launcher default is updated only if the evidence proves it will not break
  the current serving path.
- `docs/sprints/VISION.md` and a `TEMP_STATUS_REPORT_476.md` file record the
  result.

## Stop Conditions

- Do not promote `DS4_V100_TP_EP_PEER_REJECT_SYS=1` unless a serving smoke
  proves zero direct SYS peer copies.
- Do not treat NCCL graph `0 SYS` as proof for direct peer-copy paths.
- Do not hide direct SYS by disabling accounting.

## Outcome

Implemented permanent direct peer-copy accounting in the TP/EP serving binary:

- all `cudaMemcpyPeerAsync` calls in `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
  now go through `ds4_peer_copy_async`
- the wrapper classifies edges as `NV1`, `NV2`, `SYS`, or `unknown` using the
  known V100 topology and `CUDA_VISIBLE_DEVICES`
- `--tp-peer-accounting-gate` enables counters
- `--tp-peer-reject-sys-gate` turns any classified SYS edge into a hard CUDA
  error
- `/status` and `/metrics` expose `peer_copy_*` fields so HTTP profiles can
  capture the counters even when the harness terminates the server
- launcher, profile, env example, and k8s config now expose the diagnostic
  controls

Validation:

```text
artifact: /localpool/ds4/workspace/s476-peer-account-status-s32-t2
shape:    32 slots, 256K context, 32 HTTP requests, 2 generated tokens
policy:   natural CUDA order, NCCL no-SYS ring, peer accounting on, reject off
```

Topline:

| Metric | Value |
|---|---:|
| HTTP 200 | 32/32 |
| Server generated decode tok/s | 37.778247 |
| Server continuation decode tok/s | 37.825860 |
| Client generated tok/s | 3.332952 |
| Min free VRAM | 2838 MiB |
| NCCL graph SYS edges | 0 |
| Direct peer-copy ops | 1,488,745 |
| Direct peer-copy bytes | 12,587,732,288 |
| Direct NV1 bytes | 3,598,217,600 |
| Direct NV2 bytes | 3,597,693,312 |
| Direct SYS ops | 638,028 |
| Direct SYS bytes | 5,391,821,376 |
| First SYS edge | src 0 -> dst 5, 3,072 bytes |

## Decision

Do not promote `DS4_V100_TP_EP_PEER_REJECT_SYS=1`. NCCL collectives are clean
under the no-SYS ring, but direct peer copies still move about `5.39 GiB`
through SYS-classified edges in this short 32-slot serving smoke.

The next implementation target is topology-aware direct copy routing or NCCL
replacement for the direct peer-copy classes. The first visible offending edge
is `0 -> 5`; the current accounting does not yet tag call-site operation names,
so the next sprint should add call-site labels to `ds4_peer_copy_async` before
rewriting individual transfer paths.
