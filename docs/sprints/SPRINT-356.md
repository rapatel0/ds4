---
sprint: 356
title: TP/EP Compressed Fusion Serving Controls
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 356 - TP/EP Compressed Fusion Serving Controls

## Overview

Sprint 355 found the first meaningful compressed-KV stage win:
fused pool+norm reduced emitted-row compressed-KV time, but it was only tested
through the direct profile harness and remains opt-in. Before promotion, the
fusion gates need to be reachable from the TP/EP serving launcher and tested
in combination.

This sprint wires the compressed-fusion gates into the appliance launcher and
profile harness, then tests combined fused input-fill plus fused pool+norm at
the emitted-row direct shape. If the combined result is stable, it can be
tested through HTTP serving in a follow-up or this sprint if time permits.

No PP/layer-split work. No MTP. No default semantic change unless V100 evidence
justifies promotion.

## Implementation

1. Add launcher environment variables for:
   - `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_INPUT_FILL`
   - `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ROPE_ROUND`
   - `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM`
2. Validate these variables like the existing TP/EP gates.
3. Propagate them into the TP/EP binary command.
4. Add profile-harness HTTP environment propagation for the same flags.
5. Document the variables in `deploy/v100/ds4-v100-appliance.env.example`.
6. Run V100 direct combined-fusion A/B at `32` slots / `256K`.

## Verification

- Local shell and Python syntax checks pass.
- V100 `sm_70` build passes.
- Launcher `--print-command` shows the expected compressed-fusion flags.
- V100 direct combined-fusion A/B passes.
- Selected token stays stable.

## Definition of Done

- [x] Launcher environment variables are implemented and default off.
- [x] Profile harness supports HTTP/direct propagation of the fusion flags.
- [x] Env example documents the variables.
- [x] V100 build passes.
- [x] V100 direct combined-fusion A/B passes.
- [x] Docs/status/temp report are updated.
- [x] Artifacts are copied into `logs/from-cluster/`.
- [x] Sprint artifacts are committed.

## Outcome

Added TP/EP serving controls for the compressed-fusion gates:

- `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_INPUT_FILL`
- `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ROPE_ROUND`
- `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM`

The appliance launcher validates those variables, maps them to the TP/EP
binary flags, and the profile harness now propagates them in HTTP mode as well
as direct mode. `DS4_V100_TP_EP_POSITION` is also passed through the HTTP
profile harness so emitted-row HTTP tests can be targeted deliberately.

Launcher command proof:

```text
logs/from-cluster/sprint356-combined-compressed-fusions/cluster/print-command.txt
```

V100 same-binary direct A/B, `32` slots / `256K`, emitted-row
`position=262143`, one decode step:

| Variant | Fused input layers | Fused pool layers | Decode tok/s | Decode ms | Pre-EP compressed-KV ms | Compressed-KV sum ms | First token |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | `0` | `0` | `80.511365` | `397.459414` | `130.812593` | `130.329162` | `54639` |
| input-fill + pool-norm | `21` | `41` | `81.311102` | `393.550195` | `128.988052` | `128.467170` | `54639` |

The candidate preserved the selected token and finite output head. The combined
candidate improved decode by `+0.99%` and reduced compressed-KV sum by
`1.861992 ms` in this run.

## Decision

Do not promote defaults yet. The serving controls are now in place and the
combined direct result is positive, but the gain remains small and emitted-row
HTTP testing needs a cleaner harness because prompt prefill changes the
effective position. Keep both gates opt-in and use the launcher controls for
future HTTP/profile experiments.

The next sprint should either:

- add a profile mode for emitted-row HTTP/tokenized serving without prompt
  prefill ambiguity, or
- run repeated direct A/B pairs to determine whether fused pool+norm alone or
  the combined gates are stable enough to promote.

Artifacts:

```text
logs/from-cluster/sprint356-combined-compressed-fusions/cluster/
```
