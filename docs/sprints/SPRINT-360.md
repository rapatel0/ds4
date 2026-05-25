---
sprint: 360
title: TP/EP Launcher Default Pool-Norm Validation
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 360 - TP/EP Launcher Default Pool-Norm Validation

## Overview

Sprint 359 promoted fused compressed pool+norm as the TP/EP serving default
after a direct non-HTTP 32-step A/B. That proves the kernel path and default
edit locally, but production use goes through `tools/ds4-v100-run-appliance.sh`.

This sprint validates the promoted default through the launcher path on the
V100 pod:

- `--print-command` must include
  `--true-ds4-compressed-kv-fused-pool-norm-gate` without setting the env var.
- a launcher-started TP/EP HTTP server must answer selected-token requests at
  the `32` slot / `256K` / `position=262112` shape.

No PP/layer-split work. No MTP. This is an operational validation sprint for
the TP/EP default promoted by Sprint 359.

## Implementation

1. Copy the updated launcher script to the V100 workspace.
2. Run launcher `--print-command` with TP/EP serving env and verify the
   pool-norm flag is present by default.
3. Start the TP/EP HTTP server via the launcher at `32` slots / `256K`.
4. Send `32` concurrent `/v100/selected-token` requests for `32` tokens each.
5. Capture health/status/metrics/responses/server logs.
6. Confirm the server selected fused pool-norm rows and returned `32/32` HTTP
   200 responses.

## Verification

- Local shell syntax check passes.
- V100 launcher print-command proof passes.
- V100 launcher HTTP server reaches `/health`.
- Selected-token requests return `32/32` HTTP 200.
- Server output shows fused pool-norm selection.
- Artifacts are copied into `logs/from-cluster/`.

## Definition of Done

- [x] Launcher print-command proof captured.
- [x] Launcher-started HTTP selected-token run completes.
- [x] Results are summarized in this sprint doc.
- [x] `docs/sprints/STATUS.md` and `docs/sprints/VISION.md` are updated.
- [x] A temp status report is written.
- [x] Artifacts are committed.

## Execution Note

The first launcher attempt intentionally omitted the full true-attention typed
KV serving gate set and only tested the raw default command. That command
proved the pool-norm flag appears by default, but selected-token requests
returned HTTP 500 because the full TP/EP serving semantics were not enabled.

The valid run used the same required TP/EP true-attention gates as the profile
harness, while leaving
`DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM` unset. This tests the
pool-norm default without relying on the profile harness to inject it.

## Outcome

Launcher `--print-command` proof:

```text
logs/from-cluster/sprint360-launcher-pool-norm-default-valid/print-command.txt
```

includes:

```text
--true-ds4-compressed-kv-fused-pool-norm-gate
```

with no explicit pool-norm environment override.

Launcher-started selected-token HTTP run:

| Shape | Value |
|---|---:|
| slots | `32` |
| context | `262144` |
| start position | `262112` |
| requests | `32` |
| max tokens/request | `32` |
| HTTP 200 | `32/32` |
| client generated tok/s | `73.289956` |
| first token | `109328` |
| compressed projection lines | `1375` |
| fused pool-norm lines | `187` |
| fused input-fill lines | `0` |

The server was started through `tools/ds4-v100-run-appliance.sh`, not the
profile harness direct command. The default pool-norm gate was active in the
resolved launcher command and in server emitted-row logs.

## Decision

Sprint 359's default promotion is operationally validated through the launcher
HTTP path.

Keep:

- `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM=1` by default.
- fused input-fill and fused RoPE+round default-off.

Next work should move beyond pool-norm validation: either rerun the full chat
topline with the promoted default, or continue fusing the remaining
compressed-KV state/emit path.

Artifacts:

```text
logs/from-cluster/sprint360-launcher-pool-norm-default/
logs/from-cluster/sprint360-launcher-pool-norm-default-valid/
```
