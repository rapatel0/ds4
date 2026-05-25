---
sprint: 357
title: TP/EP Emitted-Row HTTP Profile Mode
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 357 - TP/EP Emitted-Row HTTP Profile Mode

## Overview

Sprint 356 wired compressed-fusion controls into the TP/EP serving launcher,
but the HTTP profile harness still used chat prompts. That made emitted-row
compressed-KV testing ambiguous because prompt prefill changes the effective
decode position.

This sprint adds a selected-token HTTP profile mode that calls the existing
`POST /v100/selected-token` endpoint directly. The mode avoids prompt prefill
and uses `DS4_V100_TP_EP_POSITION=262143`, so all 41 compressed-emitting
layers are exercised deterministically through the resident HTTP serving path.

No PP/layer-split work. No MTP. The compressed-fusion gates remain opt-in.

## Implementation

1. Add `--http-endpoint {chat,selected-token}` to
   `tools/ds4-v100-tp-ep-profile.py`.
2. Keep `chat` as the default endpoint for existing profile behavior.
3. Route `selected-token` mode to `POST /v100/selected-token` with no prompt
   payload.
4. Reuse the existing `--position` environment propagation so emitted rows are
   selected deliberately.
5. Parse TP/EP server summary lines from HTTP `server.out`, matching the direct
   profile summary fields for compressed-KV timing and fusion counts.
6. Run V100 HTTP selected-token A/B at `32` slots / `256K` /
   `position=262143`.

## Verification

- Local Python syntax check passes.
- Local whitespace check passes.
- V100 profile harness Python syntax check passes.
- V100 `sm_70` full-layer smoke build passes.
- V100 selected-token HTTP control run returns `32/32` HTTP 200 responses.
- V100 selected-token HTTP fused run returns `32/32` HTTP 200 responses.
- Both HTTP runs exercise all `41` emitted compressed-KV layers.

## Definition of Done

- [x] Profile harness supports selected-token HTTP mode.
- [x] Existing chat HTTP mode remains the default.
- [x] HTTP summaries include parsed compressed-KV stage timing.
- [x] V100 build passes.
- [x] V100 emitted-row HTTP control/fused A/B passes.
- [x] Docs/status/temp report are updated.
- [x] Artifacts are copied into `logs/from-cluster/`.
- [x] Sprint artifacts are committed.

## Outcome

Added selected-token HTTP profile support:

```text
tools/ds4-v100-tp-ep-profile.py --run-mode http --http-endpoint selected-token
```

This mode starts the TP/EP server, waits for readiness, and sends
`/v100/selected-token` requests with no prompt text. The summary now records
`http_endpoint` and parses TP/EP timing lines from server output, including
compressed-KV layer counts, emitted-row counts, fusion counts, and compressed
stage subtimers.

V100 same-binary HTTP A/B, `32` slots / `256K`, emitted-row
`position=262143`, one generated token/request:

| Variant | HTTP 200 | Emitted layers | Fused input layers | Fused pool layers | Client tok/s | Compressed-KV sum ms | Attn state/emit ms | Indexer state/emit ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | `32/32` | `41` | `0` | `0` | `19.739916` | `127.697384` | `24.309011` | `8.610316` |
| input-fill + pool-norm | `32/32` | `41` | `20` | `40` | `19.719484` | `123.651985` | `22.341132` | `7.914220` |

The fused HTTP run reduced parsed compressed-KV stage time by `4.045399 ms`
while preserving successful HTTP responses. The one-token selected-token
client throughput was flat/slightly lower because this mode intentionally
measures a tiny request body where HTTP orchestration dominates the client
rate.

## Decision

Keep the compressed-fusion gates opt-in. Sprint 357 proves the gates are
reachable and active through the resident HTTP serving path, and it confirms a
real compressed-KV stage reduction at the emitted-row shape. It does not prove
a practical serving-topline win because the selected-token one-token HTTP mode
is dominated by request overhead.

The next TP/EP sprint should stop expanding profile wrappers and move back to
end-to-end serving progress: either reduce the remaining compressed-KV
state/emit fragmentation in the direct path or run a longer selected-token/chat
serving A/B where request overhead is amortized.

Artifacts:

```text
logs/from-cluster/sprint357-http-selected-emitted-fusions/cluster/
```
