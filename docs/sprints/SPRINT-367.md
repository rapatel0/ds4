---
sprint: 367
title: TP/EP Chat Long-Decode Default Topline
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 367 - TP/EP Chat Long-Decode Default Topline

## Overview

Sprint 366 promoted compressed dense event waits based on direct and
selected-token HTTP evidence. The next production-readiness question is
whether that default improves the real `/v1/chat/completions` path at a
decode-heavy shape where HTTP/tokenization overhead is less dominant than the
short 8-token chat gate from Sprint 361.

This sprint runs a launcher-level chat A/B:

- control: promoted launcher defaults with event wait explicitly disabled,
- candidate: promoted launcher defaults with event wait enabled by default.

This is TP/EP-only. No PP/layer-split work. No MTP.

## Verification

- V100 `/v1/chat/completions` A/B at `32` slots / `256K`.
- `32` concurrent requests.
- `32` generated tokens/request.
- Both variants return `32/32` HTTP 200.
- Both variants preserve finite output-head behavior and stable first-token
  metadata if available.
- Compare client tok/s, server/scaffold decode proxy, compressed-KV sum, and
  event-wait row counts.

## Definition of Done

- [x] The promoted default is selected in the candidate chat run.
- [x] The disable control selects zero event-wait rows.
- [x] Both chat variants complete without HTTP failures.
- [x] Results are summarized in this sprint doc, `STATUS.md`, and `VISION.md`.
- [x] A temp status report is written.
- [x] Artifacts are committed.

## Outcome

The first attempted chat shape reused the selected-token start position
`262112`. That is invalid for chat because prompt prefill consumes additional
positions before the 32 generated tokens. Both variants reached decode work but
returned HTTP 500 with:

```text
tp_runtime_dense_kv_slice_failed slot 7 position is outside configured context
```

The valid rerun used `position=262080`, leaving room for chat prompt prefill
and 32 generated tokens.

Valid `/v1/chat/completions` A/B at `32` slots / `256K`, `32` concurrent
requests, and `32` generated tokens/request:

| Variant | HTTP 200 | Coalesced batch | Generated tokens | First token | Finite bad | Client tok/s | Server wall tok/s | Server decode tok/s | Compressed-KV sum | Event rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| event wait disabled | 32/32 | 32 | 1024 | 89340 | 0 | 50.648397 | 81.426024 | 96.116667 | 5100.469710 ms | 0 |
| default event wait | 32/32 | 32 | 1024 | 89340 | 0 | 52.022782 | 83.891024 | 99.521680 | 4681.992882 ms | 1967 |

## Decision

Sprint 366's promotion holds through the real chat endpoint at a decode-heavy
shape. The default event-wait path improves:

```text
client tok/s:        50.648397 -> 52.022782
server wall tok/s:   81.426024 -> 83.891024
server decode tok/s: 96.116667 -> 99.521680
compressed-KV sum:   5100.469710 -> 4681.992882 ms
```

This is still far below the long-term target, but it is a real production-path
gain and confirms the selected-token improvement was not an endpoint artifact.

Artifacts:

```text
logs/from-cluster/sprint367-chat-long-default-topline/
logs/from-cluster/sprint367-chat-long-default-topline-valid/
```
