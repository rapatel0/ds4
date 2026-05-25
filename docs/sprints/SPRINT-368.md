---
sprint: 368
title: TP/EP Chat Context Admission
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 368 - TP/EP Chat Context Admission

## Overview

Sprint 367 confirmed the promoted event-wait default through the real chat
endpoint, but it also exposed a serving correctness gap: chat can accept a
request shape that reaches the configured context boundary after prompt
prefill, then fails inside the GPU decode path with HTTP 500.

Production serving should reject that request before GPU work with a clear
client error. Admission must account for:

- request start position,
- cache hit vs miss,
- prompt prefill tokens on misses,
- requested generation tokens,
- configured 256K context.

This is TP/EP-only. No PP/layer-split work. No MTP.

## Implementation

1. Add a TP/EP HTTP context admission helper.
2. Reject over-context generation requests before slot assignment/decode.
3. Return a structured JSON error and HTTP 400 instead of HTTP 500.
4. Preserve valid long-context chat behavior from Sprint 367.

## Verification

- V100 invalid chat shape:
  - `ctx=262144`,
  - `position=262112`,
  - `32` generated tokens,
  - prompt prefill present,
  - expect HTTP 400 and no `tp_ep_http_decode_failed`.
- V100 valid chat shape:
  - `ctx=262144`,
  - `position=262080`,
  - `32` generated tokens,
  - expect `32/32` HTTP 200.
- Local syntax checks pass.

## Definition of Done

- [x] Invalid over-context chat is rejected before GPU decode.
- [x] Valid long-context chat still succeeds.
- [x] The response error includes enough context for operators to fix the
      request shape.
- [x] Results are summarized in this sprint doc, `STATUS.md`, and `VISION.md`.
- [x] A temp status report is written.
- [x] Artifacts are committed.

## Outcome

Implemented explicit TP/EP HTTP context admission for generation requests.
The guard computes:

```text
start_position + prompt_prefill_steps + requested_decode_steps
```

before slot assignment and GPU decode. Cache hits use the cached slot position
and zero prompt prefill. Cache misses include `prompt_tokens - 1` prefill
steps. Requests that exceed the configured 256K context now return HTTP 400
with a structured error.

Invalid chat shape from Sprint 367:

```text
ctx=262144
position=262112
max_tokens=32
prompt_prefill_steps=16
final_position=262160
```

now returns:

```json
{"error":"context_window_exceeded","ctx":262144,"start_position":262112,"prompt_prefill_steps":16,"requested_decode_steps":32,"final_position":262160,"cache_hit":0}
```

with `HTTP_STATUS:400`. The server log contains
`tp_ep_http_context_rejected`, and no `tp_ep_http_decode_failed` line is
emitted.

Valid long-context chat at `position=262080` still passes:

| Shape | HTTP 200 | Coalesced batch | Generated tokens | First token | Finite bad | Client tok/s | Server wall tok/s | Server decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 requests x 32 tokens | 32/32 | 32 | 1024 | 89340 | 0 | 51.069220 | 82.089657 | 98.301727 |

V100 build passed:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Local validation passed:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
./ds4_test --server
./ds4_test --metal-kernels
```

## Decision

Keep the admission guard as production behavior. Over-context chat requests
are client-shape errors, not backend decode failures. This closes the specific
Sprint 367 failure mode and makes long-context serving more predictable for
operators.

Artifacts:

```text
logs/from-cluster/sprint368-chat-context-admission-invalid/
logs/from-cluster/sprint368-chat-context-admission-valid/
```
