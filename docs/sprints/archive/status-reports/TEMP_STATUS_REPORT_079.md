# TEMP Status Report 079 - Sprint 367

Date: 2026-05-25

## Current Focus

Validate Sprint 366's promoted compressed dense event-wait default through the
real `/v1/chat/completions` endpoint, not only selected-token diagnostics.

## Finding

The selected-token start position `262112` is invalid for chat at
`32` generated tokens because chat prompt prefill consumes additional context
positions. Both variants decoded but returned HTTP 500 when the position hit
the configured `256K` boundary.

The valid chat run used `position=262080`.

## Valid Chat A/B

Shape:

```text
endpoint: /v1/chat/completions
ctx: 262144
slots: 32
position: 262080
requests: 32
tokens/request: 32
```

| Variant | HTTP 200 | Coalesced batch | Generated tokens | First token | Bad | Client tok/s | Server wall tok/s | Server decode tok/s | Compressed-KV sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| event wait disabled | 32/32 | 32 | 1024 | 89340 | 0 | 50.648397 | 81.426024 | 96.116667 | 5100.469710 ms |
| default event wait | 32/32 | 32 | 1024 | 89340 | 0 | 52.022782 | 83.891024 | 99.521680 | 4681.992882 ms |

## Decision

Sprint 366's default promotion holds through the actual chat API. Keep
`DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DENSE_EVENT_WAIT=1`.

## Artifacts

```text
logs/from-cluster/sprint367-chat-long-default-topline/
logs/from-cluster/sprint367-chat-long-default-topline-valid/
```
