# TEMP Status Report 070 - Sprint 358 Amortized Fusion A/B

Date: 2026-05-25

## Current Focus

TP/EP serving performance only. This sprint tested whether the compressed-KV
fusion gates that looked promising in one-token emitted-row tests still help
when HTTP overhead is amortized across a longer selected-token decode.

## Important Finding

Starting a 32-token run at `position=262143` with `ctx=262144` is invalid.
The first decode can run, then the next step is outside the configured context.
The server returned `tp_ep_decode_failed` with:

```text
tp_runtime_dense_kv_slice_failed slot 7 position is outside configured context
```

The valid 32-token run starts at `position=262112`, ending at the emitted-row
boundary.

## Latest V100 A/B

Shape:

```text
32 slots
256K context
position=262112
32 tokens/request
32 concurrent selected-token HTTP requests
HC current stream sync enabled
```

| Variant | HTTP 200 | Client tok/s | Emitted layers | Fused input layers | Fused pool layers | Compressed-KV sum ms | Scaffold decode tok/s | First token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 32/32 | 71.818394 | 187 | 0 | 0 | 3506.921796 | 98.772310 | 109328 |
| input-fill + pool-norm | 32/32 | 72.297469 | 187 | 671 | 187 | 3509.986423 | 98.505291 | 109328 |
| pool-norm only | 32/32 | 73.052883 | 187 | 0 | 187 | 3474.878472 | 97.552747 | 109328 |

## Decision

Do not promote any compressed-fusion defaults.

- Combined input-fill + pool-norm is not promotable; the longer run shows
  compressed-KV sum and scaffold decode proxy regression.
- Pool-norm only remains promising but ambiguous: better client wall tok/s and
  lower compressed-KV sum, worse scaffold decode proxy.

## Next Best Step

Move back to implementation, not wrapper work:

1. Confirm pool-norm with a repeated/direct multi-step A/B that removes HTTP
   orchestration from the decision, or
2. Fuse the deeper compressed state/emit boundary where the stage timer still
   shows substantial work.

Artifacts:

```text
logs/from-cluster/sprint358-amortized-selected-token-fusions/
logs/from-cluster/sprint358-amortized-selected-token-fusions-valid/
```
