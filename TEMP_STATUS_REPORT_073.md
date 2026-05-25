# TEMP Status Report 073 - Sprint 361 Chat A/B

Date: 2026-05-25

## Current Focus

TP/EP launcher chat-serving metrology. Sprint 361 checks whether the
pool-norm default promoted for decode also improves the short
`/v1/chat/completions` path.

## Chat A/B

Shape:

```text
endpoint: /v1/chat/completions
requests: 32 concurrent
tokens/request: 8
slots: 32
context: 256K
launcher: tools/ds4-v100-run-appliance.sh
```

| Variant | HTTP 200 | Generated tokens | Client tok/s | Fused pool rows | First token |
|---|---:|---:|---:|---:|---:|
| pool off | 32/32 | 256 | 24.280060 | 0 | 24893 |
| default pool on | 32/32 | 256 | 24.118711 | 126 | 24893 |

## Interpretation

- The pool-norm default is active through the full chat endpoint.
- It is stable: same first selected token and all responses returned HTTP 200.
- It is not a visible short-chat throughput win at `8` tokens/request.
- This is consistent with the win being in decode/compressed-KV work while
  short chat remains dominated by tokenization, prefill, HTTP orchestration,
  and response handling.

## Decision

Keep pool-norm default-on based on Sprint 359 direct decode and Sprint 360
selected-token launcher validation. Do not claim full chat topline improvement
from Sprint 361.

## Next Best Step

Either:

1. run a longer decode-heavy chat matrix, or
2. continue fusing the remaining compressed-KV state/emit boundary, which is
   more likely to move full serving throughput materially.

Artifacts:

```text
logs/from-cluster/sprint361-launcher-chat-pool-norm/
```
