---
sprint: 361
title: TP/EP Launcher Chat Pool-Norm Topline
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 361 - TP/EP Launcher Chat Pool-Norm Topline

## Overview

Sprint 360 validated fused compressed pool+norm through the launcher on the
selected-token endpoint. The next serving-facing question is whether the
promoted default survives the actual chat/completions path, including
tokenization, prompt prefill, output-head sampling/feed, and HTTP response
formatting.

This sprint runs launcher-started `/v1/chat/completions` A/B:

- control: explicit `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM=0`
- candidate: launcher default, pool-norm env unset

No PP/layer-split work. No MTP. The goal is operational metrology, not a new
kernel.

## Implementation

1. Start the TP/EP launcher with full true-attention typed-KV serving gates and
   pool-norm explicitly off.
2. Send `32` concurrent `/v1/chat/completions` requests with `8` generated
   tokens each.
3. Start the TP/EP launcher again with the same gates and pool-norm env unset.
4. Send the same request set.
5. Compare HTTP 200 count, client generated tok/s, response token counts,
   server logs, first selected token metadata, and fused pool-norm evidence.

## Verification

- Both launcher runs reach `/health`.
- Both runs return `32/32` HTTP 200 responses.
- Candidate server logs show fused pool-norm rows.
- Control server logs show no fused pool-norm rows.
- Artifacts are copied into `logs/from-cluster/`.

## Definition of Done

- [x] Control chat run completes.
- [x] Default-pool chat run completes.
- [x] Results are summarized in this sprint doc.
- [x] `docs/sprints/STATUS.md` and `docs/sprints/VISION.md` are updated.
- [x] A temp status report is written.
- [x] Artifacts are committed.

## Outcome

Launcher-started `/v1/chat/completions` A/B at `32` slots / `256K`,
`32` concurrent requests, and `8` generated tokens/request:

| Variant | HTTP 200 | Generated tokens | Client tok/s | Fused pool rows | Fused input rows | Compressed lines | First token |
|---|---:|---:|---:|---:|---:|---:|---:|
| pool off | `32/32` | `256` | `24.280060` | `0` | `0` | `1032` | `24893` |
| launcher default pool on | `32/32` | `256` | `24.118711` | `126` | `0` | `1032` | `24893` |

The promoted default is active through the full chat endpoint and preserves
the first selected token, but it does not produce a visible chat-wrapper
topline win at this short `8` token/request shape. Client tok/s is `-0.66%`,
which is within the range where tokenization, prefill, HTTP orchestration, and
short generation length can dominate the small pool-norm kernel win.

## Decision

Keep fused compressed pool+norm default-on because Sprint 359 and Sprint 360
proved a direct/selected-token decode win and launcher correctness. Do not
claim a full chat throughput improvement from this sprint.

Next work should target a larger bottleneck:

- run a longer chat/completions serving matrix where decode dominates, or
- continue compressed-KV state/emit fusion, or
- return to correctness/parity gaps if the next milestone is production
  readiness rather than micro-optimization.

Artifacts:

```text
logs/from-cluster/sprint361-launcher-chat-pool-norm/
```
