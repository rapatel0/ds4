# Sprint 271 - TP/EP Compose Stage Breakdown

Date: 2026-05-23
Status: Complete

## Overview

Sprint 271 adds timing visibility inside the TP/EP token-major compose stage.
After Sprint 270, compose/all-to-all remained the dominant scaffold stage, but
the tool only reported one aggregate compose number.

## Implementation

`tools/ds4-v100-tp-ep-full-layer-smoke` now reports compose as:

```text
compose_reduce_ms
compose_copy_ms
compose_final_ms
```

for token-major summaries and per-layer decode-loop logs.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint271-compose-stage-breakdown/cluster/stage-breakdown-16step.log`
- `logs/from-cluster/sprint271-compose-stage-breakdown/cluster/stage-breakdown-16step-summary.log`

Result at `32` slots / `256K`, `16` token steps:

| Metric | Value |
|---|---:|
| Passing invocations | 688/688 |
| ms/token proxy | 37.288880 |
| Projected slot-step tok/s | 858.164677 |
| Sum EP/overlap ms | 268.723414 |
| Sum compose ms | 327.657087 |
| Compose reduce ms | 49.805028 |
| Compose copy ms | 242.803068 |
| Compose final ms | 35.048991 |
| Checksum | 8244145680 |
| Result | PASS |

## Decision

The peer-copy/all-to-all portion dominates compose. Local contribution
reduction and final compose are much smaller. This supports treating the next
compose optimization as copy-scheduler work, but per user steer the broader
priority after recording the scheduler result is TP/EP end-to-end serving.
