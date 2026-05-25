---
sprint: 350
title: TP/EP HC Current Substage Telemetry
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 350 - TP/EP HC Current Substage Telemetry

## Overview

Sprint 349 promoted stream-scoped HC-current barriers, but the HC-current path
is still the largest direct-stage timer. The next implementation step is to
make that monolithic timer actionable by splitting it into stable substages:
HC state gather/mix, split broadcast/current shard reduction, current gather,
FFN/router control, and dense/routed input fill.

No PP/layer-split work. No MTP.

## Implementation

1. Add cumulative HC-current substage timing fields to the direct token-major
   summary.
2. Keep the timing passive and always available in direct summaries.
3. Parse the new fields in `tools/ds4-v100-tp-ep-profile.py`.

## Verification

- Local syntax/build checks pass.
- V100 direct 32-slot / 256K / 2-step run passes with finite output head.
- Summary JSON records the new substage fields.

## Definition of Done

- [x] Direct summary includes HC-current substage fields.
- [x] Profiler harness parses the fields.
- [x] V100 build passes.
- [x] V100 direct run passes and records substage data.
- [x] Docs/status/temp report are updated.
- [x] Artifacts are copied into `logs/from-cluster/`.
- [x] Sprint artifacts are committed.

## Outcome

Added HC-current substage timing to the token-major direct summary and parsed
those fields in the profiler harness.

V100 direct validation:

```text
slots: 32
ctx: 262144
decode steps: 2
stream sync: on
generated tok/s decode: 92.630324
continuation tok/s decode: 99.664366
output_head_finite_bad: 0
```

Substage totals across `86` layer-step invocations:

| Field | ms |
|---|---:|
| `sum_hc_current_seed_ms` | `2.485326` |
| `sum_hc_current_attn_mix_ms` | `42.340819` |
| `sum_hc_current_split_ms` | `1.245295` |
| `sum_hc_current_gather_ms` | `6.960973` |
| `sum_hc_current_ffn_router_ms` | `1.784974` |
| `sum_hc_current_fill_pack_ms` | `28.248863` |
| Substage total | `83.066250` |
| Old labeled `sum_hc_current_input_ms` | `557.301289` |

## Decision

The old `sum_hc_current_input_ms` label is too broad. It times everything
before the EP/dense/compose stage begins, not only HC-current work. The actual
HC-current substages measured here account for only about `83.1 ms` of the
`557.3 ms` pre-EP timer. The remaining `~474.2 ms` is dominated by true
attention/compressed-KV work that currently runs before the EP timer starts.

Next sprint should split or optimize that true-attention/compressed-KV prefix,
especially compressed projection/store and dense-fill/WMMA fragmentation,
rather than continuing to chase HC-current gather/broadcast alone.

Artifacts:

```text
logs/from-cluster/sprint350-hc-current-breakdown/cluster/
```
