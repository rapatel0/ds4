---
sprint: 351
title: TP/EP True-Attention Prefix Telemetry
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 351 - TP/EP True-Attention Prefix Telemetry

## Overview

Sprint 350 proved that `sum_hc_current_input_ms` is a broad pre-EP timer, not
only HC-current input work. The measured HC-current substages account for only
about `83 ms` of the `557 ms` pre-EP total in the direct 32-slot / 256K run.
The likely hidden cost is true-attention and compressed-KV prefix work.

This sprint splits that prefix into stable stage timers so the next
optimization sprint can target the measured owner instead of guessing.

No PP/layer-split work. No MTP. No semantic change to TP/EP serving.

## Implementation

1. Add a passive `PreEpPrefixBreakdown` to the TP/EP full-layer smoke path.
2. Time the existing true-attention prefix gates:
   - HC-current bridge
   - attention projection prefix
   - compressed KV projection/store
   - attention state update
   - typed KV history load
   - raw/window attention read
   - attention output projection
   - post-attention FFN input
3. Emit per-layer and scaffold summary fields for those timers.
4. Parse the new fields in `tools/ds4-v100-tp-ep-profile.py`.

## Verification

- Local syntax/build checks pass.
- V100 `sm_70` build of `tools/ds4-v100-tp-ep-full-layer-smoke` passes.
- V100 direct 32-slot / 256K / 2-step TP/EP run passes with finite output
  head.
- Summary JSON records the new pre-EP fields.

## Definition of Done

- [x] Direct layer rows include true-attention prefix substage fields.
- [x] Direct scaffold summary includes true-attention prefix totals.
- [x] Profiler harness parses the new fields into `summary.json`.
- [x] V100 build passes.
- [x] V100 direct run passes and records the split.
- [x] Docs/status/temp report are updated.
- [x] Artifacts are copied into `logs/from-cluster/`.
- [x] Sprint artifacts are committed.

## Outcome

Added pre-EP prefix timing around the existing TP/EP direct true-attention
gates and parsed those fields into the permanent profiler summary.

V100 direct validation:

```text
slots: 32
ctx: 262144
decode steps: 2
stream sync: on
generated tok/s decode: 83.265760
continuation tok/s decode: 99.612333
output_head_finite_bad: 0
sum_decode_ms: 768.623264
old broad sum_hc_current_input_ms: 626.823138
```

Pre-EP prefix split across `86` layer-step invocations:

| Field | ms |
|---|---:|
| `sum_pre_ep_compressed_kv_ms` | `228.813152` |
| `sum_pre_ep_attention_projection_ms` | `170.865666` |
| `sum_pre_ep_attention_state_ms` | `105.654904` |
| `sum_pre_ep_hc_current_ms` | `85.249101` |
| `sum_pre_ep_raw_read_ms` | `34.932798` |
| `sum_pre_ep_typed_history_ms` | `1.271677` |
| `sum_pre_ep_attention_output_ms` | `0.000000` |
| `sum_pre_ep_post_attention_ffn_input_ms` | `0.000000` |
| Measured prefix total | `626.787298` |

## Decision

The broad pre-EP timer is now explained by measured prefix stages. The largest
single owner is compressed KV projection/store, then attention projection and
attention state update. HC-current itself is no longer the primary target.

Next sprint should attack compressed-KV projection/store fragmentation first:
reduce BF16/F8 dense-fill and staging work, batch or fuse the emitted-row
stores, and validate with the same direct profiler harness before an HTTP A/B.

Artifacts:

```text
logs/from-cluster/sprint351-true-attn-prefix-breakdown/cluster/
```
