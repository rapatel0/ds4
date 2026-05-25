---
sprint: 352
title: TP/EP Compressed-KV Internal Timing
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 352 - TP/EP Compressed-KV Internal Timing

## Overview

Sprint 351 showed compressed KV projection/store is the largest visible
pre-EP stage at `228.813152 ms` across the 32-slot / 256K / 2-step direct
run. That stage still contains several different costs: dense input fill,
F8 dense projections, rank-to-full gathers, compressor state/store/emit,
typed KV store/load boundaries, ratio-4 indexer work, and state shifts.

This sprint makes that internal breakdown measurable and adds permanent
profiler switches for the existing compressed/indexer typed-store suppression
gates.

No PP/layer-split work. No MTP. No default semantic change.

## Implementation

1. Add passive timing fields to the existing `tp_ep_compressed_kv_projection`
   line:
   - attention input fill
   - attention dense/stat path
   - attention gather-to-full
   - attention state/store/emit
   - attention typed KV store/load boundary
   - indexer input fill
   - indexer dense/stat path
   - indexer gather/RoPE
   - indexer state/store/emit
   - indexer typed KV store/load/score boundary
   - compact reference diff
   - ratio-4 shift
2. Parse and sum these fields in `tools/ds4-v100-tp-ep-profile.py`.
3. Add direct profiler flags for:
   - `--skip-compressed-store`
   - `--skip-indexer-store`

## Verification

- Local syntax checks pass.
- V100 `sm_70` build passes.
- V100 direct 32-slot / 256K / 2-step baseline passes and records internal
  compressed-KV timing.
- V100 direct store-suppression variant passes or is rejected with concrete
  failure evidence.

## Definition of Done

- [x] Compressed-KV layer rows include internal timing fields.
- [x] Profiler summary includes summed compressed-KV timing fields.
- [x] Profiler supports compressed/indexer store-suppression flags.
- [x] V100 build passes.
- [x] V100 direct baseline passes with new fields.
- [x] V100 direct store-suppression test runs.
- [x] Docs/status/temp report are updated.
- [x] Artifacts are copied into `logs/from-cluster/`.
- [x] Sprint artifacts are committed.

## Outcome

Added internal timers to `tp_ep_compressed_kv_projection` and parser support
for summed compressed-KV fields in `summary.json`. Also added
`--position`, `--skip-compressed-store`, and `--skip-indexer-store` to the
direct profiler harness.

The original direct profile default at `position=100000` does not emit
compressed rows. The emitted-row run at `position=262143` correctly exercises
compressed and indexer typed stores, but must be one token because a second
step would advance to position `262144`, outside the configured `256K`
context. The attempted two-token emitted run failed for that reason after
`43` pass invocations, with `tp_runtime_dense_kv_slice_failed`.

Emitted-row one-token V100 baseline:

```text
slots: 32
ctx: 262144
position: 262143
decode steps: 1
returncode: 0
generated tok/s decode: 81.647302
sum_decode_ms: 391.929670
sum_pre_ep_compressed_kv_ms: 129.990107
compressed_kv_emitted_layers: 41
```

Internal compressed-KV baseline totals:

| Field | ms |
|---|---:|
| `compressed_kv_sum_indexer_dense_ms` | `36.615896` |
| `compressed_kv_sum_attn_dense_ms` | `24.659453` |
| `compressed_kv_sum_attn_state_emit_ms` | `24.362932` |
| `compressed_kv_sum_attn_input_fill_ms` | `12.725079` |
| `compressed_kv_sum_indexer_state_emit_ms` | `9.007686` |
| `compressed_kv_sum_indexer_gather_rope_ms` | `5.327741` |
| `compressed_kv_sum_indexer_typed_score_ms` | `4.916388` |
| `compressed_kv_sum_attn_gather_ms` | `4.428356` |
| `compressed_kv_sum_indexer_input_fill_ms` | `4.051283` |
| `compressed_kv_sum_attn_typed_ms` | `2.161061` |
| `compressed_kv_sum_ratio_shift_ms` | `1.272814` |
| `compressed_kv_sum_ms` | `129.536775` |

Store-suppression variant:

```text
flags: --skip-compressed-store --skip-indexer-store
returncode: 0
generated tok/s decode: 81.733945
sum_decode_ms: 391.514201
sum_pre_ep_compressed_kv_ms: 128.338783
compressed_kv_sum_ms: 127.895362
```

## Decision

Typed compressed/indexer stores are not the main bottleneck at this shape.
Suppressing both stores changes generated decode by only `+0.11%` and reduces
the compressed-KV stage by about `1.65 ms` for the one-token emitted run.

The dominant compressed-KV costs are dense projection and compressor
state/emit work:

- indexer dense: `36.6 ms`
- attention dense: `24.7 ms`
- attention state/emit: `24.4 ms`
- input fills: `16.8 ms` combined
- indexer state/emit: `9.0 ms`

Next sprint should target a fused or shared-fill compressor/indexer input path
and/or a fused compressor state+emit path, not typed KV store suppression.

Artifacts:

```text
logs/from-cluster/sprint352-compressed-kv-internal/cluster/
logs/from-cluster/sprint352-compressed-kv-emitted-1tok/cluster/
```
