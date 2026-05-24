---
sprint: 339
title: TP/EP Typed History Reload Cache
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 339 - TP/EP Typed History Reload Cache

## Goal

Reduce the Sprint 338 typed-history serving regression by avoiding redundant
typed F8 to f32 history reloads when a bounded compressed/indexer row is
already staged for the same source position.

## Scope

TP/EP only. No PP/layer-split work. No MTP. This sprint keeps the existing
bounded f32 staging surface; it does not implement direct typed-row attention
reads yet.

## Definition of Done

- [x] Track staged typed compressed-attention row positions.
- [x] Track staged typed ratio-4 indexer row positions.
- [x] Skip typed runtime row loads when the bounded row is already staged for
      the requested source position.
- [x] Preserve `loaded_*_rows` serving evidence while adding `reloaded_*_rows`
      counters.
- [x] Build on the V100 pod.
- [x] Run the Sprint 338 A/B harness and record whether typed-history serving
      throughput improves.

## Outcome

Added per-rank bounded-row staging markers:

- `attn_comp_row_loaded_layers`
- `attn_comp_row_loaded_position_layers`
- `index_comp_row_loaded_layers`
- `index_comp_row_loaded_position_layers`

The history reload gate now skips runtime F8-to-f32 row loads when the bounded
row is already staged for the same source position. The log keeps
`loaded_attn_rows` / `loaded_indexer_rows` as "available staged rows" and adds
`reloaded_attn_rows` / `reloaded_indexer_rows` for actual runtime reloads.

## Validation

Build on `llm/llamacpp-build-8gpu`:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS.

Same-shape A/B as Sprint 338:

```text
endpoint: /v1/chat/completions
requests: 32 concurrent
slots: 32
ctx: 262144
tokens/request: 8
diagnostic output head: on
HC persistent state: on
```

Summary:

```text
case           http  batch  server tok/s  decode tok/s  client tok/s
control        32/32 32     311.293794    735.203733    101.079973
typed-history  32/32 32      68.358523     78.858737     18.295122
```

Typed-history evidence:

```text
typed_raw_lines 943
typed_compressed_lines 105
typed_indexer_lines 105
typed_history_lines 899
history_loaded_attn_rows_2 84
history_loaded_indexer_rows_2 84
reloaded_attn_rows 0 for all 899 history lines
reloaded_indexer_rows 0 for all 899 history lines
```

Final GPU state was idle: all devices reported `0 MiB` used.

## Decision

The cache works: history reloads were skipped. Throughput improved from Sprint
338's `56.495098` server wall tok/s to `68.358523`, but the path remains about
`4.6x` slower than the same-run control and about `9.3x` slower by decode tok/s.

The dominant cost is no longer repeated compressed-history reload. The next
sprint should remove diagnostic current-row roundtrips from the hot path:
typed raw-SWA and emitted compressed/indexer rows should store the production
typed KV row while keeping the already-computed f32 staging row for immediate
attention, instead of store-then-load roundtripping through typed KV in the
same layer step.

## Artifacts

- `logs/from-cluster/sprint339-typed-history-reload-cache/cluster/summary.tsv`
- `logs/from-cluster/sprint339-typed-history-reload-cache/cluster/control/`
- `logs/from-cluster/sprint339-typed-history-reload-cache/cluster/typed-history/`
