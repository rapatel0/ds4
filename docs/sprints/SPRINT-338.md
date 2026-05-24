---
sprint: 338
title: TP/EP Typed KV HTTP A/B
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 338 - TP/EP Typed KV HTTP A/B

## Goal

Measure the serving cost of the typed production KV path against the current
no-typed-KV TP/EP HTTP serving baseline.

## Scope

TP/EP only. No PP/layer-split work. No MTP. This sprint is measurement and
gate hardening, not kernel optimization.

## Definition of Done

- [x] Run a same-shape HTTP control with typed KV gates off.
- [x] Run a same-shape HTTP candidate with typed-history serving enabled.
- [x] Use tokenizer-enabled `/v1/chat/completions`, `32` concurrent requests,
      `32` slots, `256K` context, and resident HC/session state.
- [x] Record generated tok/s, decode tok/s, cache/session state, typed KV
      PASS-line counts, and error status.
- [x] Update status/vision with the A/B result and commit artifacts.

## Expected Decision

If typed KV serving is within noise or modestly slower, keep it as the
correctness-facing path and continue toward the remaining reference semantics.
If it is materially slower, keep the gate opt-in and decide whether direct
typed-row attention reads or staging reduction should be the next optimization.

## Result

Same-shape HTTP A/B:

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
case           http  batch  server tok/s  decode tok/s  typed history
control        32/32 32     260.529425    698.278847    0
typed-history  32/32 32      56.495098     63.381174    1
```

Typed-history evidence:

```text
typed_raw_lines 942
typed_compressed_lines 105
typed_indexer_lines 105
typed_history_lines 898
history_loaded_attn_rows_2 84
history_loaded_indexer_rows_2 84
```

The typed production KV path is operational under 32-way HTTP serving, but the
current staged store/load implementation is a large throughput regression in
this diagnostic shape: about `4.6x` slower by server wall tok/s and about `11x`
slower by decode tok/s. Keep typed KV history opt-in for correctness gates.

## Decision

Do not make typed-history serving the default performance path yet.

The next implementation sprint should reduce the typed-KV staging overhead
instead of adding more serving surface. The likely target is to stop decoding
typed F8 rows back into broad f32 staging for every visible row and move toward
direct typed-row attention reads or a narrower per-layer/per-slot reload cache.

## Artifacts

- `logs/from-cluster/sprint338-typed-kv-http-ab/cluster/summary.tsv`
- `logs/from-cluster/sprint338-typed-kv-http-ab/cluster/control/`
- `logs/from-cluster/sprint338-typed-kv-http-ab/cluster/typed-history/`
