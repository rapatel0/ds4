# TEMP_STATUS_REPORT_050

Date: 2026-05-24

## Topline

Sprint 338 ran a same-shape TP/EP HTTP serving A/B for typed production KV.

Typed KV serving is operational at `32` concurrent chat requests, `32` slots,
and `256K` context, but the current staged typed-history implementation is
materially slower than the no-typed-KV serving baseline.

## A/B Shape

```text
endpoint: /v1/chat/completions
requests: 32 concurrent
slots: 32
ctx: 262144
tokens/request: 8
diagnostic output head: on
HC persistent state: on
```

## Result

```text
case           http  batch  server tok/s  decode tok/s  continuation tok/s  continuation decode tok/s
control        32/32 32     260.529425    698.278847    260.977482         696.627630
typed-history  32/32 32      56.495098     63.381174     55.835127          62.538741
```

Control client-side generated throughput was `86.144900` tok/s. The first
harness version tripped while parsing one typed response's UTF-8 text payload
after all candidate responses were already captured, so typed client elapsed
was not recorded. The harness is now hardened to parse response bodies with
replacement decoding.

Typed-history server evidence:

```text
typed_raw_lines 942
typed_compressed_lines 105
typed_indexer_lines 105
typed_history_lines 898
history_loaded_attn_rows_2 84
history_loaded_indexer_rows_2 84
```

Final GPU state:

```text
0, 0, 32495, 0
1, 0, 32495, 0
2, 0, 32495, 0
3, 0, 32495, 0
4, 0, 32495, 0
5, 0, 32495, 0
6, 0, 32495, 0
7, 0, 32495, 0
```

## Decision

Keep typed-history serving opt-in. It is the correctness-facing path, not the
performance default yet.

The next sprint should reduce typed KV staging overhead. The likely direction
is direct typed-row attention reads, or at least a narrower reload cache that
does not decode every visible F8 row into broad f32 staging on every layer
step.

## Artifact

- `logs/from-cluster/sprint338-typed-kv-http-ab/cluster/summary.tsv`
