# TEMP_STATUS_REPORT_051

Date: 2026-05-24

## Topline

Sprint 339 added a bounded-row staging cache for typed compressed-history
reloads.

The cache works: the typed-history serving run showed `reloaded_attn_rows=0`
and `reloaded_indexer_rows=0` on all `899` history lines while still reporting
visible loaded rows. Throughput improved modestly, but typed-history serving
is still much slower than control.

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
case           http  batch  server tok/s  decode tok/s  client tok/s
control        32/32 32     311.293794    735.203733    101.079973
typed-history  32/32 32      68.358523     78.858737     18.295122
```

Sprint 338 typed-history baseline:

```text
server tok/s: 56.495098
decode tok/s: 63.381174
```

Sprint 339 improved typed-history server wall throughput by about `21%`, but
the remaining gap is still large.

## Typed KV Evidence

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

The repeated compressed-history reload was not the main remaining bottleneck.
The next sprint should remove same-step typed KV roundtrips for current rows:
store the production typed row, but keep the already-computed f32 row in
staging for immediate attention instead of loading it back through the typed
runtime in the same layer step.

## Artifact

- `logs/from-cluster/sprint339-typed-history-reload-cache/cluster/summary.tsv`
