# TEMP_STATUS_REPORT_052

Date: 2026-05-24

## Topline

Sprint 340 added an opt-in typed KV performance gate that stores production
typed rows but skips same-step current-row reloads.

It works and improves typed-history serving modestly, but typed-history is
still much slower than control. The next bottleneck is likely typed KV store
overhead itself.

## A/B Shape

```text
endpoint: /v1/chat/completions
requests: 32 concurrent
slots: 32
ctx: 262144
tokens/request: 8
diagnostic output head: on
HC persistent state: on
typed candidate: typed-history + skip-current-load
```

## Result

```text
case           http  batch  server tok/s  decode tok/s  client tok/s
control        32/32 32     316.297621    735.600737    104.071414
typed-history  32/32 32      74.383163     86.322558     17.524156
```

Prior typed-history results:

```text
Sprint 338: 56.495098 server tok/s, 63.381174 decode tok/s
Sprint 339: 68.358523 server tok/s, 78.858737 decode tok/s
Sprint 340: 74.383163 server tok/s, 86.322558 decode tok/s
```

## Typed KV Evidence

```text
typed_raw_lines 942
typed_compressed_lines 105
typed_indexer_lines 105
typed_history_lines 898
history_loaded_attn_rows_2 84
history_loaded_indexer_rows_2 84
history_reloaded_attn_rows_nonzero 105
history_reloaded_indexer_rows_nonzero 105
typed_current_load_0 1152
typed_current_load_1 0
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

Keep `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_CURRENT_LOAD=1` as the
typed-history performance candidate. It is better than the strict same-step
roundtrip path, but still not production-default fast enough.

Next sprint should measure and reduce typed KV store overhead by family:
raw-SWA, compressed-attention, and indexer. The likely implementation direction
is batched row stores or fused producer-store kernels rather than more reload
avoidance.

## Artifact

- `logs/from-cluster/sprint340-skip-current-typed-load/cluster/summary.tsv`
