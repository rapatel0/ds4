# TEMP_STATUS_REPORT_055

Date: 2026-05-24

## Topline

Sprint 343 batched the typed KV row API across slots.

Result: batching helps, but it does not close the gap. Typed-batch-rows-quiet
improved typed serving from `73.452667` to `79.984163` server tok/s and from
`86.332914` to `95.624885` decode tok/s.

## Shape

```text
endpoint: /v1/chat/completions
requests: 32 concurrent
slots: 32
ctx: 262144
tokens/request: 8
diagnostic output head: on
HC persistent state: on
typed candidates: skip-current-load on
```

## Results

```text
case                    server tok/s  decode tok/s  quiet  batch_rows
control                 303.282600    735.908031    0      0
typed-history            68.450954     79.760324    0      0
typed-quiet              73.452667     86.332914    1      0
typed-batch-rows-quiet   79.984163     95.624885    1      1
```

## Interpretation

Batched row calls are the right direction:

- wall throughput improved `+8.9%` versus typed-quiet
- decode throughput improved `+10.8%` versus typed-quiet
- HTTP correctness stayed `32/32`
- coalesced batch stayed `32`

But this does not explain most of the typed-KV regression. The remaining gap
to control is still roughly:

- `3.8x` by wall server tok/s
- `7.7x` by decode tok/s

The next likely bottleneck is broad device synchronization around typed row
work. The row API is now batched enough that the next sprint should make those
stores/history loads stream-ordered rather than device-synchronized.

## Final GPU State

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

## Artifact

- `logs/from-cluster/sprint343-typed-kv-batch-rows/cluster/summary.tsv`
