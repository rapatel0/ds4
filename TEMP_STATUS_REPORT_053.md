# TEMP_STATUS_REPORT_053

Date: 2026-05-24

## Topline

Sprint 341 isolated typed KV store-family cost.

Result: typed KV stores are not the primary remaining bottleneck. Disabling
all typed stores improved the typed candidate only from `75.577828` to
`79.039985` server tok/s.

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
case                       server tok/s  decode tok/s  typed stores
control                    308.223158    722.800920    0
typed-history               75.577828     87.938497    1156
typed-no-raw-store          77.304656     91.097009     210
typed-no-compressed-store   74.992652     87.648516    1051
typed-no-indexer-store      73.921469     86.162083    1051
typed-no-stores             79.039985     93.242875       0
```

## Interpretation

Store suppression gives only a small gain:

- no raw store: `+2.3%` wall tok/s vs typed baseline
- no compressed store: roughly flat
- no indexer store: slightly slower
- no stores: `+4.6%` wall tok/s vs typed baseline

So the remaining typed path regression is probably not row store bandwidth
itself. More likely sources are device synchronizations, verbose PASS logging,
typed-history bookkeeping, or other diagnostic scaffolding in the hot serving
loop.

## Note

The first full run completed control and typed baseline, then `no-raw-store`
hit CUDA OOM during startup after rapid serial process teardown. The harness
was hardened with GPU-idle cooldown and the store-family variants were resumed
successfully.

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

## Next

Sprint 342 should strip or gate typed diagnostic overhead:

- avoid per-row/layer PASS logging in serving measurements
- remove unnecessary `cudaDeviceSynchronize` calls from typed store/load gates
- measure whether typed bookkeeping alone is causing the remaining gap

## Artifact

- `logs/from-cluster/sprint341-typed-store-family-cost/cluster/combined-summary.tsv`
