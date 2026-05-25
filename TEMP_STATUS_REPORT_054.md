# TEMP_STATUS_REPORT_054

Date: 2026-05-24

## Topline

Sprint 342 tested whether typed KV per-layer PASS logging was the main TP/EP
serving bottleneck.

Result: it was not. Quiet mode removed the typed log flood but improved typed
serving only from `73.427107` to `75.284862` server tok/s.

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
case           server tok/s  decode tok/s  typed lines  quiet
control        309.202473    730.769885    0            0
typed-history   73.427107     85.479279    2058         0
typed-quiet     75.284862     87.627420    0            1
```

Verbose typed evidence:

```text
raw lines:        946
compressed lines: 105
indexer lines:    105
history lines:    902
loaded rows == 2: 84 attn, 84 indexer
```

Quiet candidate evidence:

```text
typed_quiet_meta: 1
typed PASS lines: 0
HTTP 200: 32/32
coalesced batch: 32
```

## Interpretation

Suppressing typed PASS logs gives only about `+2.5%` wall/decode throughput.
That rules out stdout formatting as the dominant remaining regression.

The next likely bottleneck is the typed row API shape itself:

- one store/load call per slot per row family
- one kernel launch per rank per row operation inside the row API
- broad per-device synchronizations around typed row operations
- typed-history bookkeeping that is still in the hot layer loop

The next sprint should batch typed row operations across slots/families or add
a serving-safe asynchronous typed-row path, then re-run this same HTTP A/B.

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

- `logs/from-cluster/sprint342-typed-kv-quiet-overhead/cluster/summary.tsv`
