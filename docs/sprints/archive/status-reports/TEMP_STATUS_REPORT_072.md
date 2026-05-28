# TEMP Status Report 072 - Sprint 360 Launcher Validation

Date: 2026-05-25

## Current Focus

TP/EP serving operational validation. Sprint 360 checks that the pool-norm
default promoted in Sprint 359 works through `tools/ds4-v100-run-appliance.sh`,
not only the direct/profile harness.

## Result

Launcher `--print-command` includes the pool-norm gate by default:

```text
--true-ds4-compressed-kv-fused-pool-norm-gate
```

This was verified without setting
`DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM`.

## Launcher HTTP Run

Shape:

```text
serve mode: tp-ep
slots: 32
context: 256K
position: 262112
tokens/request: 32
requests: 32
```

Result:

| Metric | Value |
|---|---:|
| HTTP 200 | 32/32 |
| client generated tok/s | 73.289956 |
| first token | 109328 |
| compressed projection lines | 1375 |
| fused pool-norm rows | 187 |
| fused input-fill rows | 0 |

The first bare launcher attempt returned HTTP 500 because I did not pass the
full true-attention typed-KV gate set that the profile harness uses. The valid
run includes those required serving gates but leaves the pool-norm env unset,
so it still proves the promoted default.

## Decision

Pool-norm default promotion is validated through the launcher path.

Next best step:

1. rerun the full chat/completions topline with the promoted default, or
2. continue implementation on the remaining compressed state/emit fusion.

Artifacts:

```text
logs/from-cluster/sprint360-launcher-pool-norm-default/
logs/from-cluster/sprint360-launcher-pool-norm-default-valid/
```
