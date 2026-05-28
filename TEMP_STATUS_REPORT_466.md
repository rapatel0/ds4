# TEMP Status Report 466

## Focus

TP/EP graph-event-order first-divergence diagnostics.

## Added

- `--decode-stage-checksum-gate`
- `DS4_V100_TP_EP_DECODE_STAGE_CHECKSUM`
- Profile and A/B harness flags for control/candidate stage checksums.

## Evidence

Heavy diagnostic artifact:

```text
/localpool/ds4/workspace/logs/s466-stage-checksum-s8-t1
```

First overlapping mismatch:

```text
step=0 layer=0 stage=hc_current tensor=current_shard rank=0
control checksum:   260522477
candidate checksum: 264538364
```

Lite completed artifact:

```text
/localpool/ds4/workspace/logs/s466-stage-checksum-lite-s8-t1
```

Result:

```text
response parity: 8/8 matched
stage checksum keys: 6880 control, 6880 candidate
checksum mismatches: 0
```

## Read

Per-stage synchronization repairs graph-event-order correctness. The graph
failure is an ordering/race issue, not an output-head issue and not a tensor math
format issue. The earliest observed bad state is HC-current layer 0
`current_shard`.

## Next

Test the minimal HC-current-only synchronization point, then replace that host
sync with precise graph-safe event/NCCL ordering.
