# TEMP Status Report 375: Async Output Gate

Date: 2026-05-25

## Topline

Sprint 375 implemented and tested `--async-output-gate`. It is correct but not
promotable as a default.

The gate removes early output-head `cudaDeviceSynchronize` calls and replaces
them with stream/event sequencing, but the current serving loop still must wait
for selected-token D2H before seeding the next decode step.

## Best Current Metrics

Direct 2-step smoke, `32` slots / `256K`:

| Mode | First token | Output checksum | Gen decode tok/s | Output-head ms | Device syncs | Event syncs |
|---|---:|---:|---:|---:|---:|---:|
| control | 98751 | 81959669916 | 88.889462 | 9.459974 | 26 | 0 |
| async-output | 98751 | 81959669916 | 88.744811 | 8.173478 | 0 | 8 |

HTTP chat matrix, `32` active requests / `32` slots / `256K` /
`32` generated tokens/request:

| Mode | HTTP 200 | First token | Output checksum | Client tok/s | Server decode tok/s | Avg GPU util | Max GPU util |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 32/32 | 89340 | 101896170076 | 42.000194 | 99.476540 | 8.212209% | 40.0% |
| async-output | 32/32 | 89340 | 101896170076 | 41.286901 | 93.764276 | 8.204545% | 39.0% |

## Decision

REJECT as a default. Keep opt-in for Sprint 376 investigation.

The result proves the output-head sync audit is working, but the HTTP topline
and GPU utilization are lower. This does not validate async output as a
serving optimization. It remains useful to identify which host waits are
outside the `run_one_step` graph-capture target.

## Artifacts

```text
logs/from-cluster/sprint375-async-output/direct-smoke
logs/from-cluster/sprint375-async-output/matrix
```

## Next

Sprint 376 should start with a read-only CUDA graph capture audit of the
remaining `run_one_step` host waits. Do not assume async output solved graph
eligibility; it only moved the output-head path to event sequencing and showed
the CPU selected-token dependency explicitly.
