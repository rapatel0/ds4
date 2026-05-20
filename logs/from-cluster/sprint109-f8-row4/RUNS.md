# Sprint 109 Cluster Runs

Date: 2026-05-20

Cluster target:

- pod: `llamacpp-build-8gpu`
- node: `gpu-01`
- repo: `/workspace/ds4`
- appliance: `/workspace/ds4-appliance-full-tm-s090`

## Correctness

| Check | Result |
|---|---|
| Config check, row4 on | PASS, `cuda_f8_row4=1` |
| Source dtype smoke, row4 on | PASS |
| Projection attention smoke, row4 on | PASS |
| Full scheduler, 8 slots, row4 on | PASS |
| Selected token, row4 on | PASS, token id `926`, hex `3136` |

## Soak Benchmarks

| Log directory | Context | Slots | Row4 | Generated tok/s | Continuation tok/s | Token Match |
|---|---:|---:|---:|---:|---:|---:|
| `soak-8slot-row4` | 262,144 | 8 | on | 30.998275 | 29.060883 | 8/8 |
| `soak-8slot-row2` | 262,144 | 8 | off | 31.380225 | 29.418961 | 8/8 |
| `soak-4slot-row4` | 1,048,576 | 4 | on | 19.898462 | 18.654808 | 4/4 |
| `soak-4slot-row2` | 1,048,576 | 4 | off | 20.041787 | 18.789175 | 4/4 |

## Decision

Row4 is correct but slower in both measured tiers. Keep it off by default.
