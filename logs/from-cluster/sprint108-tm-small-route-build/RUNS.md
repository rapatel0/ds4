# Sprint 108 Cluster Runs

Date: 2026-05-20

Cluster target:

- pod: `llamacpp-build-8gpu`
- node: `gpu-01`
- repo: `/workspace/ds4`
- appliance: `/workspace/ds4-appliance-full-tm-s090`

## Correctness

| Check | Result |
|---|---|
| TurboMind adapter smoke | PASS |
| Stage scheduler, stage 0, 4 slots | PASS |
| Full scheduler, 8 slots | PASS |
| Selected token, small-route opt-in | PASS, token id `926`, hex `3136` |
| Selected token, rollback | PASS, token id `926`, hex `3136` |
| Default-off config check | PASS, `turbomind_small_route_build=0` |
| Default-off selected token | PASS, token id `926`, hex `3136` |

## Soak Benchmarks

| Log directory | Context | Slots | Small Route | Generated tok/s | Continuation tok/s | Token Match |
|---|---:|---:|---:|---:|---:|---:|
| `soak-8slot-fast` | 262,144 | 8 | on | 31.393301 | 29.431220 | 8/8 |
| `soak-8slot-rollback` | 262,144 | 8 | off | 31.571254 | 29.598051 | 8/8 |
| `soak-8slot-fast-repeat` | 262,144 | 8 | on | 31.759013 | 29.774074 | 8/8 |
| `soak-8slot-rollback-repeat` | 262,144 | 8 | off | 31.794180 | 29.807044 | 8/8 |
| `soak-4slot-fast` | 1,048,576 | 4 | on | 20.249531 | 18.983935 | 4/4 |
| `soak-4slot-rollback` | 1,048,576 | 4 | off | 20.081695 | 18.826589 | 4/4 |
