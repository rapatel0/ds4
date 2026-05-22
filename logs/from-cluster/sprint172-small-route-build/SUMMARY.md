# Sprint 172 Cluster Summary

## Served Runs

Configuration:

- Model: `/models/DSv4-Flash-256e-fixed.gguf`
- Appliance: `/workspace/ds4-appliance-full-tm-gated-s127`
- Context: `262144`
- Slots: `16`
- Active microbatch: `16`
- Tokens/request: `16`
- Timed requests: `16`
- Async pipeline: `per-step`
- Event handoff: `1`
- TurboMind gated-SiLU: `1`
- TurboMind compact schedule: `1`
- TurboMind down-reduce epilogue: `0`
- TurboMind graph: `0`

| Mode | Generated tok/s | Continuation tok/s | Prompt tok/s | Token match |
|---|---:|---:|---:|---:|
| small-route candidate | `46.550101` | `43.640720` | `52.368864` | `16/16` |
| control repeat | `46.136775` | `43.253227` | `51.903872` | `16/16` |
| small-route repeat | `45.927784` | `43.057298` | `51.668757` | `16/16` |

## Decision

Do not promote. The first run was mildly positive, but the repeat was below
the fresh control repeat. The averaged lift versus nearby controls is only
about `0.4%`, which is run noise for this harness. Keep
`DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=0` as the production default.

Next work should move to a larger execution boundary: persistent routed-FFN or
persistent TP/EP scheduling.
