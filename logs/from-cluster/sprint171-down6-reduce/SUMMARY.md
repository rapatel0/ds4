# Sprint 171 Cluster Summary

## Build

- Pod: `llm/llamacpp-build-8gpu` on `gpu-01`
- TurboMind build: passed with `cmake --build build/turbomind-v100-s127 --target ggml-turbomind -j80`
- Replay build: passed with `make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay`
- Exported symbol:
  `00000000000331e0 T ggml_turbomind_ds4_mxfp4_down_6_m16_reduce`

## Smoke

Direct replay selected the new path:

```text
ds4: TurboMind down-reduce epilogue selected total_routes=6
```

## Served A/B

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

| Mode | Generated tok/s | Continuation tok/s | Prompt tok/s | Token match |
|---|---:|---:|---:|---:|
| control | `45.941120` | `43.069800` | `51.683760` | `16/16` |
| down6 reduce | `43.887560` | `41.144588` | `49.373505` | `16/16` |

## Decision

Reject for promotion. The 6-route down-reduce epilogue is correct but regresses
the served 16-slot/256K target by roughly `4.5%`. Keep
`DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=0` for production. The next sprint
should move to a persistent gate/up + down routed-FFN executor or a broader
persistent TP/EP boundary.
