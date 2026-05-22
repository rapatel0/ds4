# Sprint 178 TP/EP Parallel Halves Evidence

Date: 2026-05-22
Cluster pod: `llm/llamacpp-build-8gpu`

## Artifacts

- Base appliance: `/workspace/ds4-appliance-full-tm-gated-s127`
- TP overlay pack: `/workspace/ds4-tp-span-s176`
- Combined appliance view: `/workspace/ds4/logs/sprint176-tp-ep-span/appliance-tp-span-combined`
- TurboMind library: `/workspace/ds4/build/turbomind-v100-s127/libggml-turbomind.so`

## Build

```bash
cd /workspace/ds4
make ds4_v100_layer_execute.o ds4_v100_scheduler.o ds4_v100_replay.o \
  tools/ds4-v100-replay \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke \
  CUDA_ARCH=sm_70 -j80
```

Result: passed.

## Correctness

```text
cuda_v100_stage_scheduler_smoke: stage=0 gpu=0 layers=0-5 executed=6 tm_layers=6 tp2_layers=2 ... ok
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=16 layers=43 tm_layers=43 ... ok
cuda_v100_selected_token_smoke: prompt_tokens=18 selected=926 ... expected=3136 ... ok
```

The stage smoke logs show the new gate executing:

```text
parallel_halves=1
```

## Served A/B

Same binary, 16 slots, 256K context, 16 requests x 64 generated tokens,
per-step async + event handoff.

| Mode | Prompt tok/s | Generated tok/s | Continuation tok/s | Token match |
|---|---:|---:|---:|---:|
| no TP/EP control | `20.053070` | `71.299803` | `70.185744` | `16/16` |
| TP2 span sequential | `18.624110` | `66.219059` | `65.184386` | `16/16` |
| TP2 span parallel halves | `18.670979` | `66.385703` | `65.348426` | `16/16` |

Decision: `DS4_V100_TP_EP_PARALLEL_HALVES` stays default-off and diagnostic-only.
It recovers only about `0.25%` over the sequential TP2 span and remains about
`6.9%` slower than the no-TP production path.
