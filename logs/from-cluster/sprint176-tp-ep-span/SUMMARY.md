# Sprint 176 TP/EP Span Evidence

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
make tools/ds4-v100-appliance-pack \
  ds4_v100_layer_execute.o \
  ds4_v100_scheduler.o \
  tools/ds4-v100-replay \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke \
  CUDA_ARCH=sm_70 -j80
```

Result: passed.

## Pack

Final overlay was generated with `--layer 3 --layer-count 2 --emit-tp-split --tp-split-only`.

```text
tm_rows=8
tm_weight_bytes=6442450944
tm_scale_bytes=402653184
gpu0.weights bytes=3422552064
gpu3.weights bytes=3422552064
```

The earlier duplicate-full-weight pack exceeded the available 32 GiB GPU0 VRAM
when overlaid with the base appliance, so it was replaced by the TP-only
overlay format.

## Correctness

```text
cuda_v100_stage_scheduler_smoke: stage=0 gpu=0 layers=0-5 executed=6 tm_layers=6 tp2_layers=2 ... ok
cuda_v100_selected_token_smoke: prompt_tokens=18 selected=926 ... expected=3136 ... ok
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=16 layers=43 tm_layers=43 ... ok
```

## Served A/B

Same binary, 16 slots, 256K context, 16 requests x 64 generated tokens,
per-step async + event handoff.

| Mode | Prompt tok/s | Generated tok/s | Continuation tok/s | Token match |
|---|---:|---:|---:|---:|
| control | `20.110142` | `71.502728` | `70.385497` | `16/16` |
| TP2 span verbose | `18.420008` | `65.493363` | `64.470029` | `16/16` |
| TP2 span quiet | `18.644305` | `66.290863` | `65.255068` | `16/16` |

Decision: TP2 span remains default-off and diagnostic-only.
