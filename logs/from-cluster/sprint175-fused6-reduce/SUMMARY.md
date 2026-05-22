# Sprint 175 Fused6 Reduce Evidence

Date: 2026-05-22
Cluster pod: `llm/llamacpp-build-8gpu`
Appliance: `/workspace/ds4-appliance-full-tm-gated-s127`
TurboMind library: `/workspace/ds4/build/turbomind-v100-s127/libggml-turbomind.so`

## Build

```bash
cd /workspace/ds4
make ds4_cuda.o \
  tools/ds4-v100-replay \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke \
  CUDA_ARCH=sm_70 -j80
```

Result: passed.

## Correctness

Selected-token smoke passed with expected token `3136`.

Required liveness evidence:

```text
ds4: TurboMind routed executor fused6_reduce shape total_routes=6 active_experts=6 max_routes_per_expert=1
ds4: routed-FFN liveness executor=fused6_reduce total_routes=6 route_expanded_a_half=0 compact_a_half=1 gate_out=elided mid_half=materialized down_routes=elided output_mode=full_sum
ds4: TurboMind down-reduce epilogue selected total_routes=6
```

Full 16-slot/256K scheduler smoke passed.

## Served A/B

Same binary, 16 slots, 256K context, 16 requests x 64 generated tokens,
per-step async + event handoff.

| Mode | Prompt tok/s | Generated tok/s | Continuation tok/s | Token match |
|---|---:|---:|---:|---:|
| control | `20.066988` | `71.349289` | `70.234456` | `16/16` |
| fused6_reduce | `19.959853` | `70.968366` | `69.859485` | `16/16` |

Decision: keep `fused6_reduce` default-off and diagnostic-only.
