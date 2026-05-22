# Sprint 179 Compact No-Host-Sync Evidence

Date: 2026-05-22
Cluster pod: `llm/llamacpp-build-8gpu`

## Artifacts

- Base appliance: `/workspace/ds4-appliance-full-tm-gated-s127`
- TurboMind library: `/workspace/ds4/build/turbomind-v100-s127/libggml-turbomind.so`

## Build

```bash
cd /workspace/ds4
make ds4_cuda.o tools/ds4-v100-replay \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke \
  CUDA_ARCH=sm_70 -j80
```

Result: passed.

## Correctness

```text
cuda_v100_selected_token_smoke: prompt_tokens=18 selected=926 ... expected=3136 ... ok
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=16 layers=43 tm_layers=43 ... ok
```

The gate selected in the real fused6_reduce path:

```text
ds4: TurboMind routed executor fused6_reduce no_host_sync total_routes=6 compact_groups=6
ds4: routed-FFN liveness executor=fused6_reduce total_routes=6 route_expanded_a_half=0 compact_a_half=1 gate_out=elided mid_half=materialized down_routes=elided output_mode=full_sum
ds4: TurboMind down-reduce epilogue selected total_routes=6
```

## Served A/B

Same binary, 16 slots, 256K context, 16 requests x 64 generated tokens,
per-step async + event handoff.

| Mode | Prompt tok/s | Generated tok/s | Continuation tok/s | Token match |
|---|---:|---:|---:|---:|
| production control | `20.083371` | `71.407542` | `70.291799` | `16/16` |
| fused6_reduce host-sync | `19.730169` | `70.151713` | `69.055593` | `16/16` |
| fused6_reduce no-host-sync verbose | `19.210761` | `68.304927` | `67.237662` | `16/16` |
| fused6_reduce no-host-sync quiet | `19.151022` | `68.092522` | `67.028577` | `16/16` |

Decision: `DS4_V100_TURBOMIND_COMPACT_NO_HOST_SYNC` stays default-off and
diagnostic-only. Empty compact groups cost more than the avoided host
synchronization for the six-route served shape.
