# Sprint 112 - Fused Appliance Hot-Path Profile

Date: 2026-05-20

## Objective

Capture a warmed served profile of the Sprint 111 fused gate_up appliance and
use it to choose the next implementation target.

## Plan

1. Let the soak harness run an alternate replay wrapper so `nvprof` can wrap the
   appliance server without hand-copying the HTTP benchmark logic.
2. Run the fused appliance with `DS4_V100_CUDA_PROFILER_WINDOW=1` so profiling
   starts only around request generation, not model upload.
3. Record the top GPU kernel buckets and CUDA API buckets.
4. Decide whether Sprint 112 should continue into:
   - persistent/grouped TurboMind expert execution;
   - fused SwiGLU/down scheduling;
   - larger F8 dequant/dot pipeline changes.

## Profile

Fused 8-slot/256K profile:

- appliance: `/workspace/ds4-appliance-full-tm-fused-s111`
- generated throughput: `33.972205 tok/s`
- continuation throughput: `31.848943 tok/s`
- correctness: `8/8` expected-token matches

Top GPU buckets:

| Kernel bucket | Time | Calls | Avg | Share |
|---|---:|---:|---:|---:|
| `arena_f8_e4m3_b128_matmul_rows2_kernel` | `813.77 ms` | `12,341` | `65.940 us` | `41.65%` |
| TurboMind SM70 MXFP4 grouped GEMM | `404.88 ms` | `3,526` | `114.83 us` | `20.72%` |
| `arena_f8_e4m3_b128_matmul_grouped_rows2_ds4_attn_o_kernel` | `252.71 ms` | `1,763` | `143.34 us` | `12.93%` |
| `matmul_f32_kernel` | `95.731 ms` | `3,526` | `27.150 us` | `4.90%` |
| `rms_norm_plain_kernel` | `62.557 ms` | `3,526` | `17.741 us` | `3.20%` |
| `attention_decode_mixed_kernel` | `60.713 ms` | `1,763` | `34.437 us` | `3.11%` |

Top CUDA API buckets:

| API bucket | Time | Calls | Avg | Share |
|---|---:|---:|---:|---:|
| `cudaMemcpy` | `1.31900 s` | `3,526` | `374.08 us` | `69.41%` |
| `cudaLaunchKernel` | `403.97 ms` | `74,433` | `5.427 us` | `21.26%` |
| `cudaSetDevice` | `53.926 ms` | `75,829` | `711 ns` | `2.84%` |
| `cudaDeviceSynchronize` | `41.994 ms` | `325` | `129.21 us` | `2.21%` |

The profile says the next larger target is F8 projection execution, not more
gate/up fusion. F8 row-pair plus DS4 grouped attention-output are `54.58%` of
GPU time after Sprint 111.

## Implementation

Add an opt-in F8 row-pair scale-hoisting variant:

- `DS4_V100_CUDA_F8_WARP_SCALE=1` in the launcher exports
  `DS4_CUDA_F8_WARP_SCALE=1`.
- `arena_f8_block_scale_warp()` loads each E8M0 block scale once per warp and
  broadcasts it with `__shfl_sync`.
- `arena_f8_e4m3_b128_matmul_rows2_warp_scale_kernel` targets the main
  ungrouped source-F8 row-pair path.
- `arena_f8_e4m3_b128_matmul_grouped_rows2_ds4_attn_o_warp_scale_kernel`
  targets the DS4 fixed-shape attention-output-A grouped path.

This is a narrow software-pipeline/fusion-style test: it reduces repeated scale
decode/load work inside the hot scalar F8 dot kernels without changing the pack
format, row ownership, output layout, or reduction order.

## Definition of Done

- [x] Soak harness supports a replay wrapper without changing normal behavior.
- [x] Fused 8-slot/256K served profile is captured on `gpu-01`.
- [x] Profile artifacts are copied under `logs/from-cluster/`.
- [x] The next implementation target is documented with evidence.
- [x] V100 `sm_70` build passes for the guarded F8 warp-scale variant.
- [x] Correctness smokes pass with `DS4_V100_CUDA_F8_WARP_SCALE=1`.
- [x] 8-slot/256K same-binary A/B decides whether to ship warp-scale as default.

## V100 Validation

Cluster target:

- pod: `llamacpp-build-8gpu`
- node: `gpu-01`
- storage: k8s-local `/workspace`
- appliance: `/workspace/ds4-appliance-full-tm-fused-s111`

Build:

```text
CUDA_ARCH=sm_70 make -j80 tools/ds4-v100-replay \
  tests/cuda_source_dtypes_smoke \
  tests/cuda_v100_projection_attention_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke
```

Passed with `DS4_CUDA_F8_WARP_SCALE=1`:

- `cuda_source_dtypes_smoke`
- `cuda_v100_projection_attention_smoke`
- `cuda_v100_full_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 --slots 8`
- `cuda_v100_selected_token_smoke --model /models/DSv4-Flash-256e-fixed.gguf --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 --expected-token-hex 3136`

The selected-token smoke selected token id `926`, text hex `3136`.

## Throughput

Same-binary 8-slot/256K A/B:

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Avg latency | Correctness |
|---|---:|---:|---:|---:|---:|---|
| Control, warp-scale off | 262,144 | 8 | `33.484099` | `31.391343` | `3773.474 ms` | 8/8 |
| Warp-scale on | 262,144 | 8 | `29.009399` | `27.196312` | `4365.094 ms` | 8/8 |

## Decision

Do not default the warp-scale F8 kernels. They are correct, but the 8-slot/256K
production target regressed by about `13.4%`. The likely cause is that the
extra warp shuffle and added instruction dependency cost more than the removed
per-lane E8M0 scale load/decode on V100 for this scalar dot shape.

Keep `DS4_V100_CUDA_F8_WARP_SCALE=0` in launcher and deployment defaults. The
opt-in path can remain as a diagnostic because it is narrow and rollback is a
single environment flag.

The next useful work should not be another tiny scalar F8 tweak. The fused
profile still says F8 projection kernels are the largest bucket, but the
successful Sprint 111 result came from changing a larger boundary. Next target:
either a true CUTLASS/TurboMind-style persistent/tiled F8 projection rewrite, or
deeper TurboMind expert scheduling that changes route-expanded work size and
tensor-core occupancy.

## Artifacts

- `logs/from-cluster/sprint112-fused-profile/`
- `logs/from-cluster/sprint112-f8-warp-scale/`
