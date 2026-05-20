# Sprint 105 - BF16/F32 Warp-Reduction Probe

Date: 2026-05-20

## Objective

Test whether the Sprint 104 warp-reduction pattern should also be applied to
BF16 output-head and F32 control/router arena matmuls.

## Context

Sprint 104 produced a modest but repeatable improvement by replacing F8 arena
shared-memory tree reductions with warp-shuffle block reductions. The nearby
BF16/F32 arena matmul kernels still used the older shared-memory reduction
shape, so they were a plausible follow-up target with no extra VRAM pressure.

## Implementation Tested

The probe moved the `arena_block_sum_256_f32` helper above the BF16/F32 arena
matmuls and replaced shared-memory tree reductions in:

- `arena_bf16_matmul_kernel`;
- `arena_bf16_matmul_rows_kernel`;
- `arena_f32_matmul_kernel`.

The code change was reverted after benchmarking because the result was not
stable enough to ship as a default.

## V100 Validation

Cluster target: `llamacpp-build-8gpu` on `gpu-01`, using k8s-local
`/workspace`, 80 CPU build parallelism, and the full 8x V100 stack.

Build passed:

```text
make tools/ds4-v100-replay tests/cuda_source_dtypes_smoke \
  tests/cuda_v100_projection_attention_smoke \
  tests/cuda_v100_stage_scheduler_smoke tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke CUDA_ARCH=sm_70 -j80
```

Correctness passed:

- `./tests/cuda_source_dtypes_smoke`
- `./tests/cuda_v100_projection_attention_smoke`
- `./tests/cuda_v100_stage_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-s090 --stage 0 --slots 4`
- `./tests/cuda_v100_full_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-s090 --slots 8`
- `./tests/cuda_v100_selected_token_smoke --model /models/DSv4-Flash-256e-fixed.gguf --appliance-dir /workspace/ds4-appliance-full-tm-s090 --expected-token-hex 3136`

The selected-token smoke selected token id `926`, text hex `3136`.

## Throughput

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Token Match |
|---|---:|---:|---:|---:|---:|
| Sprint 104 F8 warp reduce | 262,144 | 8 | 31.383579 | 29.422106 | 8/8 |
| Sprint 104 F8 warp reduce repeat | 262,144 | 8 | 31.451185 | 29.485486 | 8/8 |
| Sprint 105 BF16/F32 probe | 262,144 | 8 | 31.612471 | 29.636691 | 8/8 |
| Sprint 105 BF16/F32 probe repeat | 262,144 | 8 | 31.479378 | 29.511917 | 8/8 |
| Sprint 104 F8 warp reduce | 1,048,576 | 4 | 20.026385 | 18.774736 | 4/4 |
| Sprint 105 BF16/F32 probe | 1,048,576 | 4 | 20.190395 | 18.928496 | 4/4 |

## Decision

Do not ship this code change. It is correct, but the 8-slot repeat is too close
to the Sprint 104 band to justify changing more reduction order in output/control
paths. Keep Sprint 104 as the committed baseline and move Sprint 106 to a fresh
profile or a larger F8/TurboMind execution-shape change.
