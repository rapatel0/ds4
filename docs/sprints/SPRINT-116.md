# Sprint 116 - Attention Projection F8 HMMA Batch

Date: 2026-05-21

## Objective

Profile the Sprint 115 default pair-SwiGLU HMMA appliance, then use the
remaining hot buckets to ship the next measured default improvement.

## Profile Finding

The warmed 8-slot/256K `nvprof` run with pair-SwiGLU HMMA enabled and
shared-down HMMA disabled still showed the ungrouped F8 row-pair kernel as the
dominant device bucket:

| Bucket | GPU time | Calls | Avg |
|---|---:|---:|---:|
| `arena_f8_e4m3_b128_matmul_rows2_kernel` | 41.65% | 12,341 | 66.087 us |
| TurboMind SM70 MXFP4 grouped GEMM | 20.72% | 3,526 | 115.06 us |
| DS4 grouped attention-output-A rows2 | 12.94% | 1,763 | 143.68 us |
| F32 matmul | 4.90% | 3,526 | 27.194 us |
| attention decode mixed | 3.11% | 1,763 | 34.495 us |

CUDA API accounting was still dominated by many small synchronous operations:
`cudaMemcpy` was 68.65% of API time and `cudaLaunchKernel` was 21.94%, while
GPU-side memcpy time was small. This says the next worthwhile target is still
device kernel shape and launch count, not disk, host RAM, or bulk NVLink/PCIe
traffic.

Artifacts:

- `logs/from-cluster/sprint116-pair-hmma-profile/`

## Implementation

Sprint 116 adds an opt-in Volta WMMA/HMMA batch path for the fixed DS4
attention projection shapes:

| Projection | Shape | Entry point |
|---|---:|---|
| `attn_q_a` | rows 1024, cols 4096 | `ds4_gpu_arena_f8_e4m3_b128_matmul_batch_ptr_table_f32` |
| `attn_kv_latent` | rows 512, cols 4096 | `ds4_gpu_arena_f8_e4m3_b128_matmul_batch_ptr_table_f32` |
| `attn_q_b` | rows 32768, cols 1024 | `ds4_gpu_arena_f8_e4m3_b128_matmul_batch_f32` |

The kernels tile active slots as the WMMA M dimension and output rows as the N
dimension. They load F32 activations, unpack F8_E4M3_B128 weights and E8M0
scales into FP16 tiles inside the kernel, accumulate through Volta HMMA into
FP32, and store token-major F32 outputs.

Runtime controls:

- `DS4_V100_ENABLE_BATCH_ATTN_PROJ=1` enables batched attention projections for
  active 4/8-slot batches.
- `DS4_V100_CUDA_F8_HMMA_ATTN_BATCH=1` selects the Sprint 116 HMMA kernels for
  the fixed DS4 attention projection shapes.
- `DS4_V100_ENABLE_BATCH_ATTN_PROJ_ANY=1` is an explicit escape hatch for
  non-4/8 active-slot experiments; by default 2-slot configs fall back to the
  older per-slot attention projection path instead of using scalar batching.

Decision: ship both batch-attention projection flags as launcher defaults for
the measured 4/8-slot serving modes. Roll back with either
`DS4_V100_ENABLE_BATCH_ATTN_PROJ=0` or `DS4_V100_CUDA_F8_HMMA_ATTN_BATCH=0`.

## Validation

Build:

- `CUDA_ARCH=sm_70 make -j80 tests/cuda_f8_hmma_attn_batch_smoke tests/cuda_source_dtypes_smoke tests/cuda_v100_projection_attention_smoke tests/cuda_v100_full_scheduler_smoke tests/cuda_v100_selected_token_smoke tools/ds4-v100-replay`

Correctness:

- `tests/cuda_f8_hmma_attn_batch_smoke`: passed.
- `tests/cuda_source_dtypes_smoke` with attention HMMA flag: passed.
- `tests/cuda_v100_projection_attention_smoke` with attention HMMA flag:
  passed.
- `tests/cuda_v100_full_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 --model /models/DSv4-Flash-256e-fixed.gguf --slots 8 --expect-tm-layers 43` with attention HMMA flag:
  passed, `tm_layers=43`.
- `tests/cuda_v100_selected_token_smoke --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 --model /models/DSv4-Flash-256e-fixed.gguf --expected-token-hex 3136` with attention HMMA flag:
  passed, selected token id `926`, logit `35.202522`.
- `tools/ds4-v100-run-appliance.sh --check`: reports
  `cuda_f8_hmma_attn_batch=1` and `batch_attn_proj=1`.

## Throughput

Same-binary served A/B:

| Mode | Context | Slots | Batch attn HMMA | Generated tok/s | Continuation tok/s | Token match |
|---|---:|---:|---|---:|---:|---:|
| Control | 262,144 | 8 | off | `33.380614` | `31.294325` | 8/8 |
| Candidate | 262,144 | 8 | on | `33.697698` | `31.591592` | 8/8 |
| Control | 1,048,576 | 4 | off | `21.333447` | `20.000107` | 4/4 |
| Candidate | 1,048,576 | 4 | on | `21.469010` | `20.127197` | 4/4 |

Promoted-default smoke, with launcher defaults and no explicit attention-batch
flags:

| Context | Slots | Generated tok/s | Continuation tok/s | Token match |
|---:|---:|---:|---:|---:|
| 262,144 | 8 | `33.540586` | `31.444300` | 8/8 |

Artifacts:

- `logs/from-cluster/sprint116-attn-hmma/`

## Definition of Done

- [x] Pair-HMMA default served profile is captured on the V100 node.
- [x] Profile artifacts are copied under `logs/from-cluster/`.
- [x] Top GPU/API buckets are summarized.
- [x] Next implementation target is documented with a rollback flag and focused
      validation plan.
- [x] Attention projection HMMA batch path is implemented behind rollback
      flags.
- [x] Focused CUDA smoke passes for `q_a`, `kv_latent`, and `q_b` shapes.
- [x] Full scheduler and selected-token oracle pass on the full TurboMind
      fused appliance.
- [x] 8-slot/256K and 4-slot/1M served A/B both improve with full token match.
