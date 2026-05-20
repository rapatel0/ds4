# Sprint 061 Report: Batched Shared F8 Expert Path

## Result

`SHIP`, with the shared F8 batch path kept opt-in.

## Changes Implemented

1. Added V100 CUDA APIs for DS4 source `F8_E4M3_B128` batching.
   - `ds4_gpu_arena_f8_e4m3_b128_matmul_batch_f32`
   - `ds4_gpu_arena_f8_e4m3_b128_pair_swiglu_batch_ptrs_f32`
   - `ds4_gpu_arena_f8_e4m3_b128_pair_swiglu_batch_ptr_table_f32`
2. Extended `tests/cuda_source_dtypes_smoke.c`.
   - Validates batched F8 matmul against per-row CPU reference dots.
   - Validates pointer-input pair-SwiGLU.
   - Validates the no-upload device pointer-table variant used by the layer
     executor.
3. Wired shared F8 batching into `execute_ffn_delta_batch` behind
   `DS4_V100_BATCH_SHARED_F8=1`.
   - Default path remains the Sprint 060 per-slot shared expert path because
     the batched F8 path was slower in cluster measurements.
4. Added persistent FFN batch output views.
   - `ffn_routed_out_view[slot]` removes per-layer routed-output view
     allocation in the default batched FFN path.
   - `ffn_shared_batch_view[slot]` supports the opt-in shared F8 batch path.

## Validation

Local:

```bash
cc -fsyntax-only -I. ds4_v100_layer_execute.c
cc -fsyntax-only -I. tests/cuda_source_dtypes_smoke.c
make ds4_v100_layer_execute.o tests/cuda_source_dtypes_smoke.o
git diff --check
```

V100 build:

```bash
CUDA_ARCH=sm_70 make \
  tests/cuda_source_dtypes_smoke \
  tools/ds4-v100-replay \
  tests/cuda_v100_mxfp4_moe_smoke \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke
```

V100 correctness:

```text
cuda_source_dtypes_smoke: ok
cuda_v100_mxfp4_moe_smoke: ok
cuda_v100_integrated_layer_smoke: ok
cuda_v100_stage_scheduler_smoke --slots 2: ok
cuda_v100_full_scheduler_smoke --slots 2: ok
DS4_V100_BATCH_SHARED_F8=1 cuda_v100_full_scheduler_smoke --slots 2: ok
cuda_v100_selected_token_smoke: selected token hex 3136
tools/ds4-v100-replay --tokens 2: first token hex 3136
```

## Sustained Decode Results

| Build | ctx | slots | generated tok/s | continuation tok/s | avg GPU util | max GPU util |
|---|---:|---:|---:|---:|---:|---:|
| Sprint 060 pointer-input routed | 1M | 2 | 3.915266 | 3.670562 | 12.265% | 20.000% |
| Sprint 061 shared F8 batch, first pass | 1M | 2 | 3.868384 | 3.626610 | 12.155% | 20.000% |
| Sprint 061 shared F8 batch, no extra pointer upload | 1M | 2 | 3.884237 | 3.641472 | 11.325% | 20.000% |
| Sprint 061 default restored | 1M | 1 | 3.600787 | 3.375738 | 11.035% | 20.000% |
| Sprint 061 default restored | 1M | 2 | 3.848032 | 3.607530 | 11.427% | 20.000% |
| Sprint 061 persistent views | 1M | 2 | 3.858791 | 3.617617 | 12.222% | 20.000% |
| Sprint 061 persistent views | 256K | 4 | 3.834046 | 3.594418 | 11.717% | 40.000% |

Artifacts:

- `logs/from-cluster/sprint061-shared-f8-batch/`
- `logs/from-cluster/sprint061-shared-f8-batch-no-ptr-upload/`
- `logs/from-cluster/sprint061-default-final/`
- `logs/from-cluster/sprint061-persistent-views/`

## Assessment

The shared F8 batch primitive is correct, but it is not a default throughput
win on V100. The no-extra-pointer-upload variant was faster than the first
batched implementation, but still slower than the Sprint 060 reference run.

The 4-slot 256K run is the more important signal: increasing active slots from
2 to 4 does not increase aggregate tok/s. That means the current layer-batched
runtime still serializes or underfills the real hot work, and further small
FFN staging cleanup is unlikely to reach practical serving throughput.

Next work should move to a larger execution-shape change: either committed MTP
drafting to reduce target-model work per output token, stage wavefronting so
different GPUs work on different token batches concurrently, or a profiler-led
rewrite of the F8/MXFP4 row-reduction kernels into kernels that actually feed
Volta tensor cores or DP4A-style integer throughput.
