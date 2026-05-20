# Sprint 099 - Batch Attention Projection Probe

Date: 2026-05-20

## Objective

Test whether batching attention Q-A, Q-B, and KV F8 projections across active
slots improves practical multi-slot serving throughput after Sprint 098 reduced
attention output launch count.

## Changes

- Added a reusable device row-pointer upload helper:
  - `ds4_gpu_tensor_write_f32_row_ptrs()`
- Added pointer-table F8 batch matmul:
  - `ds4_gpu_arena_f8_e4m3_b128_matmul_batch_ptr_table_f32()`
- Added an opt-in attention projection batch path behind:
  - `DS4_V100_ENABLE_BATCH_ATTN_PROJ=1`
- The production default remains the Sprint 098 path. The probe is disabled
  unless explicitly enabled.

The probe batches the Q-A and KV F8 matmuls from separate slot tensors, batches
Q-A/KV RMS normalization, and batches Q-B over the contiguous Q-A-normalized
rows. Per-slot RoPE, compressed KV update, indexed attention, attention decode,
and attention output stay on the established correctness path so different slot
positions remain valid.

## Validation

Cluster build:

```text
make -C /workspace/ds4-sprint082 tools/ds4-v100-replay CUDA_ARCH=sm_70 -j8
```

Same-binary soak comparisons:

| Scenario | Path | Generated tok/s | Continuation tok/s | Correctness |
| --- | --- | ---: | ---: | --- |
| 4 slots, 1M ctx | batch projection | `17.742637` | `16.633723` | `token_match=4/4` |
| 4 slots, 1M ctx | rollback/default | `17.764257` | `16.653991` | `token_match=4/4` |
| 8 slots, 256K ctx | batch projection | `26.128571` | `24.495535` | `token_match=8/8` |
| 8 slots, 256K ctx | rollback/default | `26.149613` | `24.515262` | `token_match=8/8` |
| 4 slots, 1M ctx | production default after opt-in flip | `17.950763` | `16.828841` | `token_match=4/4` |

## Decision

Do not ship batch attention projection as the default. It is correct, but it is
flat to slightly slower in both measured serving fixtures. The likely cause is
that the saved F8 launches are offset by pointer-table upload, larger temporary
working sets, and the fact that per-slot RoPE/cache/attention work still
dominates this section.

The code remains as an explicit opt-in probe so the result is reproducible and
we do not reimplement the same shape later.

Next practical targets:

1. Profile and reduce remaining `cudaMemcpy` API overhead with an API trace or
   targeted counters.
2. Batch work at a larger boundary than projections only, such as full
   per-slot attention decode/output where slot positions and cache pointers can
   be represented without pointer-table upload overhead.
3. Revisit low-bit F8 kernels only if they replace the row-dot F8 path with
   Volta tensor-core work rather than merely reducing launches.

Artifacts:

- `logs/from-cluster/sprint099-batch-attn-proj/soak-4slot-batch/summary.json`
- `logs/from-cluster/sprint099-batch-attn-proj/soak-4slot-rollback/summary.json`
- `logs/from-cluster/sprint099-batch-attn-proj/soak-8slot-batch/summary.json`
- `logs/from-cluster/sprint099-batch-attn-proj/soak-8slot-rollback/summary.json`
- `logs/from-cluster/sprint099-batch-attn-proj/soak-4slot-default-off/summary.json`
