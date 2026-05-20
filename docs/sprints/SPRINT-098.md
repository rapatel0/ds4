# Sprint 098 - Grouped F8 Attention Output

Date: 2026-05-20

## Objective

Reduce served F8 matmul launch count in the attention output path without
changing DS4 math, model layout, or the appliance shard format.

## Changes

- Added `ds4_gpu_arena_f8_e4m3_b128_matmul_grouped_f32()`.
  - It consumes one F8_E4M3_B128 source view whose rows are grouped.
  - Each output row keeps the same per-row dot-product reduction as the old
    single-group kernel.
  - The only scheduling change is that grouped slices are launched together.
- Switched `attn_output_a` from eight per-group F8 matmul launches to one
  grouped launch over all output-A rows.
- Added operator rollback:
  - `DS4_V100_DISABLE_GROUPED_ATTN_OUTPUT_A=0` by default.
  - Set it to `1` to restore the older one-launch-per-output-group path.
- Updated deployment env, k8s config, and runbook docs for the rollback flag.

## Validation

Cluster build:

```text
make -C /workspace/ds4-sprint082 tools/ds4-v100-replay CUDA_ARCH=sm_70 -j8
```

Same-binary soak comparisons:

| Scenario | Path | Generated tok/s | Continuation tok/s | Correctness |
| --- | --- | ---: | ---: | --- |
| 4 slots, 1M ctx | grouped | `17.904697` | `16.785654` | `token_match=4/4` |
| 4 slots, 1M ctx | rollback | `16.897788` | `15.841676` | `token_match=4/4` |
| 8 slots, 256K ctx | grouped | `26.206100` | `24.568219` | `token_match=8/8` |
| 8 slots, 256K ctx | rollback | `25.456942` | `23.865883` | `token_match=8/8` |

## Profile Result

Warmed served-path profile for 4-slot/1M grouped output-A:

| Bucket | Sprint 097 pool default | Sprint 098 grouped output-A |
| --- | ---: | ---: |
| Single F8 matmul kernel calls | `11880` | `5544` |
| Grouped output-A F8 calls | `0` | `792` |
| F8 GPU time | `750.04 ms` | `528.77 + 167.25 ms` |
| Total CUDA kernel launches | `39684` | `34140` |
| `cudaLaunchKernel` API time | `210.81 ms` | `200.52 ms` |

The grouped kernel is intentionally slower per launch because each launch
covers all output groups. The end-to-end win comes from removing thousands of
small launches and tensor-view calls.

Artifacts:

- `logs/from-cluster/sprint098-f8-grouped-output/soak-4slot-grouped/summary.json`
- `logs/from-cluster/sprint098-f8-grouped-output/soak-4slot-rollback/summary.json`
- `logs/from-cluster/sprint098-f8-grouped-output/soak-8slot-grouped/summary.json`
- `logs/from-cluster/sprint098-f8-grouped-output/soak-8slot-rollback/summary.json`
- `logs/from-cluster/sprint098-f8-grouped-output/profile-4slot-grouped/nvprof.log`

## Decision

Ship grouped attention output-A as the default. It preserves correctness,
improves the 4-slot fixture by about `6%`, improves the 8-slot fixture by about
`3%`, and directly reduces the F8 launch count shown in Sprint 097.

Next optimization should continue in the F8 path:

1. Batch or fuse attention Q/KV projection matmuls across active slots.
2. Reduce `cudaMemcpy` API overhead from remaining control/result copies.
3. Revisit F8-to-F16 cached cuBLAS only where it can replace single-token F8
   projection kernels, not just shared-down batch matmul.
