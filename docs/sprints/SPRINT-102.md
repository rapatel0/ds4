# Sprint 102 - F8 Row-Pair Kernel Shape Probe

Date: 2026-05-20

## Objective

Use the V100 node more aggressively and test a broader F8 arena matmul kernel
shape change against the current production appliance baseline.

## Context

Sprint 096 and later profiles kept pointing at F8 arena matmul as the largest
warmed served-path GPU bucket. Sprint 101 repaired projection batching semantics
but did not make projection-only batching a production win. The next useful
probe should change the F8 kernel execution shape itself.

## Plan

1. Add an opt-in row-pair F8 arena matmul path behind
   `DS4_CUDA_F8_ROWPAIR=1`, then expose it through the appliance launcher as
   `DS4_V100_CUDA_F8_ROWPAIR`.
2. Cover the hot arena F8 variants, not only one call site:
   - single `matmul_f32`;
   - contiguous batch `matmul_batch_f32`;
   - pointer-table batch `matmul_batch_ptr_table_f32`;
   - grouped attention-output `matmul_grouped_f32`.
3. Build on the V100 pod with full-node parallelism (`-j80`) on k8s-local
   `/workspace`.
4. Validate correctness, then compare production default versus row-pair on:
   - 8 slots, 256K context;
   - 4 slots, 1M context.

## Definition of Done

- Row-pair kernels are guarded by an env flag and have a clean rollback.
- CUDA smokes and selected-token correctness pass on the V100 pod.
- A/B throughput evidence is recorded before any default decision.
- The path defaults on only if the measured win is material and stable.

## Implementation

- Added row-pair F8 arena matmul kernels that compute two output rows per CTA:
  - `arena_f8_e4m3_b128_matmul_rows2_kernel`
  - `arena_f8_e4m3_b128_matmul_batch_rows2_kernel`
  - `arena_f8_e4m3_b128_matmul_ptrs_rows2_kernel`
  - `arena_f8_e4m3_b128_matmul_grouped_rows2_kernel`
- Wired `DS4_CUDA_F8_ROWPAIR=1` into the single, contiguous-batch,
  pointer-table batch, and grouped F8 arena APIs.
- Added production launcher/config knob `DS4_V100_CUDA_F8_ROWPAIR=1`, exported
  internally as `DS4_CUDA_F8_ROWPAIR`.
- Updated deployment env, Kubernetes config, and the appliance operations
  runbook with the rollback knob.

## V100 Validation

Cluster target: `llamacpp-build-8gpu` on `gpu-01`, using 80 CPU build
parallelism and k8s-local `/workspace`.

Build:

```text
make tools/ds4-v100-replay tests/cuda_v100_projection_attention_smoke \
  tests/cuda_v100_stage_scheduler_smoke tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke CUDA_ARCH=sm_70 -j80
```

Passed with `DS4_CUDA_F8_ROWPAIR=1`:

- `./tests/cuda_v100_projection_attention_smoke`
- `./tests/cuda_v100_stage_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-s090 --stage 0 --slots 4`
- `./tests/cuda_v100_full_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-s090 --slots 8`
- `./tests/cuda_v100_selected_token_smoke ... --expected-token-hex 3136`

Launcher config check:

- `./tools/ds4-v100-run-appliance.sh --env deploy/v100/ds4-v100-appliance.env.example --check`
  reports `cuda_f8_rowpair=1`.

## Throughput

| Mode | Context | Slots | F8 Row-Pair | Generated tok/s | Continuation tok/s | Token Match |
|---|---:|---:|---|---:|---:|---:|
| long default | 1,048,576 | 4 | off | 17.821073 | 16.707256 | 4/4 |
| long row-pair | 1,048,576 | 4 | on | 18.500281 | 17.344013 | 4/4 |
| throughput default | 262,144 | 8 | off | 26.447308 | 24.794352 | 8/8 |
| throughput row-pair | 262,144 | 8 | on | 27.037514 | 25.347670 | 8/8 |
| launcher default | 262,144 | 8 | on | 27.049799 | 25.359186 | 8/8 |

Artifacts:

- `logs/from-cluster/sprint102-f8-rowpair/soak-4slot-default`
- `logs/from-cluster/sprint102-f8-rowpair/soak-4slot-rowpair`
- `logs/from-cluster/sprint102-f8-rowpair/soak-8slot-default`
- `logs/from-cluster/sprint102-f8-rowpair/soak-8slot-rowpair`
- `logs/from-cluster/sprint102-f8-rowpair/soak-8slot-launcher-default`

## Decision

Ship F8 row-pair as the production appliance default. It improves the
8-slot/256K profile by about `2.2%`, improves the 4-slot/1M profile by about
`3.8%`, preserves token-match correctness, and validates through the launcher
default path.
