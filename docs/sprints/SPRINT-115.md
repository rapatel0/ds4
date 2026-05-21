# Sprint 115 - Shared Gate/Up SwiGLU F8 HMMA Probe

Date: 2026-05-20

## Objective

Move a larger production shared-FFN F8 region onto V100 tensor cores by adding
an opt-in DS4-shaped HMMA kernel for the batched shared gate/up projections and
SwiGLU activation.

## Context

Sprint 114 proved that the shared-down F8 batch projection can be expressed as
a small-M Volta WMMA kernel, but the isolated win was too small to promote. The
next F8 target should own more work per launch. Shared gate/up is a better
candidate because it computes two `2048 x 4096` projections from the same
activation rows and immediately applies SwiGLU:

```text
gate = A[n_tokens, 4096] * W_gate[2048, 4096]^T
up   = A[n_tokens, 4096] * W_up[2048, 4096]^T
mid  = swiglu(gate, up)
```

This path runs in the production batched shared-FFN flow before shared-down.

## Plan

1. Add an off-by-default CUDA env flag:
   `DS4_CUDA_F8_HMMA_PAIR_SWIGLU`.
2. Expose it through the appliance launcher as
   `DS4_V100_CUDA_F8_HMMA_PAIR_SWIGLU`.
3. Add one DS4-shaped Volta WMMA/HMMA kernel for the fixed production shape:
   - `rows == 2048`;
   - `cols == 4096`;
   - `n_tokens == 4 || n_tokens == 8`;
   - input rows are provided by the existing device pointer table.
4. Dispatch from
   `ds4_gpu_arena_f8_e4m3_b128_pair_swiglu_batch_ptr_table_f32()` only when
   the flag and shape match.
5. Keep the scalar pair-SwiGLU kernel as the default and fallback.
6. Add a focused CUDA smoke comparing the target-shape HMMA path against the
   current scalar pair-SwiGLU path.
7. Run V100 correctness and same-binary throughput A/B:
   - 8-slot/256K primary serving tier;
   - 4-slot/1M long-context tier;
   - selected-token correctness with expected bytes `3136`.

## Definition of Done

- [x] `sm_70` build passes for replay and new/affected CUDA smokes.
- [x] Focused target-shape pair-SwiGLU HMMA smoke passes.
- [x] Full scheduler smoke passes with the fused appliance and HMMA pair flag
      on.
- [x] Selected-token smoke passes with expected hex `3136`.
- [x] Same-binary 8-slot/256K A/B is recorded for
      `DS4_V100_CUDA_F8_HMMA_PAIR_SWIGLU=0/1`.
- [x] Same-binary 4-slot/1M A/B is recorded for
      `DS4_V100_CUDA_F8_HMMA_PAIR_SWIGLU=0/1`.
- [x] Default decision is documented.

## Implementation

- `ds4_cuda.cu` adds
  `arena_f8_e4m3_b128_pair_swiglu_ptr_table_hmma_kernel`, a Volta WMMA kernel
  for the fixed shared gate/up batch shape.
- The kernel reuses the existing device pointer table for activation rows,
  loads each 16-token activation tile once, runs separate gate and up HMMA
  accumulators, then applies the existing DS4 SwiGLU clamp/activation before
  writing `mid[tokens, 2048]`.
- The dispatch is guarded by `DS4_CUDA_F8_HMMA_PAIR_SWIGLU=1` and only fires
  for `rows=2048`, `cols=4096`, and `n_tokens=4` or `8`.
- The launcher default is now `DS4_V100_CUDA_F8_HMMA_PAIR_SWIGLU=1`, with
  rollback by setting it to `0`.
- `DS4_V100_CUDA_F8_HMMA_SHARED_DOWN` remains default-off because the combined
  pair+down result improved 8-slot throughput but regressed 4-slot/1M.
- `tests/cuda_f8_hmma_pair_swiglu_smoke.c` compares the target-shape HMMA path
  against the existing scalar pair-SwiGLU path.

## Validation

Cluster build:

```text
CUDA_ARCH=sm_70 make -j80 tests/cuda_f8_hmma_pair_swiglu_smoke \
  tests/cuda_f8_hmma_shared_down_smoke \
  tests/cuda_source_dtypes_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke \
  tools/ds4-v100-replay
```

Correctness:

- `tests/cuda_f8_hmma_pair_swiglu_smoke`: passed.
- `tests/cuda_f8_hmma_shared_down_smoke`: passed.
- `tests/cuda_source_dtypes_smoke`: passed.
- `tests/cuda_v100_full_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 --slots 8`
  with `DS4_CUDA_F8_HMMA_PAIR_SWIGLU=1`: passed, `tm_layers=43`.
- `tests/cuda_v100_selected_token_smoke --expected-token-hex 3136` with
  `DS4_CUDA_F8_HMMA_PAIR_SWIGLU=1`: passed, selected token id `926`, logit
  `35.254646`.

## Throughput

Same-binary fused Sprint111 appliance A/B, `tokens=16`,
`DS4_V100_FFN_DIRECT_DELTA=0`,
`DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS=1`.

Isolated pair-SwiGLU HMMA:

| Mode | Context | Slots | Pair HMMA | Down HMMA | Generated tok/s | Continuation tok/s | Token match |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control | 262,144 | 8 | `0` | `0` | `33.292541` | `31.211757` | 8/8 |
| Candidate | 262,144 | 8 | `1` | `0` | `33.578236` | `31.479596` | 8/8 |
| Control | 1,048,576 | 4 | `0` | `0` | `21.430420` | `20.091019` | 4/4 |
| Candidate | 1,048,576 | 4 | `1` | `0` | `21.455638` | `20.114660` | 4/4 |

Combined Sprint114+Sprint115 flags:

| Mode | Context | Slots | Pair HMMA | Down HMMA | Generated tok/s | Continuation tok/s | Token match |
|---|---:|---:|---:|---:|---:|---:|---:|
| Combined | 262,144 | 8 | `1` | `1` | `33.674684` | `31.570016` | 8/8 |
| Combined | 1,048,576 | 4 | `1` | `1` | `21.370925` | `20.035242` | 4/4 |

Artifacts:

```text
logs/from-cluster/sprint115-hmma-pair-swiglu/
```

Decision:

- Promote `DS4_V100_CUDA_F8_HMMA_PAIR_SWIGLU=1` to the launcher default.
- Keep `DS4_V100_CUDA_F8_HMMA_SHARED_DOWN=0` as default because the combined
  path regresses the 4-slot/1M tier even though it sets a new 8-slot best.
- Treat the combined `33.674684` 8-slot result as an opt-in data point for the
  next F8 fusion sprint, not as the production default.

## Risks

- As in Sprint 114, WMMA uses FP16 inputs with FP32 accumulation while the
  scalar reference uses F32 decoded F8 values and F32 activations.
- This still leaves the shared-down launch separate. If gate/up wins, the next
  step should evaluate the pair flag combined with the Sprint 114 shared-down
  flag, then decide whether a larger fused shared-FFN kernel is worth building.
