# Sprint 114 - Shared-Down F8 HMMA Probe

Date: 2026-05-20

## Objective

Move one production F8 projection from scalar row-pair dot-product CTAs toward
V100 tensor-core execution by adding an opt-in DS4-shaped HMMA kernel for the
batched shared-FFN down projection.

## Context

The fused TurboMind gate/up appliance is now the default best path, but the
current topline remains only `33.589285` generated tok/s at 8-slot/256K.
Sprint 112 profiling showed F8 row-pair and grouped row-pair kernels at about
`54.58%` of GPU time after fused TurboMind expert dispatch. Sprints 112 and 113
also showed that small scalar-kernel or host-boundary tweaks can stay correct
while reducing throughput.

The shared-down FFN batch call is a better first tensor-core target than the
grouped attention-output-A kernel because it has a real small-M GEMM shape:

```text
C[n_tokens, 4096] = A[n_tokens, 2048] * W_f8[4096, 2048]^T
```

with `n_tokens=4` or `8` in the current practical serving tiers.

## Plan

1. Add an off-by-default CUDA env flag:
   `DS4_CUDA_F8_HMMA_SHARED_DOWN`.
2. Expose it through the appliance launcher as
   `DS4_V100_CUDA_F8_HMMA_SHARED_DOWN`.
3. Add one DS4-shaped Volta WMMA/HMMA kernel inside
   `ds4_gpu_arena_f8_e4m3_b128_matmul_batch_f32()`.
4. Dispatch only when all constraints match:
   - F8_E4M3_B128 layout is valid;
   - `rows == 4096`;
   - `cols == 2048`;
   - `n_tokens == 4 || n_tokens == 8`;
   - the opt-in flag is enabled.
5. Keep the current scalar row-pair batch kernel as the production default and
   fallback.
6. Add a focused CUDA smoke that compares the opt-in HMMA path against the
   existing scalar path on the target shape.
7. Run V100 correctness and same-binary throughput A/B:
   - 8-slot/256K primary serving tier;
   - 4-slot/1M long-context tier;
   - selected-token correctness with expected bytes `3136`.

## Definition of Done

- [x] `sm_70` build passes for the replay tool and new/affected CUDA smokes.
- [x] Focused target-shape F8 HMMA smoke passes.
- [x] Full scheduler smoke passes with the fused appliance and HMMA flag on.
- [x] Selected-token smoke passes with expected hex `3136`.
- [x] Same-binary 8-slot/256K A/B is recorded for
      `DS4_V100_CUDA_F8_HMMA_SHARED_DOWN=0/1`.
- [x] Same-binary 4-slot/1M A/B is recorded for
      `DS4_V100_CUDA_F8_HMMA_SHARED_DOWN=0/1`.
- [x] Default decision is documented:
  - enable only if throughput improves outside run noise and correctness holds;
  - otherwise keep it opt-in/off.

## Implementation

- `ds4_cuda.cu` adds
  `arena_f8_e4m3_b128_matmul_batch_hmma_shared_down_kernel`, a Volta WMMA
  kernel for the fixed shared-down batch shape:
  `tokens x 2048` activations multiplied by `4096 x 2048` F8_E4M3_B128
  weights.
- The dispatch is guarded by `DS4_CUDA_F8_HMMA_SHARED_DOWN=1` and only fires
  for `rows=4096`, `cols=2048`, and `n_tokens=4` or `8`.
- All non-matching calls continue using the existing scalar row-pair batch
  kernel. The existing F8-to-F16 cuBLAS cache path is bypassed only when this
  opt-in HMMA path is selected.
- `tools/ds4-v100-run-appliance.sh` exposes the runtime flag as
  `DS4_V100_CUDA_F8_HMMA_SHARED_DOWN`.
- `deploy/v100/ds4-v100-appliance.env.example` documents the flag as opt-in.
- `tests/cuda_f8_hmma_shared_down_smoke.c` compares the target-shape HMMA path
  against the existing scalar batch path.

## Validation

Cluster build:

```text
CUDA_ARCH=sm_70 make -j80 tests/cuda_f8_hmma_shared_down_smoke \
  tests/cuda_source_dtypes_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke \
  tools/ds4-v100-replay
```

Correctness:

- `tests/cuda_f8_hmma_shared_down_smoke`: passed.
- `tests/cuda_source_dtypes_smoke`: passed.
- `tests/cuda_v100_full_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 --slots 8`
  with `DS4_CUDA_F8_HMMA_SHARED_DOWN=1`: passed, `tm_layers=43`.
- `tests/cuda_v100_selected_token_smoke --expected-token-hex 3136` with
  `DS4_CUDA_F8_HMMA_SHARED_DOWN=1`: passed, selected token id `926`, logit
  `35.249512`.

## Throughput

Same-binary fused Sprint111 appliance A/B, `tokens=16`,
`DS4_V100_FFN_DIRECT_DELTA=0`,
`DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS=1`:

| Mode | Context | Slots | HMMA flag | Generated tok/s | Continuation tok/s | Token match |
|---|---:|---:|---:|---:|---:|---:|
| Control | 262,144 | 8 | `0` | `33.397763` | `31.310403` | 8/8 |
| Candidate | 262,144 | 8 | `1` | `33.550415` | `31.453514` | 8/8 |
| Control | 1,048,576 | 4 | `0` | `21.365610` | `20.030259` | 4/4 |
| Candidate | 1,048,576 | 4 | `1` | `21.396331` | `20.059061` | 4/4 |

Artifacts:

```text
logs/from-cluster/sprint114-hmma-shared-down/
```

The candidate is correct and slightly faster in this run, but the win is inside
normal run noise and below the previous best 8-slot fused repeat
(`33.589285`). Keep `DS4_V100_CUDA_F8_HMMA_SHARED_DOWN=0` as the production
default and leave the HMMA path opt-in for further kernel work.

## Risks

- WMMA computes with FP16 inputs and FP32 accumulation, while the current scalar
  path multiplies F32-decoded F8 weights by F32 activations. Selected-token and
  serving A/B decide whether this precision change is acceptable.
- Small `M=4/8` pads to the Volta `m16` WMMA tile, so the kernel may reduce
  launches and scalar work but still waste tensor-core lanes.
- This targets only shared-down first. If it wins, shared gate/up or a
  fused pair-SwiGLU HMMA path should be planned separately.
