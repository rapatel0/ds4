# Sprint 103 - Exact-Bit F8 Decode

Date: 2026-05-20

## Objective

Move the current production appliance throughput by attacking the dominant F8
arena decode/matmul cost without changing the DS4 source dtype contract.

## Context

Sprint 096 and Sprint 100 profiles consistently showed F8 arena matmul as the
largest warmed served-path GPU bucket, with TurboMind routed MXFP4 second. NCU
evidence from the F8 path showed the kernel was instruction/SM bound, not
DRAM-bound. Sprint 102 improved the F8 launch shape with row-pair CTAs, but the
per-weight E4M3 decode still used `ldexpf()`.

## Plan

1. Keep the existing TurboMind appliance expert path as-is.
2. Replace the E4M3FN F8 decode helper with exact IEEE-F32 bit construction.
3. Rebuild on the V100 pod with full-node parallelism.
4. Validate source dtype, projection attention, stage scheduler, full
   scheduler, selected-token correctness, and production soaks.

## Implementation

- Replaced `arena_e4m3fn_to_f32()` in `ds4_cuda.cu` with exact integer
  float-bit construction.
- Normal E4M3 values now construct the F32 exponent/mantissa directly.
- E4M3 subnormals now normalize the 3-bit mantissa into F32 exponent/mantissa
  bits directly.
- Zero and NaN behavior is preserved.
- Extended `tests/cuda_v100_selected_token_smoke` with `--tm-index`,
  `--shard-dir`, and `--appliance-dir` so the selected-token smoke can validate
  the actual single-directory TurboMind appliance.

## V100 Validation

Cluster target: `llamacpp-build-8gpu` on `gpu-01`, using k8s-local
`/workspace`, 80 CPU build parallelism, and the full 8x V100 stack.

Build:

```text
make tools/ds4-v100-replay tests/cuda_source_dtypes_smoke \
  tests/cuda_v100_projection_attention_smoke \
  tests/cuda_v100_stage_scheduler_smoke tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke CUDA_ARCH=sm_70 -j80
```

Passed:

- `./tests/cuda_source_dtypes_smoke`
- `./tests/cuda_v100_projection_attention_smoke`
- `./tests/cuda_v100_stage_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-s090 --stage 0 --slots 4`
- `./tests/cuda_v100_full_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-s090 --slots 8`
- `./tests/cuda_v100_selected_token_smoke --model /models/DSv4-Flash-256e-fixed.gguf --appliance-dir /workspace/ds4-appliance-full-tm-s090 --expected-token-hex 3136`
- `./tools/ds4-v100-replay --model /models/DSv4-Flash-256e-fixed.gguf --appliance-dir /workspace/ds4-appliance-full-tm-s090 --ctx 262144 --slots 8 --active-microbatch 8 --tokens 2 --expected-token-hex 3136 --json`

The selected-token replay selected token id `926`, text hex `3136`.

## Throughput

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Token Match |
|---|---:|---:|---:|---:|---:|
| Sprint 102 launcher default | 262,144 | 8 | 27.049799 | 25.359186 | 8/8 |
| Sprint 103 exact-bit F8 decode | 262,144 | 8 | 30.862791 | 28.933867 | 8/8 |
| Sprint 102 row-pair | 1,048,576 | 4 | 18.500281 | 17.344013 | 4/4 |
| Sprint 103 exact-bit F8 decode | 1,048,576 | 4 | 19.733742 | 18.500384 | 4/4 |

Artifacts:

- `logs/from-cluster/sprint103-f8-bitdecode/soak-8slot`
- `logs/from-cluster/sprint103-f8-bitdecode/soak-4slot`

## Decision

Ship exact-bit E4M3 decode as the default F8 decode implementation. It preserves
selected-token correctness and improves the production 8-slot/256K appliance
profile by about `14.1%` over Sprint 102, with a `6.7%` improvement at
4-slot/1M.
