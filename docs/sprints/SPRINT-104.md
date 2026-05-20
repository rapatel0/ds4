# Sprint 104 - F8 Warp-Reduction Kernels

Date: 2026-05-20

## Objective

Move the current production appliance throughput again by reducing per-CTA
reduction overhead in the hot F8 arena kernels without changing model format,
sharding, or residency.

## Context

Sprint 103 made the E4M3 decode itself cheaper, raising the 8-slot/256K
appliance profile to `30.862791` generated tok/s. The next candidate was to
use spare VRAM for F8-to-F16 cache plus cuBLAS. A direct served-path probe was
rejected during validation because the cache path was far too slow to build/run
and did not meet the practical-use bar.

The sprint then pivoted to the lower-risk F8 kernel synchronization cost:
the F8 matmul kernels reduced 256 thread partials through shared memory and a
full-block barrier at every reduction level.

## Implementation

- Added `arena_warp_sum_f32`, `arena_block_sum_256_f32`, and
  `arena_block_sum2_256_f32` helpers.
- Replaced the F8 arena shared-memory tree reductions with warp-shuffle block
  reductions in:
  - single F8 matmul;
  - row-pair F8 matmul;
  - contiguous batch F8 matmul;
  - pointer-table batch F8 matmul;
  - grouped F8 matmul;
  - shared F8 pair-SwiGLU pointer-table matmul.
- Left BF16/F32/MXFP4 reductions unchanged in this sprint.

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

The selected-token smoke selected token id `926`, text hex `3136`.

## Throughput

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Token Match |
|---|---:|---:|---:|---:|---:|
| Sprint 103 exact-bit F8 decode | 262,144 | 8 | 30.862791 | 28.933867 | 8/8 |
| Sprint 104 F8 warp reduce | 262,144 | 8 | 31.383579 | 29.422106 | 8/8 |
| Sprint 104 F8 warp reduce repeat | 262,144 | 8 | 31.451185 | 29.485486 | 8/8 |
| Sprint 103 exact-bit F8 decode | 1,048,576 | 4 | 19.733742 | 18.500384 | 4/4 |
| Sprint 104 F8 warp reduce | 1,048,576 | 4 | 20.026385 | 18.774736 | 4/4 |

Artifacts:

- `logs/from-cluster/sprint104-f8-warp-reduce/soak-8slot`
- `logs/from-cluster/sprint104-f8-warp-reduce/soak-8slot-repeat`
- `logs/from-cluster/sprint104-f8-warp-reduce/soak-4slot`

## Decision

Ship the warp-reduction F8 kernels. The gain is modest but repeatable, does not
increase VRAM pressure, and preserves the production appliance correctness
checks. The rejected F8-to-F16 cache experiment should not become a default
path unless it is redesigned with bounded eager admission and much faster cache
construction.
