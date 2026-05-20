# Sprint 107 - DS4 Grouped F8 Attention-Output Kernel

Date: 2026-05-20

## Objective

Use the Sprint 106 served profile to make a small, production-safe improvement
in the hottest remaining F8 grouped rows2 path.

## Context

Sprint 106 showed that after exact-bit F8 decode and warp reductions, the warmed
served profile was still dominated by:

- F8 rows2 arena matmul: `38.97%` GPU time.
- F8 grouped rows2 arena matmul: `12.39%` GPU time.
- TurboMind SM70 MXFP4 grouped GEMM: `25.42%` GPU time.

The grouped F8 rows2 bucket is mainly the DS4 attention output-A projection.
That shape is fixed in the base model:

- groups: `8`
- rows per group: `1024`
- columns per group: `4096`

## Implementation

- Added `arena_f8_e4m3_b128_matmul_grouped_rows2_ds4_attn_o_kernel`.
- The specialized kernel removes the generic grouped rows2 per-CTA group
  divisions and cross-group checks for the fixed DS4 attention-output-A shape.
- Kept the existing generic grouped rows2 kernel as fallback for all nonmatching
  shapes.
- Added rollback/config knob:
  - launcher: `DS4_V100_CUDA_F8_GROUPED_DS4_FAST`
  - CUDA runtime: `DS4_CUDA_F8_GROUPED_DS4_FAST`
- Defaulted the specialized path on in the launcher and deployment example.

## V100 Validation

Cluster target:

- pod: `llamacpp-build-8gpu`
- node: `gpu-01`
- storage: k8s-local `/workspace`
- host resources: 8x V100-SXM2-32GB, 80 CPU cores, 256 GB RAM

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
| Sprint 104 baseline repeat | 262,144 | 8 | 31.451185 | 29.485486 | 8/8 |
| Sprint 107 generic rollback | 262,144 | 8 | 31.098630 | 29.154965 | 8/8 |
| Sprint 107 DS4 grouped fast | 262,144 | 8 | 31.811137 | 29.822941 | 8/8 |
| Sprint 107 DS4 grouped fast repeat | 262,144 | 8 | 31.630774 | 29.653851 | 8/8 |
| Sprint 104 baseline | 1,048,576 | 4 | 20.026385 | 18.774736 | 4/4 |
| Sprint 107 DS4 grouped fast | 1,048,576 | 4 | 20.095510 | 18.839541 | 4/4 |
| Sprint 107 generic rollback | 1,048,576 | 4 | 20.105807 | 18.849194 | 4/4 |

## Decision

Ship the specialized grouped F8 path as the default. It produces a measurable
8-slot/256K serving improvement over both the Sprint 104 baseline and same-binary
rollback, while the 4-slot/1M case is neutral within run noise. Keep
`DS4_V100_CUDA_F8_GROUPED_DS4_FAST=0` as the rollback for focused A/B runs.

The next larger performance target should be TurboMind route-build fusion:
combine count/prefix/scatter for small route counts, remove output memset plus
atomic scatter where possible, and then evaluate indexed gate/up or fused
gate/up appliance packing.

## Artifacts

- `logs/from-cluster/sprint107-f8-grouped-ds4-fast/`
