# Sprint 109 - F8 Row4 CTA Probe

Date: 2026-05-20

## Objective

Test a larger F8 execution-shape change for the current DS4 V100 appliance:
compute four F8 output rows per CTA for the hottest source-F8 matmul paths.

## Context

The latest useful profile showed the decode hot path is dominated by F8 arena
matmul work:

- `arena_f8_e4m3_b128_matmul_rows2_kernel`: about `39%` GPU time.
- grouped F8 attention-output matmul: about `12%` GPU time before the Sprint
  107 DS4 specialization.
- TurboMind routed experts: about `25%` GPU time.

Sprint 108 proved route metadata fusion is too small to move the 8-slot target.
The next sprint should therefore change a larger device execution shape.

## Plan

Add opt-in row4 variants for the two F8 matmul surfaces that dominate serving:

1. ungrouped source-F8 matmul rows;
2. grouped DS4 attention-output-A rows.

Keep the existing row2 kernels as the production default and rollback path until
V100 A/B evidence says row4 is better.

## Implementation

- Add a 4-accumulator warp/block reduction helper.
- Add `arena_f8_e4m3_b128_matmul_rows4_kernel`.
- Add `arena_f8_e4m3_b128_matmul_grouped_rows4_kernel`.
- Add a DS4-specialized grouped attention-output row4 kernel for
  `groups=8`, `rows_per_group=1024`, `cols_per_group=4096`.
- Add `DS4_V100_CUDA_F8_ROW4` in the launcher and export it as
  `DS4_CUDA_F8_ROW4`.

## Definition of Done

- Local syntax and diff checks pass.
- V100 `sm_70` build passes for replay and focused CUDA smoke tests.
- Correctness passes:
  - source dtype/projection attention smoke or equivalent F8 smoke;
  - full scheduler smoke;
  - selected-token oracle, expected hex `3136`.
- Served A/B evidence exists for:
  - `ctx=262144`, `slots=8`, `active_microbatch=8`;
  - `ctx=1048576`, `slots=4`, `active_microbatch=4`.
- Decision is documented:
  - default row4 only if it improves the primary 8-slot/256K target;
  - otherwise keep row4 opt-in and move to the next larger hot-path target.

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
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke CUDA_ARCH=sm_70 -j80
```

Passed with `DS4_CUDA_F8_ROW4=1`:

- `cuda_source_dtypes_smoke`
- `cuda_v100_projection_attention_smoke`
- `cuda_v100_full_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-s090 --slots 8`
- `cuda_v100_selected_token_smoke --expected-token-hex 3136`

The selected-token smoke selected token id `926`, text hex `3136`.

## Throughput

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Token Match |
|---|---:|---:|---:|---:|---:|
| Row4 opt-in | 262,144 | 8 | 30.998275 | 29.060883 | 8/8 |
| Row2 default control | 262,144 | 8 | 31.380225 | 29.418961 | 8/8 |
| Row4 opt-in | 1,048,576 | 4 | 19.898462 | 18.654808 | 4/4 |
| Row2 default control | 1,048,576 | 4 | 20.041787 | 18.789175 | 4/4 |

## Decision

Do not make row4 the production default. The path is correct, but it regressed
both the main 8-slot/256K serving target and the 4-slot/1M long-context target.
The likely explanation is register pressure / lower occupancy overpowering the
activation-load reuse and lower CTA count.

Keep `DS4_V100_CUDA_F8_ROW4=0` in the launcher and deployment manifests. The
row4 kernel can remain as an opt-in diagnostic for focused profiling, but the
next throughput sprint should move to a different larger boundary: fused
TurboMind gate+up packing/GEMM, persistent grouped expert scheduling, or a
software-pipelined F8 dequant+dot kernel that improves instruction throughput
without reducing occupancy.

## Risks

- Four accumulators may increase register pressure enough to offset reduced
  block count and activation reloads.
- Grouped attention-output row4 assumes DS4 fixed shapes; all nonmatching
  grouped shapes must retain the generic fallback.
- Minor FP reduction-order drift is expected; selected-token correctness is the
  production gate.

## Artifacts

- `logs/from-cluster/sprint109-f8-row4/`
