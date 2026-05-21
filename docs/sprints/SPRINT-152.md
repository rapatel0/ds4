# Sprint 152 - Fused Gate/Up Software-Pipeline Sweep

Date: 2026-05-21

## Objective

Fully test whether deeper SM70 software pipelining inside the fused MXFP4
gate/up+gated-SiLU kernel is a material throughput lever before spending more
time on that single-GEMM boundary.

## Changes

- Added 3-stage TurboMind MXFP4 probe variants beside the existing 2-stage and
  4-stage variants:
  - `ggml_turbomind_ds4_mxfp4_gated_silu_768_m64_s3`
  - `ggml_turbomind_ds4_mxfp4_gated_silu_768_m128_s3`
  - `ggml_turbomind_ds4_mxfp4_gated_silu_1536_m64_s3`
  - `ggml_turbomind_ds4_mxfp4_gated_silu_1536_m128_s3`
- Extended `test_ggml_turbomind_grouped_gate_up_fusion` so the same benchmark
  can select `m64_s3`, `m128_s3`, and the 1536-route aliases.
- Wired the new symbols as explicit appliance diagnostics through the
  TurboMind C ABI, runtime `dlsym` table, and launcher whitelist. `auto` remains
  unchanged.
- Built the V100 TurboMind target on the 8-GPU cluster and ran compact
  6-expert sweeps with 100 timed iterations.

## Results

768 routed rows, the 128-slot/32K compact served shape:

| Probe | Probe time | Relative to `m128` | Correctness |
|---|---:|---:|---|
| `m64` | `0.6523 ms` | `0.890x` | PASS |
| `m64_s3` | `0.6531 ms` | `0.889x` | PASS |
| `m64_s4` | `0.6522 ms` | `0.891x` | PASS |
| `m128` | `0.5809 ms` | `1.000x` | PASS |
| `m128_s3` | `0.5863 ms` | `0.991x` | PASS |
| `m128_s4` | `0.5794 ms` | `1.003x` | PASS |

1536 routed rows, the 256-slot/16K compact shape:

| Probe | Probe time | Relative to `m128_1536` | Correctness |
|---|---:|---:|---|
| `m64_s3_1536` | `1.0982 ms` | `0.796x` | PASS |
| `m64_s4_1536` | `1.1178 ms` | `0.782x` | PASS |
| `m128_1536` | `0.8743 ms` | `1.000x` | PASS |
| `m128_s3_1536` | `0.8821 ms` | `0.991x` | PASS |
| `m128_s4_1536` | `0.8774 ms` | `0.996x` | PASS |

NCU all-GEMM profiling of the 768-route fixed probe launch shows the same
hardware-counter story:

| Probe | Fixed probe time | SM throughput | DRAM throughput | HMMA instructions |
|---|---:|---:|---:|---:|
| `m128` | `690.18 us` | `40.65%` | `11.64%` | `50,331,648` |
| `m128_s3` | `695.30 us` | `40.19%` | `11.51%` | `50,331,648` |
| `m128_s4` | `688.77 us` | `40.57%` | `11.61%` | `50,331,648` |

## Decision

Do not promote deeper stage-count variants. Stage count alone is not the
material software-pipelining lever for the current fused gate/up GEMM.

The next fused-kernel attempt must change a larger routed-FFN boundary, such as
gate/up plus activation plus down/reduce, or it should pause in favor of the
bounded 2-way TP prototype for the 128-slot/32K tier.

## Artifacts

- `logs/from-cluster/sprint152-sw-pipeline/`
- `logs/from-cluster/sprint152-sw-pipeline-ncu/`

## Validation

- `cmake --build build/turbomind-v100-s127 --target ggml-turbomind test_ggml_turbomind_grouped_gate_up_fusion -j80`
- `make -j80 CUDA_ARCH=sm_70 ds4_cuda.o tools/ds4-v100-replay`
- `DS4_V100_TURBOMIND_GATE_UP_PROBE=m128_s3 ./tools/ds4-v100-run-appliance.sh --check --allow-missing`
- `nm -D build/turbomind-v100-s127/libggml-turbomind.so | grep "gated_silu_.*_s3"`
