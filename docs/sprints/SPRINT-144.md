# Sprint 144 - SM70 MXFP4 N256 Tile Probe

Date: 2026-05-21

## Objective

Test whether a wider-N SM70 MXFP4 tile improves the 768-route high-slot
routed-FFN path. Sprint 142 showed route/gather/scatter plumbing is too small
to move end-to-end throughput; this sprint changes the actual TurboMind GEMM
tile geometry for gate/up and down.

## Implementation

- Added fixed-shape `m64n256` TurboMind DS4 probes for:
  - `ggml_turbomind_ds4_mxfp4_gated_silu_768_m64n256`
  - `ggml_turbomind_ds4_mxfp4_down_768_m64n256`
- The tile uses the existing SM70 MXFP4 `64x256x32` family from the copied
  TurboMind registry.
- Wired explicit runtime selection through:
  - `DS4_V100_TURBOMIND_GATE_UP_PROBE=m64n256`
  - `DS4_V100_TURBOMIND_DOWN_PROBE=m64n256`
- Kept defaults unchanged.

## V100 Validation

Builds:

```text
cmake --build build/turbomind-v100-s127 \
  --target ggml-turbomind test_ggml_turbomind_grouped_gate_up_fusion -j80

make -j80 CUDA_ARCH=sm_70 \
  ds4 ds4-server tools/ds4-v100-replay tests/cuda_v100_full_scheduler_smoke
```

Full 43-layer smoke passed for both candidate shapes:

```text
DS4_V100_TURBOMIND_GATE_UP_PROBE=auto
DS4_V100_TURBOMIND_DOWN_PROBE=m64n256
cuda_v100_full_scheduler_smoke ... ok

DS4_V100_TURBOMIND_GATE_UP_PROBE=m64n256
DS4_V100_TURBOMIND_DOWN_PROBE=off
cuda_v100_full_scheduler_smoke ... ok
```

Standalone compact 768-route benchmark:

| Variant | Gate/up probe ms | Down probe ms | Correctness |
|---|---:|---:|---|
| gate `m128` | `0.5801` | n/a | pass |
| gate `m64n256` | `0.5947` | n/a | pass |
| gate `m64` | `0.6526` | n/a | pass |
| down `m128` | n/a | `0.2936` | pass |
| down `m64n256` | n/a | `0.2896` | pass |

Served 128-slot/32K A/B with split prefill/decode metrics:

| Run | Generated tok/s | Prompt tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---|
| control | `59.993301` | `67.492463` | `56.243719` | 128/128 |
| down `m64n256` | `59.791839` | `67.265819` | `56.054849` | 128/128 |
| gate `m64n256` | `59.797232` | `67.271886` | `56.059905` | 128/128 |

Artifacts:

- `logs/from-cluster/sprint144-m64n256-standalone/`
- `logs/from-cluster/sprint144-served-control/`
- `logs/from-cluster/sprint144-served-down-m64n256/`
- `logs/from-cluster/sprint144-served-gate-m64n256/`

## Decision

Keep `m64n256` as an explicit probe, but do not promote it. The isolated down
projection improved slightly, but the served path regressed by about `0.3%`.
The isolated gate/up result was also slower than the current `m128` probe.

This reinforces the current conclusion: individual GEMM tile tweaks can improve
microbenchmarks but do not close the appliance throughput gap. The next useful
implementation should target a larger routed-FFN executor boundary or a
persistent/scheduler change that keeps expert work resident and better fed.
