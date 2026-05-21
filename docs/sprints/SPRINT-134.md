# Sprint 134 - Fixed-Shape Compact Gate/Up ABI Probe

Date: 2026-05-21

## Objective

Test whether bypassing TurboMind's generic dispatch path helps the compact
production routed gate/up shape. Sprint 133 established the correct acceptance
baseline: compact 96-route interleaved gated-SiLU at about `0.1740 ms`.

## Implementation

Added an experimental DS4/V100-only C ABI:

```text
ggml_turbomind_ds4_mxfp4_gated_silu_96
```

The probe is intentionally fixed:

- MXFP4 only;
- interleaved gate/up rows;
- `K=4096`, fused `N=4096`, output `N/2=2048`;
- group size `32`;
- compact `num_experts=6`;
- `total_tokens=96`;
- direct SM70 kernel instantiation using TurboMind's existing
  `Config_MXF4<kColMajor, 0>::Type<16,128,32,...>` path.

The benchmark now dlsyms this optional symbol and calls it only for the compact
96-route case. If the symbol is absent, older TurboMind builds still run the
existing benchmark.

Files:

- `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h`
- `kernels/turbomind/ggml-turbomind/api.cc`
- `kernels/turbomind/ggml-turbomind/ggml-turbomind-ds4-probe.cu`
- `kernels/turbomind/ggml-turbomind/CMakeLists.txt`
- `kernels/turbomind/ggml-turbomind/test_grouped_gate_up_fusion.cpp`

## V100 Validation

Build:

```text
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc \
  'cd /workspace/ds4 && cmake --build build/turbomind-v100-s127 \
   --target ggml-turbomind test_ggml_turbomind_grouped_gate_up_fusion -j80'
```

Focused compact run:

```text
DS4_TURBOMIND_GATE_UP_COMPACT_GROUPS=1 \
DS4_TURBOMIND_GATE_UP_CASES=16 \
DS4_TURBOMIND_GATE_UP_WARMUP_ITERS=5 \
DS4_TURBOMIND_GATE_UP_BENCH_ITERS=100 \
  build/turbomind-v100-s127/test_ggml_turbomind_grouped_gate_up_fusion \
  build/turbomind-v100-s127/libggml-turbomind.so
```

| Mode | Time |
|---|---:|
| compact separate gate+up | `0.1888 ms` |
| compact fused gate_up | `0.1734 ms` |
| compact gated-SiLU generic | `0.1746 ms` |
| fixed-shape DS4 probe | `0.1746 ms` |

Correctness:

```text
probe_max_abs=0.0000e+00
probe_rel=0.0000e+00
probe_bad=0/196608
PASS
```

## Decision

Do not promote this probe into the appliance runtime.

The fixed-shape launch is correct, but it is exactly neutral against the
generic TurboMind gated path. That means the generic dispatch path is already
selecting the same effective SM70 kernel for the compact production shape. The
next useful work must change the kernel computation itself or the served
scheduling shape; merely bypassing dispatch is not enough.
