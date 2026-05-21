# Sprint 132 - Production-Shaped TurboMind Gate/Up Benchmark

Date: 2026-05-21

## Objective

Move the routed-FFN optimization loop from wrapper-level tweaks to the actual
expert GEMM boundary. Before modifying TurboMind SM70 templates, make the
standalone gate/up benchmark cover the production served shape from the
Sprint 126 profile:

```text
avg_tokens=16
avg_routes=96
avg_active_experts=6
max_routes_expert=16
```

The benchmark must provide a fast 1-GPU V100 signal for any DS4-specific
software-pipelined gate/up mainloop work.

## Implementation

`test_ggml_turbomind_grouped_gate_up_fusion` now includes the production
`tokens_per_active=16` case by default. It also accepts environment controls so
V100 runs can target one shape without editing source:

```text
DS4_TURBOMIND_GATE_UP_CASES=16
DS4_TURBOMIND_GATE_UP_WARMUP_ITERS=N
DS4_TURBOMIND_GATE_UP_BENCH_ITERS=N
```

The benchmark still compares the three meaningful baselines:

- separate gate + up grouped GEMMs;
- fused gate_up grouped GEMM;
- interleaved gate_up with TurboMind gated-SiLU epilogue.

The gated-SiLU absolute tolerance was widened from `8.0` to `16.0` while
preserving the `1e-3` relative-error gate. Sprint 127 already established that
the fused gated epilogue should be judged primarily by relative error because
it applies SiLU before the separate half materialization used by the reference.

## Definition of Done

- The benchmark builds on the V100 pod.
- The default benchmark still runs the historical small cases.
- `DS4_TURBOMIND_GATE_UP_CASES=16` runs only the production-shaped case.
- The production-shaped case passes correctness.
- The sprint records separate, fused, and gated timings for the production
  shape.
- No appliance default changes unless the benchmark result justifies them.

## Notes

This sprint intentionally does not promote a runtime flag. It creates the
measurement harness needed for a real DS4/V100 software-pipeline kernel rather
than continuing to infer from full-server tok/s noise.

## Validation

Build on the V100 pod:

```text
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc \
  'cd /workspace/ds4 && cmake --build build/turbomind-v100-s127 \
   --target ggml-turbomind test_ggml_turbomind_grouped_gate_up_fusion -j80'
```

Historical cases:

```text
DS4_TURBOMIND_GATE_UP_CASES=1,4,8 DS4_TURBOMIND_GATE_UP_BENCH_ITERS=30 \
  build/turbomind-v100-s127/test_ggml_turbomind_grouped_gate_up_fusion \
  build/turbomind-v100-s127/libggml-turbomind.so
```

| Tokens/active expert | Total routes | Separate gate+up | Fused gate_up | Gated-SiLU | Gated speedup | Correctness |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 6 | `0.2445 ms` | `0.1710 ms` | `0.1677 ms` | `1.458x` | PASS |
| 4 | 24 | `0.2533 ms` | `0.1687 ms` | `0.1698 ms` | `1.491x` | PASS |
| 8 | 48 | `0.2236 ms` | `0.1533 ms` | `0.1517 ms` | `1.474x` | PASS |

Default invocation also passed all four default cases (`1,4,8,16`) with
`DS4_TURBOMIND_GATE_UP_BENCH_ITERS=30`.

Production-shaped case:

```text
DS4_TURBOMIND_GATE_UP_CASES=16 \
DS4_TURBOMIND_GATE_UP_WARMUP_ITERS=5 \
DS4_TURBOMIND_GATE_UP_BENCH_ITERS=100 \
  build/turbomind-v100-s127/test_ggml_turbomind_grouped_gate_up_fusion \
  build/turbomind-v100-s127/libggml-turbomind.so
```

| Tokens/active expert | Total routes | Separate gate+up | Fused gate_up | Gated-SiLU | Gated speedup | Correctness |
|---:|---:|---:|---:|---:|---:|---|
| 16 | 96 | `0.2889 ms` | `0.1775 ms` | `0.1776 ms` | `1.626x` | PASS |

## Decision

The production-shaped standalone result confirms that fused/interleaved
gate_up with the gated-SiLU epilogue is already a strong primitive. The next
optimization should not be another gate/up launch fusion. It should either:

- specialize the SM70 packed MXFP4 mainloop/epilogue beyond TurboMind's current
  general grouped GEMM path; or
- change scheduling so those fast standalone kernels see enough steady work in
  the served appliance.

The smallest credible implementation slice is a side-by-side DS4-only probe
beside `ggml-turbomind`, not a generic TurboMind registry rewrite. It should
consume the already-packed TurboMind MXFP4 operands and DS4 compact offsets for
the fixed V100 shape (`K=4096`, `mid=2048`, fused `N=4096`, group size `32`,
six active experts, 96 routes), then compare against the existing gated path
in this benchmark before any appliance integration.
