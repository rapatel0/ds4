# Sprint 133 - Compact-Group Gate/Up Benchmark Correction

Date: 2026-05-21

## Objective

Correct the standalone TurboMind gate/up benchmark so it can match the served
runtime's compact active-expert schedule. Sprint 132 added the right route
count, but the benchmark still used a sparse 256-expert offset table. The
production runtime defaults to compact grouped scheduling after Sprint 128, so
future DS4-only kernel probes need a compact 6-group benchmark baseline.

## Implementation

`test_ggml_turbomind_grouped_gate_up_fusion` now accepts:

```text
DS4_TURBOMIND_GATE_UP_COMPACT_GROUPS=1
```

When enabled:

- the benchmark builds six compact groups instead of a sparse 256-expert table;
- the active expert table is `[0,1,2,3,4,5]`;
- offsets length is `7`, matching the compact active-expert schedule used by
  the served appliance path;
- the printed shape line reports `group_mode=compact` or `group_mode=sparse256`.

This is a benchmark-only change. No appliance runtime defaults changed.

## V100 Validation

Build:

```text
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc \
  'cd /workspace/ds4 && cmake --build build/turbomind-v100-s127 \
   --target test_ggml_turbomind_grouped_gate_up_fusion -j80'
```

Focused 96-route A/B:

```text
for compact in 0 1; do
  DS4_TURBOMIND_GATE_UP_COMPACT_GROUPS=$compact \
  DS4_TURBOMIND_GATE_UP_CASES=16 \
  DS4_TURBOMIND_GATE_UP_WARMUP_ITERS=5 \
  DS4_TURBOMIND_GATE_UP_BENCH_ITERS=100 \
    build/turbomind-v100-s127/test_ggml_turbomind_grouped_gate_up_fusion \
    build/turbomind-v100-s127/libggml-turbomind.so
done
```

| Group mode | Groups | Total routes | Separate gate+up | Fused gate_up | Gated-SiLU | Gated speedup | Correctness |
|---|---:|---:|---:|---:|---:|---:|---|
| sparse256 | 256 | 96 | `0.3263 ms` | `0.2124 ms` | `0.2128 ms` | `1.534x` | PASS |
| compact | 6 | 96 | `0.1895 ms` | `0.1729 ms` | `0.1740 ms` | `1.089x` | PASS |

## Decision

This changes the interpretation of the standalone speedup. Fused/interleaved
gate_up is a large win against sparse 256-expert scheduling, but the served
runtime already uses compact scheduling. At the production compact shape,
existing TurboMind gated-SiLU is only about `1.09x` faster than separate
gate+up.

The next DS4-only kernel probe must beat the compact `0.1740 ms` gated baseline
at 96 routes, not the sparse `0.2128 ms` baseline. A new probe that only removes
sparse-group overhead is not useful for the current served appliance.
