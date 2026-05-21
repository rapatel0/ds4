# Sprint 138 - Wide Compact Gate/Up Kernel Baseline

Date: 2026-05-21

## Objective

After Sprint 137 showed admission-width scaling is positive but diminishing,
establish the kernel-side acceptance baseline for the next software-pipelined
packed MXFP4 expert work. The benchmark must cover the compact served route
shapes created by high-slot serving, not only the older 96-route case.

## Implementation

Updated the TurboMind gate/up benchmark default case list to include high-slot
compact route shapes:

```text
tokens_per_active = 1, 4, 8, 16, 32, 64, 128
```

In compact mode, the benchmark uses six active expert groups, so these map to:

```text
6, 24, 48, 96, 192, 384, 768 total routed rows
```

This gives future kernel probes a stable local acceptance harness for the
current 32-slot, 64-slot, and 128-slot serving shapes.

File:

- `kernels/turbomind/ggml-turbomind/test_grouped_gate_up_fusion.cpp`

## V100 Validation

Build:

```text
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc \
  'cd /workspace/ds4 && cmake --build build/turbomind-v100-s127 \
   --target test_ggml_turbomind_grouped_gate_up_fusion -j80'
```

Wide compact benchmark:

```text
DS4_TURBOMIND_GATE_UP_COMPACT_GROUPS=1 \
DS4_TURBOMIND_GATE_UP_CASES=16,32,64,128 \
DS4_TURBOMIND_GATE_UP_WARMUP_ITERS=5 \
DS4_TURBOMIND_GATE_UP_BENCH_ITERS=100 \
  build/turbomind-v100-s127/test_ggml_turbomind_grouped_gate_up_fusion \
  build/turbomind-v100-s127/libggml-turbomind.so
```

| Routes | Tokens/active expert | Separate gate+up | Fused gate_up | Gated-SiLU | Best current |
|---:|---:|---:|---:|---:|---:|
| 96 | 16 | `0.2078 ms` | `0.1898 ms` | `0.1911 ms` | `0.1898 ms` |
| 192 | 32 | `0.2580 ms` | `0.1918 ms` | `0.1823 ms` | `0.1823 ms` |
| 384 | 64 | `0.3620 ms` | `0.3487 ms` | `0.3543 ms` | `0.3487 ms` |
| 768 | 128 | `0.6929 ms` | `0.6379 ms` | `0.6481 ms` | `0.6379 ms` |

Correctness passed for all cases. The widened default list was also validated
with a shorter 10-iteration compact run, and all default cases from
`tokens_per_active=1` through `128` passed.

## Decision

Ship the wider benchmark default.

The baseline says the current TurboMind compact grouped path is already fairly
efficient at the larger route counts, and simple gate/up/gated epilogue fusion
does not create another large end-to-end lever. A useful Sprint 139 kernel
needs to beat about `0.638 ms` for the 768-route compact MXFP4 gate/up shape
or reduce the surrounding routed-FFN stages enough to move served throughput.
