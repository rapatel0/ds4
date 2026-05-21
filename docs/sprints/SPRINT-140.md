# Sprint 140 - Fixed-Shape 128-Slot Down Probe

Date: 2026-05-21

## Objective

After Sprint 139 proved that a fixed-shape 768-route gate/up kernel improves
the isolated microbenchmark but barely moves served throughput, test the next
largest routed-FFN bucket: the expert down projection for the same compact
128-slot shape.

## Implementation

- Added a fixed DS4/V100 TurboMind MXFP4 down-projection ABI:
  - `ggml_turbomind_ds4_mxfp4_down_768_m128`
- The fixed down path is guarded to the exact production shape:
  - route-expanded activation input
  - six compact expert groups
  - `total_routes = 768`
  - `N = 4096`, `K = 2048`
- Added production selection through:
  - `DS4_V100_TURBOMIND_DOWN_PROBE=auto|off`
- Kept the production default `off` after served A/B showed the probe was
  slower end-to-end.
- Extended the TurboMind compact gate/up benchmark so the same harness also
  measures the fixed down shape with a bounded FP16 mid-activation fixture.

Files:

- `ds4_cuda.cu`
- `tools/ds4-v100-run-appliance.sh`
- `deploy/v100/ds4-v100-appliance.env.example`
- `docs/operations/DS4-V100-APPLIANCE.md`
- `kernels/turbomind/ggml-turbomind/api.cc`
- `kernels/turbomind/ggml-turbomind/ggml-turbomind-ds4-probe.cu`
- `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h`
- `kernels/turbomind/ggml-turbomind/test_grouped_gate_up_fusion.cpp`

## V100 Validation

Profile baseline for the current 128-slot compact gated path:

```text
DS4_V100_TURBOMIND_PROFILE=1
tests/cuda_v100_full_scheduler_smoke --ctx 32768 --slots 128
```

The natural-exit full scheduler profile showed, per GPU, roughly:

```text
gate_up_pct ~= 57-60%
down_pct    ~= 29-30%
```

That made down projection the next bounded kernel target after gate/up.

Builds:

```text
cmake --build build/turbomind-v100-s127 \
  --target ggml-turbomind test_ggml_turbomind_grouped_gate_up_fusion -j80

make -j80 CUDA_ARCH=sm_70 ds4 ds4-server tools/ds4-v100-replay \
  tests/cuda_v100_full_scheduler_smoke
```

Microbenchmark:

```text
DS4_TURBOMIND_GATE_UP_COMPACT_GROUPS=1
DS4_TURBOMIND_GATE_UP_CASES=128
DS4_TURBOMIND_GATE_UP_PROBE=m128
DS4_TURBOMIND_DOWN_PROBE=auto
```

Result:

| Shape | Generic | Fixed probe | Correctness |
|---|---:|---:|---|
| gated gate/up, 768 routes | `0.6482 ms` | `0.5996 ms` | PASS |
| down, 768 routes | `0.3272 ms` | `0.3026 ms` | PASS |

Full 43-layer production smoke with both fixed probes enabled:

```text
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=128 \
layers=43 tm_layers=43 last=40-42 gpu=7 uploaded_tensors=8 \
uploaded_bytes=156142896212 expert_last=26 ok
```

Served 128-slot/32K A/B:

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| gated + gate-up probe + down probe auto | 32,768 | 128 | `60.038469` | `56.286064` | 128/128 token match |
| gated + gate-up probe + down probe off | 32,768 | 128 | `60.129772` | `56.371661` | 128/128 token match |

## Decision

Keep `DS4_V100_TURBOMIND_DOWN_PROBE=off` by default.

The fixed down kernel is correct and faster in isolation, but the served path
is slightly slower with it enabled. This confirms the Sprint 139 direction:
more single-GEMM fixed-shape tuning is not the missing throughput lever. The
next material step should fuse or reschedule a larger routed-FFN boundary, most
likely down epilogue plus weighted route reduce, or a persistent grouped
executor that reduces launch/stream boundaries without losing stage overlap.
