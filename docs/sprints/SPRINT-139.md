# Sprint 139 - Fixed-Shape 128-Slot Gate/Up Probe

Date: 2026-05-21

## Objective

Turn the Sprint 138 wide compact gate/up benchmark into a production-selectable
probe for the 128-slot served route shape. The goal was to test whether a
larger fixed SM70 TurboMind MXFP4 CTA-M family can beat the generic compact
gated-SiLU path for the appliance's high-slot shape.

## Implementation

- Added fixed DS4/V100 TurboMind MXFP4 gated-SiLU probe ABIs for the 768-route
  compact shape:
  - `ggml_turbomind_ds4_mxfp4_gated_silu_768_m64`
  - `ggml_turbomind_ds4_mxfp4_gated_silu_768_m128`
- Kept the older 96-route fixed probe as the m16 shape.
- Extended the gate/up benchmark to select `m64`, `m128`, or `off` via
  `DS4_TURBOMIND_GATE_UP_PROBE`.
- Wired the production appliance path to select the 768-route m128 probe under
  exact guards:
  - interleaved gated-SiLU gate/up pack is active
  - route-expanded activation input, not indexed-A
  - compact schedule presents six expert groups
  - `total_routes = 768`
  - `K = 4096`, `N = 8192`
- Added launcher/env support:
  - `DS4_V100_TURBOMIND_GATE_UP_PROBE=auto|off|m64|m128`
  - default `auto`, which selects m128 when the production guards match and
    otherwise falls back to generic TurboMind.

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

Builds:

```text
cmake --build build/turbomind-v100-s127 -j80
make -j80 CUDA_ARCH=sm_70 ds4 ds4-server tools/ds4-v100-replay \
  tests/cuda_v100_full_scheduler_smoke
```

Wide compact 768-route benchmark:

| Variant | Separate gate+up | Fused gate_up | Gated-SiLU | Probe | Correctness |
|---|---:|---:|---:|---:|---|
| control, probe off | `0.6930 ms` | `0.6380 ms` | `0.6482 ms` | n/a | PASS |
| m64 probe | `0.8120 ms` | `0.7546 ms` | `0.7242 ms` | `0.6955 ms` | PASS |
| m128 probe | `0.7335 ms` | `0.6385 ms` | `0.6480 ms` | `0.5999 ms` | PASS |

The m128 fixed-shape probe beats the same-run generic gated path by about
`1.08x` and the same-run fused gate_up path by about `1.06x` for the isolated
768-route compact gate/up microbenchmark.

Full 43-layer production smoke on the interleaved gated appliance:

```text
DS4_V100_TURBOMIND_GATED_SILU=1
DS4_V100_TURBOMIND_GATE_UP_PROBE=auto
DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1
tests/cuda_v100_full_scheduler_smoke \
  --appliance-dir /workspace/ds4-appliance-full-tm-gated-s127 \
  --ctx 32768 --slots 128 --expect-tm-layers 43
```

Result:

```text
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=128 \
layers=43 tm_layers=43 last=40-42 gpu=7 uploaded_tensors=8 \
uploaded_bytes=156142896212 expert_last=26 ok
```

Served 128-slot/32K A/B on the interleaved gated appliance:

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| gated + m128 auto probe | 32,768 | 128 | `60.130047` | `56.371919` | 128/128 token match |
| gated + probe off | 32,768 | 128 | `60.061899` | `56.308030` | 128/128 token match |

## Decision

Keep `DS4_V100_TURBOMIND_GATE_UP_PROBE=auto` wired into the appliance because
the shape guard is exact and the isolated kernel win is real. Do not treat it
as a major throughput step: served improvement over probe-off is only about
`0.1%`.

This sprint confirms the answer to the fusion question: fusing packed
load/dequant/HMMA/epilogue with a better software-pipelined tile shape helps,
but gate/up alone is too small a remaining end-to-end lever. The next material
optimization needs to cover a larger routed-FFN boundary, most likely
gate/up plus activation plus down plus weighted scatter/reduce, or a scheduler
that keeps expert work larger without losing stage overlap.
