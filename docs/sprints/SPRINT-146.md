# Sprint 146 - 1536-Route Fixed Probe

Date: 2026-05-21

## Objective

Test whether the Sprint 139 fixed-shape TurboMind probe strategy scales from
the 128-slot compact routed shape to the 256-slot compact routed shape.
At 256 active slots, the compact routed FFN presents `256 * 6 = 1536` routed
rows, so this sprint adds explicit 1536-route gate/up and down probes and
checks them against full-appliance served throughput with prompt/prefill and
continuation/decode metrics split.

## Implementation

- Added exported TurboMind C ABI probes:
  - `ggml_turbomind_ds4_mxfp4_gated_silu_1536_m128`
  - `ggml_turbomind_ds4_mxfp4_down_1536_m128`
- Wired the DS4 CUDA loader to discover the new symbols.
- Kept the 1536-route probes explicit opt-ins only:
  - `DS4_V100_TURBOMIND_GATE_UP_PROBE=m128_1536`
  - `DS4_V100_TURBOMIND_DOWN_PROBE=m128_1536`
- Left production `auto` on the previously validated path. Generic `m128` also
  does not select the 1536-route probe, to avoid accidentally promoting a
  served-path-neutral experimental kernel.
- Extended the TurboMind standalone gate/up fusion test to include the
  256-slot/1536-route compact case.

## V100 Validation

Build:

```text
cmake --build build/turbomind-v100-s127 \
  --target ggml-turbomind test_ggml_turbomind_grouped_gate_up_fusion -j80
make -j80 CUDA_ARCH=sm_70 \
  tools/ds4-v100-replay tests/cuda_v100_full_scheduler_smoke
```

Standalone 1536-route probe validation:

| Probe | Generic ms | Probe ms | Correctness |
|---|---:|---:|---|
| gate/up gated-SiLU `m128_1536` | `0.9651` | `0.9435` | `probe_bad=0/3145728` |
| down `m128_1536` | `0.4536` | `0.4534` | `down_probe_bad=0/6291456` |

Full scheduler smoke passed with the 1536 gate probe enabled:

```text
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=256 \
layers=43 tm_layers=43 last=40-42 gpu=7 uploaded_tensors=8 \
uploaded_bytes=156142896212 expert_last=26 ok
```

Post-policy smoke also passed with production `auto`, after the 1536 probe was
made explicit opt-in only:

```text
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=256 \
layers=43 tm_layers=43 last=40-42 gpu=7 uploaded_tensors=8 \
uploaded_bytes=156142896212 expert_last=26 ok
```

Served 256-slot/16K A/B:

| Mode | Generated tok/s | Prompt tok/s | Continuation tok/s | Avg latency ms | Correctness |
|---|---:|---:|---:|---:|---|
| control, 1536 probe off | `61.223893` | `68.876880` | `57.397400` | `66513.060` | 256/256 |
| gate `m128_1536` candidate | `61.204203` | `68.854728` | `57.378940` | `66541.518` | 256/256 |

Artifacts:

- `logs/from-cluster/sprint146-standalone-1536/`
- `logs/from-cluster/sprint146-smoke-gate1536/`
- `logs/from-cluster/sprint146-smoke-auto-postoptin/`
- `logs/from-cluster/sprint146-optin-validation/`
- `logs/from-cluster/sprint146-served-256slot-16k-control/`
- `logs/from-cluster/sprint146-served-256slot-16k-gate1536/`

## Decision

Keep the 1536-route probes as explicit opt-ins only. The gate/up probe is
correct and slightly faster in isolation, but the served path is flat to
slightly worse: continuation/decode moved from `57.397400` to `57.378940`
tok/s.

This reinforces the current performance diagnosis. Small fixed-shape launches,
tile tweaks, and epilogue fusions can improve a microbenchmark, but they do not
materially change the appliance topline unless they remove a larger served-path
boundary. The next meaningful implementation target is still a larger
software-pipelined routed-FFN executor: packed MXFP4 load/dequant, gate/up
HMMA, gated activation, down HMMA, and weighted reduce/scatter in one
overlapped path, or a scheduler change that feeds materially larger expert
microbatches without losing the existing stage overlap.
