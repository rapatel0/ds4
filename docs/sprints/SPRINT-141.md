# Sprint 141 - Half2 Route-Row Reduce Probe

Date: 2026-05-21

## Objective

Test whether the routed-FFN tail can be improved without changing TurboMind's
generic GEMM epilogue ABI. Sprint 140 showed that single-GEMM fixed-shape
specialization is not a material served-throughput lever, so this sprint
targets the `down_routes` reduce/scatter tail with a small opt-in vectorized
kernel.

## Implementation

- Added half2-vectorized by-pair reduce kernels:
  - `tm_reduce_sum_half2_to_f32_by_pair_kernel`
  - `tm_reduce_sum_weighted_half2_to_f32_by_pair_kernel`
- Added `DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE_H2=1`.
- The new path only engages when:
  - `DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=1`
  - hidden width is even
  - the existing pair-to-sorted-row map is available
- Kept the production default `off`.

Files:

- `ds4_cuda.cu`
- `tools/ds4-v100-run-appliance.sh`
- `deploy/v100/ds4-v100-appliance.env.example`
- `docs/operations/DS4-V100-APPLIANCE.md`

## V100 Validation

Build:

```text
make -j80 CUDA_ARCH=sm_70 ds4 ds4-server tools/ds4-v100-replay \
  tests/cuda_v100_full_scheduler_smoke
```

Full 43-layer production smoke with route-row-reduce plus half2 reduce:

```text
DS4_V100_TURBOMIND_GATED_SILU=1
DS4_V100_TURBOMIND_GATE_UP_PROBE=auto
DS4_V100_TURBOMIND_DOWN_PROBE=off
DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1
DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=1
DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE_H2=1
tests/cuda_v100_full_scheduler_smoke \
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
| control, route-row reduce off | 32,768 | 128 | `60.108232` | `56.351468` | 128/128 token match |
| scalar route-row reduce repeat | 32,768 | 128 | `60.112248` | `56.355232` | 128/128 token match |
| half2 route-row reduce | 32,768 | 128 | `60.104512` | `56.347980` | 128/128 token match |
| indexed-A repeat | 32,768 | 128 | `60.056960` | `56.303400` | 128/128 token match |
| route-row reduce earlier repeat | 32,768 | 128 | `60.022743` | `56.271322` | 128/128 token match |

## Decision

Keep `DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE_H2=off` by default.

The half2 reduce is correct, but it is not a served-throughput win. The scalar
route-row reduce, half2 route-row reduce, and control all landed inside the
same narrow 128-slot band. This is useful evidence: optimizing the separate
tail kernel is not enough. The next material implementation needs to change the
larger routed-FFN execution boundary, most likely a DS4-specific TurboMind down
epilogue that applies route weights and accumulates directly into
`[token, hidden]`, or a persistent routed-FFN executor that avoids materializing
and rereading the full `down_routes` surface.
