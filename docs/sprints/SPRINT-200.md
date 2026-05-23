# Sprint 200 - Persistent Six-Route Routed FFN Kernel Cut-In

Date: 2026-05-23
Status: Planned

## Objective

Start the next material serving optimization after Sprint 199 by implementing a
real routed-FFN kernel boundary, not another wrapper-level scheduler tweak.

The target is the production decode shape:

```text
total_routes = 6
active_experts = 6
max_routes_per_expert = 1
hidden = 4096
mid = 2048
weights = interleaved MXFP4 gate/up + MXFP4 down
activation boundary = FP16/HMMA inside the kernel only
```

## Rationale

Sprint 199 shipped a real serving win by promoting `fused6_reduce + graph`, but
the topline is still only about `66.825545` aggregate continuation tok/s at
16-slot/256K. The remaining gap is too large for more launch replay or scratch
staging changes.

The latest evidence says:

- graph replay reduces launch overhead and is now default for the production
  pack;
- `fused6_reduce` already removes route-expanded activation staging and
  `down_routes`;
- Sprint 197 proves the remaining `mid_half` materialization is only `24576`
  bytes per six-route call, so a buffer-only optimization is not enough;
- gate/up and down execution still account for most routed-FFN time;
- TP4 collectives are correct but not a decode win unless fused into a larger
  layer/FFN boundary.

Sprint 200 should therefore cut into the TurboMind kernel boundary itself.

## Scope

1. Add a new default-off TurboMind ABI candidate for the production six-route
   routed FFN boundary.
2. Keep packed MXFP4 weights resident and do all expansion/dequantization
   inside GPU code.
3. Use FP16 HMMA/Volta tensor-core math internally; do not introduce global
   FP16 model storage or broad FP32 GEMMs.
4. Fuse at least one boundary that Sprint 199 still pays separately:
   - gate/up output tile to gated activation; or
   - gated activation tile to down accumulation; or
   - route-weighted down accumulation to final hidden row.
5. Add a focused TurboMind test/bench that runs the exact production six-route
   shape and compares against the current `gated_silu_6 + down_6_m16_reduce`
   path.
6. Integrate the candidate as an explicit runtime mode only if the focused
   bench is correct.

## Non-Goals

- No default promotion in this sprint unless served A/B clears a material gate.
- No model/pack format change.
- No tensor-parallel production path.
- No MTP changes.
- No global FP16 expansion of packed low-bit weights.

## Implementation Map

Likely files:

| File | Work |
|---|---|
| `kernels/turbomind/ggml-turbomind/ggml-turbomind-ds4-probe.cu` | Add the six-route fused/persistent kernel candidate and launch wrapper. |
| `kernels/turbomind/ggml-turbomind/api.cc` | Export the new C ABI and validate shape/dtype constraints. |
| `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h` | Declare the ABI. |
| `kernels/turbomind/ggml-turbomind/test_grouped_gate_up_fusion.cpp` or new test | Add exact-shape correctness/bench coverage against the current two-ABI baseline. |
| `ds4_cuda.cu` | Add a default-off routed executor mode only after the TurboMind candidate is correct. |
| `tools/ds4-v100-run-appliance.sh` | Add launcher allowlist only if runtime integration is added. |

Existing entry points to use as references:

- `ggml_turbomind_ds4_mxfp4_gated_silu_6`
- `ggml_turbomind_ds4_mxfp4_down_6_m16_reduce`
- `DS4_CUDA_TM_ROUTED_EXECUTOR_FUSED6_REDUCE`
- `cuda_tm_routed_mxfp4_packed_impl`

## Definition Of Done

- [ ] V100 TurboMind build passes for the new candidate.
- [ ] New ABI is exported by `libggml-turbomind.so`.
- [ ] Focused six-route correctness test matches the current baseline within
      the established tolerance.
- [ ] Focused six-route bench records candidate vs baseline kernel time.
- [ ] If focused bench is positive, runtime integration is added as an explicit
      non-default executor mode.
- [ ] If runtime integration is added, selected-token smoke passes.
- [ ] If selected-token smoke passes, run 16-slot/256K served A/B against the
      Sprint 199 promoted baseline.
- [ ] Keep the candidate diagnostic-only unless served continuation tok/s
      improves materially over Sprint 199.
- [ ] Vision/status artifacts are updated.
- [ ] Changes are committed.

## Stop Condition

Stop this branch if the implementation cannot fuse any real kernel boundary
beyond the current graph-backed `gated_silu_6 + down_6_m16_reduce` sequence. A
new ABI that only calls the same two kernels under one C function is not enough;
in that case the next sprint should pivot to bounded full-layer TP4/EP.
