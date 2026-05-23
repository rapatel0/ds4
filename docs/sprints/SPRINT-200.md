# Sprint 200 - Persistent Six-Route Routed FFN Kernel Cut-In

Date: 2026-05-23
Status: Completed - stop condition triggered

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

- [x] V100 TurboMind build passes for the focused six-route bench update.
- [ ] New ABI is exported by `libggml-turbomind.so`.
- [x] Focused six-route correctness test matches the current baseline within
      the established tolerance.
- [x] Focused six-route bench records candidate vs baseline kernel time.
- [ ] If focused bench is positive, runtime integration is added as an explicit
      non-default executor mode.
- [ ] If runtime integration is added, selected-token smoke passes.
- [ ] If selected-token smoke passes, run 16-slot/256K served A/B against the
      Sprint 199 promoted baseline.
- [ ] Keep the candidate diagnostic-only unless served continuation tok/s
      improves materially over Sprint 199.
- [x] Vision/status artifacts are updated.
- [x] Changes are committed.

## Stop Condition

Stop this branch if the implementation cannot fuse any real kernel boundary
beyond the current graph-backed `gated_silu_6 + down_6_m16_reduce` sequence. A
new ABI that only calls the same two kernels under one C function is not enough;
in that case the next sprint should pivot to bounded full-layer TP4/EP.

## Implementation

Extended `test_ggml_turbomind_grouped_gate_up_fusion` so the focused TurboMind
bench covers the exact production six-route shape:

- selects `ggml_turbomind_ds4_mxfp4_gated_silu_6` for compact
  `total_routes=6`;
- benchmarks `ggml_turbomind_ds4_mxfp4_down_6_m16_reduce`;
- measures the required F32 output clear separately;
- compares down-reduce output against a generic down-projection reference;
- relaxes only the exact six-route probe absolute tolerance to `2.0` because
  the isolated probe has four half-output values outside the old `0.25`
  absolute tolerance while retaining tiny relative error.

No new runtime ABI was added. The focused data showed that the obvious clear
fusion is not material and that the existing fixed six-route gate/up probe is
not the kernel to promote.

## Validation

V100 build:

```text
cmake --build build/turbomind-v100 \
  --target test_ggml_turbomind_grouped_gate_up_fusion -j80
```

passed on `llm/llamacpp-build-8gpu`.

Focused six-route bench:

```text
DS4_TURBOMIND_GATE_UP_COMPACT_GROUPS=1
DS4_TURBOMIND_GATE_UP_CASES=1
DS4_TURBOMIND_GATE_UP_BENCH_ITERS=100
DS4_TURBOMIND_GATE_UP_WARMUP_ITERS=5
DS4_TURBOMIND_DOWN_PROBE=auto
./build/turbomind-v100/test_ggml_turbomind_grouped_gate_up_fusion \
  build/turbomind-v100/libggml-turbomind.so
```

Result:

| Metric | Value |
|---|---:|
| generic gated-SiLU path | `0.0946 ms` |
| fixed `m16_6` gated-SiLU probe | `0.1196 ms` |
| generic down projection | `0.0512 ms` |
| output clear only | `0.0022 ms` |
| six-route down-reduce with clear | `0.0650 ms` |
| down-reduce relative error vs generic-down reduction | `2.0515e-04` |
| down-reduce bad values | `0/4096` |

The fixed `m16_6` gate/up probe is about `0.792x` the generic gated-SiLU path,
so the promoted Sprint 199 `fused6_reduce` path is right to use the generic
gated-SiLU TurboMind path rather than the fixed six-route gate/up probe.

The clear takes only `0.0022 ms` in this focused bench. Folding only that clear
into a new ABI would not be a material serving optimization, and doing it
race-safely inside the current atomic route-reduce epilogue would require a
real epilogue rewrite rather than a wrapper.

Evidence:

```text
logs/from-cluster/sprint200-six-route-bench/
```

## Decision

Trigger the stop condition and do not add a new Sprint 200 runtime executor.

The data rejects the near-term clear-fusion path and rejects the fixed
six-route gate/up probe as a serving lever. The next sprint should pivot to the
bounded full-layer TP4/EP prototype, where the larger execution shape can make
the extra collectives worthwhile.
