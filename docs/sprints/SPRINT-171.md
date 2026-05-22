# Sprint 171 - Six-Route Down-Reduce Epilogue

Date: 2026-05-22

## Objective

Extend the existing TurboMind DS4 down-projection route-reduce epilogue to the
production served 6-route decode shape:

```text
ctx = 262144
slots = 16
async_pipeline_mode = per-step
per request routed FFN shape:
  total_routes = 6
  active_experts = 6
  max_routes_per_expert = 1
```

Sprint 170 proved that fixed6 gate/up dispatch bypass is flat on this exact
shape. This sprint tests the next smaller production boundary: fuse down GEMM
output accumulation directly into the F32 token output for `total_routes=6`,
avoiding the intermediate `down_routes` write plus weighted scatter kernel.

## Scope

- Add `ggml_turbomind_ds4_mxfp4_down_6_m16_reduce` to the TurboMind C ABI.
- Reuse the existing SM70 M16 MXFP4 probe kernel family.
- Select it behind the existing
  `DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=1` flag only when:
  - `total_routes == 6`
  - `num_experts == 6`
  - no token-index indirection is required
  - `N == 4096`
  - `K == 2048`
- Keep defaults unchanged.
- Build the TurboMind library and replay binary on the V100 pod.
- Run the production 16-slot/256K served A/B with per-step async/event
  handoff.

## Non-Goals

- No full gate/up + down persistent executor yet.
- No MTP changes.
- No TP/EP scheduler changes.
- No default promotion unless the served A/B clearly improves.

## Definition of Done

- [x] TurboMind shared library builds on the V100 pod.
- [x] `tools/ds4-v100-replay` builds on the V100 pod.
- [x] New symbol is exported from `libggml-turbomind.so`.
- [x] 6-route down-reduce selection is visible in a served or replay log, or
      equivalent profiler evidence confirms the epilogue path was used.
- [x] 16-slot/256K served A/B records generated and continuation tok/s.
- [x] Result is recorded in `docs/sprints/VISION.md` and cluster logs.
- [x] Changes are committed.

## Decision Gate

If the 6-route down-reduce epilogue improves served continuation throughput
outside run noise, keep it as an opt-in candidate and use it as evidence for a
broader fused routed-FFN boundary. If flat or slower, then even the exact
six-route down/scatter boundary is not enough; Sprint 172 should move to a true
persistent gate/up + down executor or a broader persistent TP/EP scheduler
boundary.

## Outcome

Implemented and validated on the V100 pod (`llm/llamacpp-build-8gpu`, gpu-01).

- Added `ggml_turbomind_ds4_mxfp4_down_6_m16_reduce` to the TurboMind C ABI
  (`api.cc`, `ggml-turbomind-ds4-probe.cu`,
  `include/ggml-turbomind-api.h`), reusing the SM70 M16 MXFP4 probe kernel.
- Wired `ds4_cuda.cu` so `DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=1` selects
  the new epilogue for `total_routes == 6`, `num_experts == 6`, no
  token-index indirection, `N == 4096`, and `K == 2048`.
- Rebuilt the TurboMind `.so` and `tools/ds4-v100-replay` at `sm_70`; the new
  symbol is exported:
  `T ggml_turbomind_ds4_mxfp4_down_6_m16_reduce`.
- A direct replay smoke selected the new path on the real model:
  `ds4: TurboMind down-reduce epilogue selected total_routes=6`.

Same-binary served A/B at 16-slot/256K, per-step async + event handoff:

| Mode | Generated tok/s | Continuation tok/s | Prompt tok/s | Token match |
|---|---:|---:|---:|---:|
| control | `45.941120` | `43.069800` | `51.683760` | `16/16` |
| down6 reduce | `43.887560` | `41.144588` | `49.373505` | `16/16` |

Evidence: `logs/from-cluster/sprint171-down6-reduce/`.

## Decision

The 6-route down-reduce epilogue is correct but regresses served throughput by
about `4.5%` on both generated and continuation/decode tok/s. Keep
`DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=0` as the production default.

This closes the exact six-route down/scatter epilogue line. The next
implementation should not be another epilogue or dispatch-bypass probe; it
should change the larger execution boundary: a persistent gate/up + down
routed-FFN executor, or a broader persistent TP/EP scheduler boundary.
