# Sprint 142 - TurboMind Down-Epilogue Reduce Probe

Date: 2026-05-21

## Objective

Test the next larger routed-FFN fusion boundary after Sprint 141 showed that a
separate half2 route-row reduce tail is not enough. The target is a
DS4-specific TurboMind down epilogue that applies route weights and accumulates
directly into `[token, hidden]`, avoiding the normal `down_routes` materialize
and re-read path.

## Implementation

- Extended TurboMind `EpilogueParam` with default-inactive DS4 route-reduce
  fields.
- Added a guarded DS4 route-reduce epilogue path in the SM70 GEMM epilogue.
- Added a fixed-shape DS4/V100 MXFP4 down-reduce ABI:
  - `ggml_turbomind_ds4_mxfp4_down_768_m128_reduce`
- Wired the runtime under exact production guards:
  - interleaved gated-SiLU path active
  - compact routed schedule
  - `total_routes = 768`
  - `num_experts = 6`
  - `hidden = 4096`
  - `mid = 2048`
  - no indexed-A token-indirection path
- Added `DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=1` as an opt-in diagnostic.
- Kept the production default `off`.

Files:

- `ds4_cuda.cu`
- `tools/ds4-v100-run-appliance.sh`
- `deploy/v100/ds4-v100-appliance.env.example`
- `docs/operations/DS4-V100-APPLIANCE.md`
- `kernels/turbomind/ggml-turbomind/api.cc`
- `kernels/turbomind/ggml-turbomind/ggml-turbomind-ds4-probe.cu`
- `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h`
- `kernels/turbomind/lmdeploy/src/turbomind/kernels/gemm/epilogue.h`
- `kernels/turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel_impl.h`

## V100 Validation

TurboMind build:

```text
cmake --build build/turbomind-v100-s127 --target ggml-turbomind -j80
```

Appliance build:

```text
make -j80 CUDA_ARCH=sm_70 ds4 ds4-server tools/ds4-v100-replay \
  tests/cuda_v100_full_scheduler_smoke
```

Full 43-layer 128-slot smoke with the down epilogue reduce enabled:

```text
DS4_V100_TURBOMIND_GATED_SILU=1
DS4_V100_TURBOMIND_GATE_UP_PROBE=auto
DS4_V100_TURBOMIND_DOWN_PROBE=off
DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1
DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=1
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

Served 128-slot/32K same-binary A/B on the interleaved gated appliance:

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| control, down-reduce epilogue off | 32,768 | 128 | `59.987105` | `56.237910` | 128/128 token match |
| down-reduce epilogue opt-in | 32,768 | 128 | `60.041003` | `56.288440` | 128/128 token match |

Artifacts:

- `logs/from-cluster/sprint142-control-128/`
- `logs/from-cluster/sprint142-down-reduce-epilogue-128/`

## Decision

Keep `DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=off` by default.

The fused epilogue path is correct and proves we can route a DS4-specific
weighted reduce through TurboMind's GEMM epilogue, but the served result is
only about `+0.09%` over same-binary control. This atomic-add epilogue is not a
material throughput step.

The useful follow-up is not another standalone tail kernel. The next kernel
should avoid the atomic epilogue shape entirely: either a persistent routed-FFN
executor or a CUTLASS/TurboMind-style software-pipelined block that stages
packed MXFP4 weights/scales, gate/up HMMA, activation, down HMMA, and final
per-token reduction in a way that keeps tensor-core work larger and reduces
global-memory round trips.
