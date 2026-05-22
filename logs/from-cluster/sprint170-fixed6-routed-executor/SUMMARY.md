# Sprint 170 Six-Route Routed Executor Evidence

Date: 2026-05-22

Cluster pod: `llm/llamacpp-build-8gpu` (gpu-01, 8x V100)

Appliance pack: `/workspace/ds4-appliance-full-tm-gated-s127`

TurboMind lib rebuilt with `gated_silu_6`:
`./build/turbomind-v100-s127/libggml-turbomind.so`

Build:

```text
cmake --build build/turbomind-v100-s127 --target ggml-turbomind -j80
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
nm -D build/turbomind-v100-s127/libggml-turbomind.so | grep gated_silu_6
  -> 0000000000031250 T ggml_turbomind_ds4_mxfp4_gated_silu_6
```

Runtime flags:

```text
DS4_V100_TURBOMIND_LIB=./build/turbomind-v100-s127/libggml-turbomind.so
DS4_V100_TURBOMIND_GATED_SILU=1
DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1
DS4_V100_TURBOMIND_ROUTED_EXECUTOR=off | fixed6
DS4_V100_TURBOMIND_ROUTED_EXECUTOR_VERBOSE=1   (fixed6 arm)
```

Served A/B (`tools/ds4-v100-appliance-soak.sh`):

```text
--ctx 262144
--slots 16
--active-microbatch 16
--queue-policy sequential
--tokens 16
--requests 16
--warmup-requests 1
--async-pipeline-mode per-step
--async-event-handoff 1
```

| Mode | Generated tok/s | Continuation tok/s | Prompt tok/s | Token match |
|---|---:|---:|---:|---:|
| control (executor off) | `44.454879` | `41.676449` | `50.011739` | `16/16` |
| fixed6 | `44.344945` | `41.573386` | `49.888064` | `16/16` |

fixed6 selection (server log, verbose):

```text
ds4: TurboMind routed executor fixed6 shape total_routes=6 active_experts=6 max_routes_per_expert=1
ds4: TurboMind routed executor selected fixed gate_up total_routes=6
```

Decision: `fixed6` is selected and correct on the real served 6-route shape, but
served throughput is flat-to-slightly-slower versus same-binary control.
Dispatch bypass is not the missing lever -- now established at the actual served
shape, not just the scheduler-coalesced 96-route shape. Keep `fixed6` an
explicit opt-in diagnostic; move the next implementation to a persistent/fused
six-route routed-FFN executor or a persistent TP/EP boundary.

Artifacts: `control/`, `fixed6/` (soak summaries, per-response JSON, server
logs, runtime startup env, GPU utilization samples).
