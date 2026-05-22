# Sprint 169 Explicit-Stream Graph Evidence

Date: 2026-05-21

Cluster pod: `llm/llamacpp-build-8gpu`

Appliance pack: `/workspace/ds4-appliance-full-tm-gated-s127`

Runtime flags:

```text
DS4_V100_TURBOMIND_LIB=./build/turbomind-v100-s127/libggml-turbomind.so
DS4_V100_TURBOMIND_GATED_SILU=1
DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1
DS4_V100_TURBOMIND_ROUTED_EXECUTOR=off
```

Served A/B:

```text
tools/ds4-v100-appliance-soak.sh
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
| graph off | `46.316374` | `43.421600` | `52.105920` | `16/16` |
| graph on | `40.058341` | `37.554695` | `45.065634` | `16/16` |

Graph-on server log:

```text
turbomind_graph captured=43
begin_capture_failed=0
launch_failed=0
```

Decision: graph capture now works and is correct, but served throughput
regresses. Keep `DS4_V100_TURBOMIND_GRAPH=1` as a diagnostic opt-in and move
future work to a larger persistent routed-FFN executor or persistent TP/EP
boundary.
