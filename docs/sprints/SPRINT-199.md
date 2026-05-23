# Sprint 199 - Served Graph Replay Gate For Fused Routed Executor

Date: 2026-05-23
Status: Completed

## Objective

Decide whether the Sprint 198 graph replay support for the current
`fused6_reduce` routed executor is useful in the real served 16-slot/256K
appliance path.

## Context

Sprint 198 made `DS4_V100_TURBOMIND_GRAPH=1` compatible with
`DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6_reduce`. Direct replay matched token
IDs and improved continuation throughput from `16.022442` to `17.980888`
tok/s, with `43` captures, `129` launches, and `0` failures.

That is not enough to promote. Sprint 169 already showed that graph replay can
improve direct replay while regressing served throughput. This sprint runs the
missing production-shaped served A/B before spending more time on graph replay.

## Scope

- Build the current tree on the V100 pod.
- Run a same-binary 16-slot/256K served A/B:
  - control: `DS4_V100_TURBOMIND_GRAPH=0`
  - candidate: `DS4_V100_TURBOMIND_GRAPH=1`
  - both with `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6_reduce`
- Record prompt, generated, and continuation tok/s separately.
- Require `16/16` token match for both runs.
- Record graph capture/launch/failure evidence from the candidate server log.
- Keep graph replay default-off unless served mode is clearly positive.

## Non-Goals

- No new CUDA kernel.
- No change to model, pack, or weight layout.
- No MTP promotion.
- No tensor-parallel production path.
- No default promotion on direct replay evidence alone.

## Execution

Use the persistent production appliance pack on the V100 pod:

```text
/workspace/packs/ds4-appliance-full-tm-gated-s181
```

Use the sustained decode benchmark because it starts one resident replay server
per case and records the metrics this project uses for serving decisions.

Both runs should use the same binary and production-serving shape:

```text
ctx=262144
slots=16
active_microbatch=16
tokens_per_request=64
requests=16
async_pipeline_mode=per-step
async_event_handoff=1
```

## Definition Of Done

- [x] Sprint 199 document exists before execution.
- [x] V100 build passes for `tools/ds4-v100-replay`.
- [x] Control served run completes with `16/16` token match.
- [x] Graph served run completes with `16/16` token match.
- [x] Prompt, generated, and continuation tok/s are recorded for both runs.
- [x] Candidate server log proves graph captures and graph launches, with no
      launch failures.
- [x] Decision is recorded:
      - promote graph replay only if served continuation/decode throughput is
        clearly positive without correctness loss;
      - otherwise keep default-off and pivot back to persistent routed FFN or
        full-layer TP/EP work.
- [x] Vision/status artifacts are updated.
- [x] Changes are committed.

## Implementation

The production V100 appliance defaults now select the measured Sprint 199 stack:

```text
DS4_V100_TURBOMIND_GATED_SILU=1
DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6_reduce
DS4_V100_TURBOMIND_GRAPH=1
```

These defaults apply to `tools/ds4-v100-run-appliance.sh` and the
`deploy/v100/ds4-v100-appliance.env.example` production template. The generic
rollback path remains explicit:

```text
DS4_V100_TURBOMIND_ROUTED_EXECUTOR=off
DS4_V100_TURBOMIND_GRAPH=0
```

## Validation

V100 build:

```text
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

passed on `llm/llamacpp-build-8gpu`.

Same-binary served A/B at 16-slot/256K, 16 requests x 64 generated tokens,
per-step async + event handoff:

| Mode | Prompt tok/s | Generated tok/s | Continuation tok/s | Token match | Avg GPU util | Max GPU util |
|---|---:|---:|---:|---:|---:|---:|
| `fused6_reduce`, graph off | `15.391536` | `54.725463` | `53.870377` | `16/16` | `34.098%` | `66%` |
| `fused6_reduce`, graph on | `19.093013` | `67.886268` | `66.825545` | `16/16` | `39.148%` | `69%` |

The graph candidate improved continuation throughput by about `+24.05%` versus
the same `fused6_reduce` stack with graph replay disabled.

Because `fused6_reduce` was not previously the production default, Sprint 199
also ran the production-routed-executor control:

| Mode | Prompt tok/s | Generated tok/s | Continuation tok/s | Token match | Avg GPU util | Max GPU util |
|---|---:|---:|---:|---:|---:|---:|
| routed executor off, graph off | `15.952247` | `56.719099` | `55.832863` | `16/16` | `33.471%` | `64%` |
| `fused6_reduce`, graph on | `19.093013` | `67.886268` | `66.825545` | `16/16` | `39.148%` | `69%` |

The promoted stack improved continuation throughput by about `+19.69%` versus
the current routed-executor-off production control in this same harness.

Graph evidence from the candidate server log:

```text
turbomind_graph captured:       43
turbomind_graph launched:       129
turbomind_graph launch_failed:  0
turbomind_graph capture_failed: 0
begin_capture_failed:           0
```

Launcher default check on the V100 pod:

```text
turbomind_gated_silu=1
turbomind_routed_executor=fused6_reduce
turbomind_graph=1
```

passed with the Sprint 181 production appliance pack.

Evidence:

```text
logs/from-cluster/sprint199-graph-fused-served/
```

## Decision

Promote `fused6_reduce + graph replay` for the Sprint 181+ production V100
appliance pack.

This is the first recent execution-boundary change that clears the served
16-slot/256K production gate with a material margin against both the internal
`fused6_reduce` graph-off control and the routed-executor-off production
control. It does not realize the full high-throughput vision; aggregate
throughput is still in the `~67` tok/s band, not the target `300+` tok/s band.
The next sprint should stop wrapper-level graph work and use this promoted
baseline to pursue a larger persistent routed-FFN kernel or full-layer TP/EP
prototype.
