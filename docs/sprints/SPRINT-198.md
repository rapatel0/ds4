# Sprint 198 - Graph Replay For Fused Routed Executor

Date: 2026-05-23
Status: Completed

## Objective

Re-test CUDA graph replay against the current production-shaped routed-FFN
executor path, `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6_reduce`, instead of
the older generic TurboMind wrapper path.

## Context

Sprint 169 fixed explicit-stream graph capture but found served throughput
regressed. Since then, the routed executor path changed materially:

- compact activation staging can avoid route-expanded activation rows;
- `fused6_reduce` can elide `down_routes`;
- Sprint 197 proves `mid_half` is the only remaining materialized routed
  buffer in the six-route path.

The existing graph gate still rejects all routed-executor modes, so it cannot
measure whether graph replay helps the current execution boundary.

## Scope

- Allow CUDA graph capture only for routed executor modes that do not require a
  host active-expert readback during capture:
  - `off`
  - `fused6`
  - `fused6_reduce`
- Include the routed-executor mode in the graph key/mode flags.
- During graph capture, bypass compact active-expert host readback.
- Keep graph replay default-off.
- Validate on V100 with `fused6_reduce` and graph verbose logs.

## Non-Goals

- No default promotion.
- No new MXFP4 math kernel.
- No whole-stage graph capture.
- No served-throughput claim unless a same-binary A/B is completed.

## Definition Of Done

- [x] `ds4_cuda.cu` builds on V100.
- [x] `tools/ds4-v100-replay` builds on V100.
- [x] A `fused6_reduce + graph` replay smoke emits graph warmup/capture/replay
      evidence instead of being blocked by routed-executor gating.
- [x] Correctness is preserved for a short replay smoke.
- [x] If graph replay works, record a same-binary short A/B against graph off.
- [x] Sprint/vision/status artifacts are updated.
- [x] Changes are committed.

## Implementation

`ds4_cuda.cu` now lets CUDA graph capture run when
`DS4_V100_TURBOMIND_ROUTED_EXECUTOR` is `off`, `fused6`, or `fused6_reduce`.
Other routed-executor modes remain blocked.

The graph key now includes routed-executor mode bits, preventing cache reuse
between generic, `fused6`, and `fused6_reduce` executions. During graph capture,
the compact active-expert host readback is skipped, matching the no-host-sync
condition needed for capture safety. Normal default behavior remains unchanged
because `DS4_V100_TURBOMIND_GRAPH=0` is still the default.

Verbose graph logs now include the first few graph launches per entry.

## Validation

V100 build:

```text
$ make -j80 CUDA_ARCH=sm_70 ds4_cuda.o tools/ds4-v100-replay
/usr/local/cuda/bin/nvcc -O3 --use_fast_math -arch=sm_70 ...
```

Graph-enabled fused executor smoke:

```text
DS4_V100_TURBOMIND_GRAPH=1
DS4_V100_TURBOMIND_GRAPH_VERBOSE=1
DS4_V100_TURBOMIND_GATED_SILU=1
DS4_V100_TURBOMIND_FUSED_GATE_UP=1
DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1
DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6_reduce
./tools/ds4-v100-replay \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --appliance-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --synthetic-prompt-token 926 \
  --synthetic-prompt-len 8 \
  --ctx 262144 \
  --tokens 3 \
  --json
```

Result:

```text
rc=0
turbomind_graph captured ... tokens=1 routes=6
turbomind_graph launched ... tokens=1 routes=6
```

Same-binary direct replay A/B:

| Mode | Prompt tok/s | Generated tok/s | Continuation tok/s | Output IDs | Graph evidence |
|---|---:|---:|---:|---|---|
| graph off | 4.475985 | 3.568240 | 16.022442 | `201,200,84921,200,18,90,926,14` | n/a |
| graph on | 4.679220 | 3.780181 | 17.980888 | `201,200,84921,200,18,90,926,14` | 43 captured, 129 launched, 0 failed |

## Decision

This resolves the old graph compatibility gap for the current
`fused6_reduce` routed executor. Direct replay shows a positive continuation
signal (`+12.2%`) with matching token IDs.

Do not promote graph replay to default yet. Sprint 169 showed graph replay can
improve direct replay while regressing served throughput. The next required
gate is a same-binary served 16-slot/256K A/B using the persistent appliance
pack. If served mode stays positive, graph replay becomes the first practical
execution-boundary optimization candidate for the current fused executor. If it
regresses again, move to a true persistent/tile-level routed-FFN executor.
