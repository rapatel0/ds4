# Sprint 156 - Fused Pipeline Stream-Group Validation

Date: 2026-05-21

## Objective

Fully validate the current fused routed-FFN software-pipeline candidate before
moving the main implementation line away from it.

Sprint 155 proved that the opt-in stream-per-expert path executes on V100, but
the served A/B used the default eight stream groups while the profiled
128-slot/32K stage only had six active experts. This sprint tests whether that
extra group/launch overhead hid the useful part of the software pipeline.

## Scope

- Keep the production default unchanged.
- Test `DS4_V100_TURBOMIND_GROUP_PIPELINE=1` with tighter stream-group counts,
  starting at six groups for the observed six-active-expert compact shape.
- Validate correctness with full scheduler smoke before served A/B.
- Benchmark prompt, generated, and continuation/decode tok/s separately.
- If the exact-group run is still flat or slower, treat the host-orchestrated
  fused pipeline as exhausted and move to a persistent fused FFN boundary or a
  bounded tensor-parallel prototype.

## Definition Of Done

- 128-slot/32K full scheduler smoke passes with exact stream-group pipeline.
- 128-slot/32K served A/B compares control, default eight-group pipeline, and
  exact six-group pipeline.
- If the exact six-group run is correct and promising, repeat at 256-slot/16K
  only if active-group count is validated first.
- Results are recorded in `TEMP_CURRENT_REPORT.md`,
  `docs/sprints/STATUS.md`, and `docs/sprints/EXPERIMENT-STATUS.md`.
- Artifacts are copied back under `logs/from-cluster/`.

## Decision Gate

Promote no software-pipeline flag unless it improves continuation/decode tok/s
outside normal run noise. A flat or slower exact-group run means this style of
software pipelining is not the material lever; the next sprint should target a
persistent fused routed-FFN executor or the narrow 2-way tensor-parallel layer
prototype.

## Implementation

This sprint kept the existing group-pipeline executor but validated it with the
right active stream count. The first 128-slot smoke run was discarded because
it did not set `DS4_V100_TURBOMIND_LIB` and therefore used the wrong default
TurboMind path.

After rerunning with
`DS4_V100_TURBOMIND_LIB=build/turbomind-v100-s127/libggml-turbomind.so`, the
manual exact-group path passed full scheduler smoke at both 128-slot/32K and
256-slot/16K. The profile showed exactly six active expert groups:

- 128-slot/32K: `group_pipeline_calls=6`, `group_pipeline_groups=36` on full
  six-layer stages, `avg_active_experts=6.000`.
- 256-slot/16K: `group_pipeline_calls=6`, `group_pipeline_groups=36` on full
  six-layer stages, `avg_active_experts=6.000`.

The sprint also added a safe diagnostic mode:

```text
DS4_V100_TURBOMIND_GROUP_PIPELINE_AUTO_GROUPS=1
```

That mode reads the compacted route offsets, sets the pipeline group count to
the observed active expert count, and falls back to the normal grouped path if
the active group count exceeds the configured stream limit. It passed the
128-slot/32K full scheduler smoke, but the host readback/sync made served
throughput slower.

## Served A/B

| Shape | Mode | Generated tok/s | Prompt tok/s | Continuation tok/s | Correctness |
|---|---|---:|---:|---:|---|
| 128-slot / 32K | control | `59.516392` | `66.955941` | `55.796618` | 128/128 |
| 128-slot / 32K | stream pipeline, 6 groups | `59.645848` | `67.101578` | `55.917982` | 128/128 |
| 128-slot / 32K | auto active groups | `58.988662` | `66.362245` | `55.301871` | 128/128 |
| 256-slot / 16K | control | `60.442968` | `67.998339` | `56.665283` | 256/256 |
| 256-slot / 16K | stream pipeline, 6 groups | `60.675527` | `68.259968` | `56.883307` | 256/256 |
| 256-slot / 16K | auto active groups | `60.232265` | `67.761298` | `56.467748` | 256/256 |

## Decision

Do not promote the current group-pipeline path.

The exact six-group diagnostic is slightly positive in this deterministic
benchmark, but hardcoding six groups is not safe for arbitrary real traffic.
The safe auto-group implementation is correct but slower because it adds a
host-side active-group readback. This confirms that host-orchestrated
software pipelining is not the material lever.

Keep `DS4_V100_TURBOMIND_GROUP_PIPELINE=1` and
`DS4_V100_TURBOMIND_GROUP_PIPELINE_AUTO_GROUPS=1` as explicit diagnostics.
The next material implementation should remove host stream orchestration
entirely with a persistent/larger fused routed-FFN executor, or continue the
bounded 2-way TP prototype as a separate experiment.

## Artifacts

- `logs/from-cluster/sprint156-fused-pipeline-stream-groups/`
- `TEMP_STATUS_REPORT.md`

## Validation

- `make tools/ds4-v100-replay tests/cuda_v100_full_scheduler_smoke CUDA_ARCH=sm_70 -j80`
- `tests/cuda_v100_full_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 --slots 128 --ctx 32768 --expect-tm-layers 43`
- `tests/cuda_v100_full_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 --slots 256 --ctx 16384 --expect-tm-layers 43`
- `tools/ds4-v100-appliance-soak.sh` at 128-slot/32K and 256-slot/16K for
  control, six-group pipeline, and auto-group pipeline.
