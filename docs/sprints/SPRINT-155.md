# Sprint 155 - Routed-FFN Software-Pipelined Executor

Date: 2026-05-21

## Objective

Implement and validate a production-relevant routed-FFN software-pipeline
candidate for the V100 appliance.

The prior sprints exhausted stage-count changes inside the existing fused
gate/up GEMM and the down route-reduce epilogue. This sprint changes the FFN
execution boundary: run each routed expert group as a gate/up plus down chain on
dedicated CUDA streams, then join before the existing weighted route reduction.

## Scope

- Add optional TurboMind DS4 fixed-shape one-group probe entrypoints for the
  768-route and 1536-route shapes.
- Add an opt-in DS4 CUDA executor path guarded by
  `DS4_V100_TURBOMIND_GROUP_PIPELINE=1`.
- Keep the default compact fused routed-FFN path unchanged unless the flag is
  set.
- Fix the total-tokens TurboMind ABI selection so generic grouped GEMMs avoid
  the compatibility path's synchronous offset read when the ABI is available.
- Benchmark prefill/generated/decode tok/s separately at 128-slot/32K and, if
  correct, 256-slot/16K.

## Non-Goals

- No broad tensor-parallel rewrite.
- No 8-way expert/tensor parallel scheduler.
- No persistent single CUDA kernel that rewrites TurboMind's mainloop from
  scratch.

## Definition of Done

- The appliance builds with the new TurboMind symbols.
- Full scheduler smoke passes with the default path.
- Full scheduler smoke passes with
  `DS4_V100_TURBOMIND_GROUP_PIPELINE=1`.
- Served A/B reports prompt, generated, and continuation/decode tok/s.
- Results are recorded in `docs/sprints/EXPERIMENT-STATUS.md`,
  `docs/sprints/STATUS.md`, `docs/sprints/VISION.md`, and
  `TEMP_CURRENT_REPORT.md`.

## Decision Gate

Promote only if the software-pipelined path is correct and improves served
decode throughput materially. If it is flat or slower, keep it as an explicit
diagnostic flag and move the main implementation line to the next larger FFN
kernel boundary or the bounded one-layer 2-way TP prototype.

## Implementation

This sprint added fixed-shape one-expert-group TurboMind entrypoints for the
DS4 768-route and 1536-route shapes, plus an opt-in CUDA executor path guarded
by `DS4_V100_TURBOMIND_GROUP_PIPELINE=1`.

The first attempt only supported the interleaved gated-SiLU pack and therefore
fell back on the current production appliance. The final implementation supports
the actual non-interleaved fused gate/up pack by chaining, per active expert
group and per CUDA stream:

1. fused gate/up MXFP4 matmul,
2. group-local SwiGLU,
3. down MXFP4 matmul,
4. join to the default stream before the existing weighted route reduction.

The path also forces compact active-expert scheduling when the group pipeline
flag is enabled. In the current deterministic served shape this compacts the
resident 256-expert table down to 8 stream groups, with 6 active experts.

## Validation

Builds:

- `cmake --build build/turbomind-v100-s127 --target ggml-turbomind -j80`
- `make tools/ds4-v100-replay tests/cuda_v100_full_scheduler_smoke CUDA_ARCH=sm_70 -j80`

Correctness:

- Default full 8-stage scheduler smoke passed at 128-slot/32K with
  `43/43` TurboMind routed layers.
- Pipeline full 8-stage scheduler smoke passed at 128-slot/32K with
  `43/43` TurboMind routed layers.
- Stage-0 profiled hardware smoke at 256-slot/16K passed and proved the branch
  was active:
  `group_pipeline_calls=6`, `group_pipeline_groups=48`.

## Served A/B

The first served A/B was discarded after profiling showed it had fallen back.
The table below is the corrected active-pipeline run.

| Shape | Mode | Generated tok/s | Prompt tok/s | Continuation tok/s | Correctness |
|---|---|---:|---:|---:|---|
| 128-slot / 32K | control | `59.394915` | `66.819280` | `55.682733` | 128/128 |
| 128-slot / 32K | group pipeline active | `59.125703` | `66.516416` | `55.430346` | 128/128 |
| 256-slot / 16K | control | `60.648138` | `68.229156` | `56.857630` | 256/256 |
| 256-slot / 16K | group pipeline active | `60.308689` | `67.847275` | `56.539396` | 256/256 |

## Decision

Do not promote `DS4_V100_TURBOMIND_GROUP_PIPELINE=1`.

The stream-per-expert software pipeline is correct and executes on V100, but it
regresses served throughput by roughly `0.45-0.56%`. The likely issue is launch
and stream overhead plus no-op compact groups overwhelming the small single-step
overlap benefit. The next material path should either:

- fuse the whole routed-FFN boundary into a persistent DS4-only kernel, or
- continue the bounded one-layer 2-way TP prototype for the 128-slot/32K NV2
  case.

## Artifacts

- `logs/from-cluster/sprint155-group-pipeline-active-ab/`
