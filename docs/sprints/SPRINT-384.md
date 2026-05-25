# Sprint 384: Real-Router TP/EP Serving Baseline

## Overview

Measure the TP/EP serving path with model-router routes and compact MoE decode
enabled at the target `32` slot / `256K` shape.

Sprint 383 measured launcher defaults and confirmed the launch/sync bottleneck,
but that run did not enable `--model-router-routes --compact-moe-decode`.
For intelligence-preserving serving, the real-router path must be the baseline
used for future optimization.

## Rationale

The appliance objective is not a synthetic-route throughput demo. It needs to
serve DS4 with the model's router semantics. Prior direct runs showed
model-router compact MoE is correct but slower than the synthetic/default
serving path. Before optimizing steady-state decode, we need the full active
request matrix for the real-router path with the same GPU-utilization and
VRAM telemetry as Sprint 383.

## Scope

- Teach `tools/ds4-v100-tp-ep-active-slot-matrix.py` to name artifact
  directories consistently when `--model-router-routes` and
  `--compact-moe-decode` are forwarded to the profile harness.
- Run the active-slot matrix on gpu-01 at:
  `32` configured slots, `256K` context, `position=262080`, `32`
  generated tokens, active request cases `1,4,8,16,32`.
- Enable `--model-router-routes`, `--compact-moe-decode`, GPU sampling, VRAM
  reporting, and a startup cooldown.
- Record the real-router throughput/utilization/memory table and decide the
  next implementation gate.

## Out Of Scope

- No PP/layer-split work.
- No kernel changes.
- No MTP changes.
- No promotion of E5M2 KV.

## Definition Of Done

- Local syntax checks pass for the updated matrix script.
- V100 matrix completes or records explicit startup/admission failures.
- Matrix artifacts include model-router/compact-MoE suffixes and VRAM fields.
- Sprint outcome compares real-router throughput to the Sprint 383 default
  baseline and names the next implementation target.

## Risks

- Real-router compact MoE may be substantially slower than synthetic/default
  serving; that is useful evidence, not a reason to avoid the measurement.
- Repeated resident startup still needs cooldown because Sprint 383 showed
  immediate restarts can fail at CUDA context creation.

## Outcome

Implemented and validated.

Code changes:

- `tools/ds4-v100-tp-ep-active-slot-matrix.py` now includes
  `-model-router`, `-compact-moe`, and `-no-compact-route` in expected profile
  artifact names when those flags are forwarded through `--extra-profile-arg`.

V100 artifacts:

```text
/workspace/logs/sprint384-real-router-matrix/
```

Command shape:

```text
32 configured slots
256K context
position=262080
32 generated chat tokens/request
--model-router-routes
--compact-moe-decode
--vram-report --vram-min-free-mib 64
--gpu-sample-interval-ms 500
--case-cooldown-seconds 60
```

Matrix:

| Active requests | HTTP 200 | Client tok/s | Server decode tok/s | Avg GPU util | Max GPU util | Max memory | Min free VRAM | VRAM failures |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 1.254734 | 80.934514 | 8.657051% | 42.0% | 32418 MiB | 1754 MiB | 0 |
| 4 | 4/4 | 4.987325 | 81.231383 | 8.475000% | 41.0% | 32418 MiB | 1754 MiB | 0 |
| 8 | 8/8 | 9.654914 | 79.547736 | 8.400000% | 42.0% | 32418 MiB | 1754 MiB | 0 |
| 16 | 16/16 | 18.535369 | 76.816196 | 7.916667% | 41.0% | 32418 MiB | 1754 MiB | 0 |
| 32 | 32/32 | 38.554075 | 81.505160 | 8.547222% | 40.0% | 32418 MiB | 1754 MiB | 0 |

## Decision

This is the relevant quality-preserving TP/EP serving baseline.

Compared with Sprint 383's default/synthetic-route baseline, real-router
compact MoE reduces server decode from roughly `92-98` tok/s to roughly
`77-82` tok/s and reduces the `32`-request client result from `43.853691` to
`38.554075` tok/s. The dominant added cost is visible in the HC-current
FFN/router stage, around `85-88 ms` per all-layer decode step.

The next implementation sprint should target the real-router path, not the
synthetic/default path. The first concrete target is the GPU0-heavy router /
HC-current staging path and its host/launch fragmentation; any optimization
should A/B against this Sprint 384 matrix.
