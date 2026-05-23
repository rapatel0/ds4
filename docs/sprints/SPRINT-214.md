# Sprint 214 - Tile-Local Routed FFN Workbench

Date: 2026-05-23
Status: Planned

## Overview

Build a standalone V100 workbench for a true tile-local/persistent six-route
routed-FFN executor.

The current production serving baseline is still `fused6_reduce + graph`.
Sprint 213 proved that materializing the six down-route rows and reducing them
with a separate half-to-F32 kernel is correct but not material in served mode.
The next useful implementation must change the dataflow boundary: gate/up,
activation, down, and route-weighted reduction need to be fused or persistently
scheduled so the intermediate routed activation is not just another global
memory handoff between wrapper kernels.

Sprint 214 is a workbench sprint, not a production integration sprint. It should
produce a concrete kernel candidate or a hard rejection backed by V100 timing.

## Rationale

Closed evidence:

| Sprint | Finding | Implication |
|---|---|---|
| 199 | graph-backed `fused6_reduce` is the best served baseline | keep it as production default |
| 200 | exact six-route wrapper cut-ins are slower or irrelevant | wrapper dispatch is not the lever |
| 206 | graphing the focused six-route FFN sequence is only `1.016x-1.050x` | launch overhead alone is not enough |
| 213 | materialized split-reduce is correct and focused-positive but served `+0.7%` only | reducer-only work is not enough |

The remaining hypothesis is tile locality. A useful candidate should avoid
writing the full intermediate route activation to global memory only to read it
back for the down projection. On V100 this may still compute in FP16/FP32, but
low-bit packed weight loads and dequantization must happen inside the GPU
kernel path, with no CPU-side repacking or serving-format churn.

## Scope

1. Add a new standalone workbench, for example
   `tools/ds4-v100-routed-ffn-tile-workbench.cu`.
2. Reuse the existing TurboMind/CUTLASS-style SM70 packed MXFP4 helpers where
   possible instead of inventing a new storage format.
3. Implement at least one tile-local candidate for the exact production shape:
   - total routes: `6`;
   - hidden: `4096`;
   - middle: `2048`;
   - active experts: `6`;
   - max routes per expert: `1`;
   - output: one F32 hidden row after route weighting.
4. Compare against the proven TurboMind sequence in the same executable:
   - `gated_silu -> down_reduce`;
   - `gated_silu -> down -> split_reduce`;
   - candidate tile-local/persistent path.
5. Validate correctness against the existing sequence, with finite synthetic
   MXFP4 fixtures and tolerances at least as strict as Sprint 213.
6. Measure V100 timing over enough iterations to reduce noise.
7. If practical, capture Nsight/nvprof counters for tensor-core utilization,
   global memory traffic, and kernel duration.
8. Record whether the candidate is strong enough to justify appliance
   integration in Sprint 215.

## Parallel Work Lanes

These lanes can run in parallel with subagents or separate shells:

| Lane | Work | Write scope |
|---|---|---|
| A | Kernel/workbench implementation | `tools/ds4-v100-routed-ffn-tile-workbench.cu`, TurboMind probe files if needed |
| B | Baseline and profiler harness | logs and benchmark scripts only |
| C | Correctness oracle/tolerance review | workbench compare code only |
| D | Documentation/status synthesis | sprint docs after evidence exists |

Workers are not alone in the codebase. Do not revert or overwrite unrelated
changes, and do not edit TP/PP scheduler files for this sprint.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-routed-ffn-tile-workbench.cu` | standalone tile-local candidate and baseline comparison |
| `Makefile` | CUDA target and macOS guard for the workbench |
| `kernels/turbomind/ggml-turbomind/ggml-turbomind-ds4-probe.cu` | optional fixed-shape helper kernels |
| `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h` | optional ABI only if needed by the workbench |
| `logs/from-cluster/sprint214-routed-ffn-tile-workbench/` | V100 evidence |
| `docs/sprints/SPRINT-214.md` | plan and outcome |

## Non-Goals

- No production default change.
- No served A/B unless the standalone candidate beats the focused promotion
  gate.
- No TP4/TP8 runtime work.
- No PP scheduler or `ds4_v100_scheduler.*` changes.
- No GGUF or pack-format changes.
- No model-weight logs.

## Definition Of Done

- [ ] Sprint plan exists and is committed before execution evidence is staged.
- [ ] New standalone tile-local routed-FFN workbench exists.
- [ ] Makefile target exists with CUDA build and macOS CUDA-required guard.
- [ ] V100 build passes with `CUDA_ARCH=sm_70`.
- [ ] Baseline TurboMind sequence correctness passes.
- [ ] Candidate correctness passes or the failure is root-caused.
- [ ] Timing reports atomic sequence, split-reduce sequence, candidate path,
      and speedups.
- [ ] Profiler counters are captured or explicitly skipped with reason.
- [ ] Decision is recorded:
      - continue to appliance integration only if candidate beats the focused
        sequence by at least 10%;
      - otherwise reject this candidate and pivot away from routed-FFN
        microkernel work.
- [ ] Logs are copied to
      `logs/from-cluster/sprint214-routed-ffn-tile-workbench/`.
- [ ] `docs/sprints/STATUS.md` and `docs/sprints/VISION.md` are updated.
- [ ] Changes are committed with explicit `git add` paths.

## Decision Gate

Continue to production integration only if:

- correctness passes;
- focused candidate time is at least 10% faster than the best Sprint 213 focused
  sequence (`0.1290 ms`);
- tensor-core utilization or memory-traffic evidence explains the win;
- no new global intermediate buffer larger than the current sequence is added.

Reject and pivot if:

- the candidate is correct but below the 10% focused gate;
- correctness requires looser tolerances than the existing TurboMind sequence;
- the implementation becomes a fragile wrapper around the same global-memory
  boundary;
- profiling shows the path is launch/memory bound with no credible path to
  served improvement.

## Risks

- A true fused MXFP4 gate/up+down kernel may require deeper TurboMind template
  work than fits in one sprint.
- The six-route shape may be too small to benefit from tensor-core fusion even
  with tile locality.
- A synthetic workbench can overstate served impact; served A/B remains a later
  gate.

## Security

Synthetic fixtures only. No service exposure and no model weights in logs.

## Dependencies

- Sprint 213 split-reduce diagnostic result.
- Existing TurboMind V100 build and CUDA toolchain.
- V100 pod `llm/llamacpp-build-8gpu`.
