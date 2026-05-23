# Sprint 208 Intent - Separate TP8 Investigation Path

Date: 2026-05-23

## Seed Prompt

Create a sprint plan for investigating full TP8 as a separate DS4 V100 runtime
path. The TP path should use new files by default because its scheduler, KV
ownership, pack layout, collectives, and runtime failure modes differ
substantially from the existing PP/layer-sharded appliance.

## Orientation Summary

- Current production-ish serving path is PP/layer-sharded across 8 V100s, with
  contiguous layer ownership and hidden-context relay between stages.
- Recent sprints 201-205 investigated TP4 and showed that routed expert compute
  can scale, but small-payload collectives and routed-only overlays do not
  justify integration into the PP scheduler.
- Sprint 206 showed the exact six-route FFN boundary is not meaningfully
  launch-bound; the next major lever is either a bigger fused kernel or a
  topology change.
- The user now wants 32 active slots while preserving at least 128K and ideally
  256K context. With quantized KV and sharding, planner envelopes make PP1/TP8
  plausible.
- Existing `docs/architecture/DS4-V100-LAYOUT.md` still contains older TP2/TP4
  thinking. The new note `docs/architecture/DS4-V100-TP8-INVESTIGATION.md`
  captures the updated TP8 insight and separation rule.

## Relevant Codebase Areas

- `tools/ds4-v100-plan.c`: existing PP/layer planner; should not absorb TP8
  assumptions.
- `tools/ds4-v100-plan-tp.c`: new TP envelope planner created as the first
  separate TP file.
- `tools/ds4-v100-tp4-collective-smoke.cu`: existing TP4 collective smoke;
  useful reference for a new TP8 variant.
- `tools/ds4-v100-tp4-layer-proxy.cu`: existing layer-boundary proxy; useful
  reference but should not become the TP8 production abstraction.
- `kernels/turbomind/ggml-turbomind/test_tp_split_4gpu.cpp`: existing TP4
  routed-FFN compute envelope; useful for shape and correctness patterns.
- `kernels/turbomind/ggml-turbomind/test_tp4_resident_layer_slice.cu`: existing
  resident TP4 slice; useful for reduction algorithms and pitfalls.
- `ds4_v100_scheduler.*`, `ds4_v100_replay.*`, `ds4_cuda.cu`: production PP
  path; should be treated as baseline/control, not generalized for TP8 in this
  sprint.

## Constraints

- TP development must use new files and new runtime/planner entry points by
  default.
- Reuse is allowed only below the ownership boundary: source readers, pack
  helpers, stable TurboMind kernels, CUDA utility wrappers, logging, and
  measurement patterns.
- Do not retrofit the PP scheduler into a TP scheduler.
- Do not make TP8 the default serving path in this sprint.
- Do not use replicated KV to claim 32-slot/256K viability.
- Preserve Sprint 207 dirty kernel work; this plan should not overwrite it.

## Success Criteria

- A TP-specific planner exists and builds from a separate file.
- The planner reports 32-slot/128K and 32-slot/256K envelopes for PP8/TP1,
  PP4/TP2, PP2/TP4, and PP1/TP8.
- The planner shows why KV sharding is required by printing replicated-KV
  sensitivity.
- A TP8-specific boundary/collective experiment plan exists with 32-slot,
  64-slot, and 128-slot payloads.
- The final sprint plan requires concrete V100 runs and NVLink traffic evidence
  before any TP scheduler integration.
- Documentation records that TP8 is a separate investigation path, not an
  extension of the existing PP scheduler.

## Verification Strategy

- Local build: `make tools/ds4-v100-plan-tp`.
- Local planner runs:
  - `tools/ds4-v100-plan-tp --slots 32 --ctx 131072 --kv-dtype f8`
  - `tools/ds4-v100-plan-tp --slots 32 --ctx 262144 --kv-dtype f8`
  - replicated-KV sensitivity with `--kv-sharding off`.
- V100 build for new TP8 probes using `-j80` and `CUDA_ARCH=sm_70`.
- V100 runs for TP8 collective and resident-boundary probes across all 8 GPUs.
- Record timing and NVLink counter evidence where supported by `nvidia-smi` or
  Nsight/CUPTI.

## Uncertainty Assessment

- Correctness: Medium. The planner is deterministic, but TP8 KV ownership and
  reductions are new.
- Scope: Medium. The sprint should stop at investigation/probes, not a full TP8
  runtime.
- Architecture: High. The work intentionally avoids runtime/scheduler
  implementation in this sprint and must not fight existing PP abstractions.

## Open Questions

- Does 8-GPU collective latency remain acceptable at the 32-slot/256K target
  when resident compute is present?
- Does TP8 beat PP2/TP4 or PP4/TP2 once synchronization, not wire bytes, is
  measured?
- Should the first TP8 scheduler prototype include attention/KV, or first mock
  KV while proving full-layer hidden-state residency?

## Actionable Follow-Ups

- Sprint 175 follow-up recommended a broader TP/EP topology sprint if larger
  in-GPU FFN fusion was too invasive.
- Sprint 205 decision said TP4 decode branch was paused for small-payload
  collectives, but larger-batch/prefill remained plausible.
- Sprint 206 decision says the six-route boundary is not launch-bound, pushing
  the next material lever toward a larger fused kernel or topology change.

## Vision Context

`docs/sprints/VISION.md` exists. The current vision says practical serving is
now the north star, with graph-backed `fused6_reduce`/six-route local work
largely exhausted and TP viable only if dense and routed computation stay
inside the TP boundary. This sprint updates that direction: the next topology
investigation should target `PP1/TP8` first for the 32-slot serving goal, while
keeping PP2/TP4 as a fallback/control.
