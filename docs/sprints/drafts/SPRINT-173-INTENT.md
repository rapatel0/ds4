# Sprint 173 Intent - Reusable Fused Routed-FFN Boundary

## Seed Prompt

Plan the next sprint after agreeing that Option 1 should be implemented in a
way that helps Option 2. The sprint should move toward high-throughput practical
DeepSeek V4 Flash serving on 8x V100 by building a reusable fused routed-FFN
executor boundary. The boundary should keep compact FP4/FP8/int storage traffic
small, expand to FP16/FP32 only inside GPU execution near tensor-core work, and
avoid global-memory reshaping/staging where possible. It must be designed so
the local one-GPU version can later become the TP/EP primitive.

## Orientation Summary

- Current repo state: branch `claude-takeover` is clean and ahead of
  `origin/claude-takeover` by Sprints 170-172. The latest committed result,
  Sprint 172, kept `DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=0` as default after a
  noisy, non-promotable 16-slot/256K A/B. The V100 pod has no active GPU compute
  jobs.
- Recent work theme: Sprints 154-172 have ruled out wrapper-level CUDA Graph
  replay, fixed dispatch bypass, down/scatter epilogues, small route-building,
  stage-count tweaks, host stream-per-expert scheduling, and slot/layer
  coalescing as material serving levers. The repeated conclusion is that the
  next sprint must change a larger routed-FFN execution boundary.
- Evidence to preserve: the production 16-slot/256K path presents
  `total_routes=6`, `active_experts=6`, `max_routes_per_expert=1`. The routed
  profile from Sprint 126 showed gate/up about `47%`, down about `23%`, route
  build about `17%`, gather about `4%`, and scatter about `5%` of profiled
  routed-FFN time. Sprints 170-172 confirmed dispatch, down-reduce, and route
  construction alone are not the missing lever.
- TP context: Sprints 149-165 validated TP math and descriptors but rejected
  the old layer-local overlay. The proxy was positive at practical route shapes
  (`1.260x` total-with-copy at 6 routes and about `1.328x` at 96 routes on NV2),
  while the scheduler overlay regressed because copy/sync overhead was bolted
  onto one layer. Sprint 173 should avoid breaking the TP path; any local fused
  executor should expose partial-output semantics or a clean adapter for later
  TP/EP.
- Constraints: preserve source quantized model quality and device-resident
  appliance behavior; V100 compute should use FP16 tensor cores and FP32
  accumulation where needed, because V100 has no native BF16/FP8/FP4 tensor
  cores. Keep memory traffic compact and move conversion/dequantization into
  registers/shared memory rather than writing expanded global intermediates.

## Relevant Codebase Areas

- `ds4_cuda.cu`
  - current routed TurboMind path:
    `cuda_tm_routed_mxfp4_packed_impl`
  - route build/gather/cast/reduce helpers:
    `tm_gather_f32_to_f16_kernel`, `tm_swiglu_*`,
    `tm_reduce_sum_*`, `tm_scatter_sum_*`
  - current TurboMind ABI selection and profiling hooks
- `kernels/turbomind/ggml-turbomind/`
  - copied TurboMind MXFP4 kernels and DS4 fixed-shape probe wrappers
  - `ggml-turbomind-ds4-probe.cu`
  - `api.cc`
  - `include/ggml-turbomind-api.h`
- `tools/`
  - `tools/ds4-v100-replay`
  - `tools/ds4-v100-appliance-soak.sh`
  - any TurboMind standalone benchmark harnesses used by Sprints 132-152
- `ds4_v100_layer_execute.*`, `ds4_v100_layer_state.*`,
  `ds4_v100_scheduler.*`
  - must remain compatible with current layer-sharded runtime and future TP/EP
    partial-output contracts

## Desired Sprint Shape

Sprint 173 should produce a concrete implementation artifact, not just another
analysis doc. However, because a full persistent routed-FFN executor is high
risk, the sprint should be scoped to the first reusable boundary primitive:

1. Add an explicit routed-FFN execution contract/descriptor for a fused executor
   that can run in local full-output mode now and later support partial-output
   TP/EP mode.
2. Implement a bounded prototype for the production 6-route shape that removes
   at least one global expanded intermediate from the current path. Preferred
   target: remove route-expanded `a_half` global staging by forming activation
   tiles inside the executor/kernel boundary, while keeping existing TurboMind
   gate/up and down calls available as fallback if a full monolithic kernel is
   too large for one sprint.
3. Add instrumentation that explicitly reports which global intermediates were
   eliminated or still materialized, so future work can continue the liveness
   audit.
4. Validate with real V100 build, symbol/selector evidence, direct replay smoke,
   and 16-slot/256K served A/B with prompt/generated/continuation tok/s split.

If the implementation can safely go further, fold gate/up + gated-SiLU + down +
route reduce into one boundary. If that is not feasible in one sprint, the
minimum acceptable implementation is a fused-boundary scaffold plus a working
in-kernel activation-tile path that is correctness validated and measured.

## Success Criteria

- A new opt-in runtime flag or internal executor mode exists for the fused
  routed-FFN boundary, default off until served A/B clears the gate.
- The implementation is reusable for future TP/EP:
  - explicit input layout
  - explicit route metadata layout
  - explicit output mode: full output now, partial output later
  - no hard dependency on one GPU owning all future experts
- At least one current global expanded routed-FFN intermediate is removed or
  bypassed in the candidate path, with log/profiler evidence.
- V100 build passes for TurboMind and replay.
- Correctness passes on direct replay or focused routed-FFN smoke before served
  benchmarking.
- Served 16-slot/256K A/B records generated, prompt, and continuation/decode
  tok/s separately.
- Decision gate:
  - promote only if continuation/decode tok/s improves by at least `10%` with
    correctness intact;
  - if the fused boundary is correct but below `10%`, keep opt-in and use the
    implementation as the TP/EP primitive seed;
  - if the primitive cannot remove a real global intermediate, stop and pivot to
    persistent TP/EP planning.

## Verification Strategy

- Local static checks:
  - `git diff --check`
  - compile affected host/CUDA files where possible
- V100 pod:
  - build `libggml-turbomind.so` if TurboMind ABI changes
  - build `tools/ds4-v100-replay` with `CUDA_ARCH=sm_70`
  - run focused smoke for the new executor path
  - run direct replay on real model
  - run 16-slot/256K served A/B using `tools/ds4-v100-appliance-soak.sh`
- Evidence:
  - selected path logs
  - benchmark JSON summaries
  - prompt/generated/continuation split
  - token match
  - GPU util samples when practical

## Uncertainty Assessment

- Correctness: High. The routed FFN is numerically sensitive and combines
  routing, MXFP4 expert weights, gate/up/down, activation, route weighting, and
  residual flow.
- Scope: High. A fully persistent monolithic executor may exceed one sprint; the
  sprint must define a minimum reusable primitive that still advances the final
  architecture.
- Architecture: High. The local executor should be useful for future TP/EP and
  must not repeat the old TP overlay mistake.

## Open Questions For Planning

1. Should the sprint target only the routed-FFN path, or include a small
   dtype/liveness audit table for shared FFN and attention to confirm routed FFN
   remains the best first fused island?
2. Is removing `a_half` global staging the right first implementation slice, or
   should the sprint instead start with a full fused gate/up + activation + down
   custom kernel scaffold?
3. What served A/B threshold should terminate Option 1 and force the TP/EP
   pivot? The current proposed gate is `10%` continuation/decode improvement.
4. Should partial-output mode be implemented now as an API shape only, or tested
   in a synthetic TP smoke during this sprint?

## Prior Follow-Ups Now Actionable

- `docs/sprints/SPRINT-170-FOLLOWUPS.md`: Critical follow-up to build a
  persistent/fused six-route routed-FFN executor. Prerequisites are met because
  Sprints 171-172 have closed exact served-shape down-reduce and small-route
  construction variants.
- `docs/sprints/SPRINT-170-FOLLOWUPS.md`: Nice-to-have runbook docs for
  `DS4_V100_TURBOMIND_ROUTED_EXECUTOR`; not central to this sprint and should
  be deferred unless touched naturally.

## Vision Context

`docs/sprints/VISION.md` exists. The North Star is a high-throughput
device-resident DS4 V100 appliance preserving source quantized model quality.
The current revision says the next work should be a persistent routed-FFN
executor or persistent TP/EP boundary. The user agrees Option 1 should be
implemented in a way that carries into Option 2.
