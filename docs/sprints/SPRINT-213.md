# Sprint 213 - Routed FFN Materialized-Reduce Gate And Persistent Kernel Workbench

Date: 2026-05-23
Status: Planned

## Overview

Close the existing `fused6_split_reduce` routed-FFN artifacts and turn the
result into the next concrete persistent low-bit routed-FFN direction.

Sprint 212 rejected TP4/PP1 runtime ownership as the next serving branch. The
remaining practical-serving lever is the production six-route routed FFN hot
path. The current worktree already contains uncommitted code for a
materialized six-route down reducer (`ggml_turbomind_ds4_reduce6_half_to_float`)
and an appliance mode named `fused6_split_reduce`. Sprint 213 should validate
that path on V100, compare it against the promoted `fused6_reduce + graph`
baseline, and decide whether this materialized split-reduce idea is worth
keeping as a production candidate.

This sprint is intentionally not a generic scheduler sprint and not a TP
runtime sprint. It stays in the routed-FFN kernel/runtime path.

## Rationale

Known evidence:

| Sprint | Result | Decision |
|---|---|---|
| 199 | `fused6_reduce + graph` promoted to `67.886268` generated tok/s at 16-slot/256K | Current practical serving baseline |
| 200 | exact six-route wrapper cut-ins and clear-only fusion rejected | Do not add another thin wrapper without a serving gate |
| 206 | graphing the focused six-route FFN sequence gave only `1.016x-1.050x` focused speedup | dispatch alone is not enough |
| 211 | TP8 MXFP4 shard-256 failed correctness | do not integrate TP8 runtime |
| 212 | TP4 low-bit layer body is correct but total speed is not a serving win | do not build TP4/PP1 runtime next |

The uncommitted `fused6_split_reduce` path is a materialized alternative to the
atomic down-reduce epilogue. It must be measured, not assumed. If it is slower,
it should be recorded as rejected and not allowed to distract the next sprint.
If it is faster in focused and served gates, it can become the immediate
production candidate.

## Scope

1. Audit the current dirty `fused6_split_reduce` implementation without
   disturbing unrelated work.
2. Build TurboMind and the appliance runtime on the V100 pod with `sm_70`.
3. Run focused TurboMind validation for the six-route materialized reducer:
   - symbol export;
   - standalone split-reduce correctness;
   - full FFN materialized split-reduce timing;
   - comparison against atomic down-reduce epilogue.
4. Run full scheduler smoke with `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6_split_reduce`.
5. Run same-binary served A/B at 16-slot/256K:
   - control: promoted `fused6_reduce + graph`;
   - candidate: `fused6_split_reduce + graph` if graph capture is supported;
   - report generated and continuation/decode tok/s separately.
6. Record the decision:
   - promote only if correctness passes and served continuation clears a real
     gain over the Sprint 199 baseline/control;
   - otherwise keep default unchanged and mark split-reduce as rejected or
     diagnostic-only.
7. If split-reduce is rejected, write the next persistent-kernel workbench plan
   directly into the outcome: a real fused gate/up+down tile-local executor,
   not another reducer wrapper.

## Parallel Work Lanes

These are safe to run in parallel when using subagents or separate shells:

| Lane | Work | Write scope |
|---|---|---|
| A | V100 build + focused TurboMind split-reduce bench | logs only |
| B | Runtime launcher/smoke wiring audit | `ds4_cuda.cu`, `tools/ds4-v100-run-appliance.sh` only if fixes are required |
| C | Served A/B harness and log capture | logs only |
| D | Documentation/status synthesis | sprint docs after evidence exists |

No lane may edit TP scheduler files or introduce a generic routed-FFN scheduler
abstraction.

## Files In Scope

| File | Purpose |
|---|---|
| `ds4_cuda.cu` | opt-in routed executor selection and TurboMind ABI dispatch |
| `kernels/turbomind/ggml-turbomind/api.cc` | exported split-reduce ABI |
| `kernels/turbomind/ggml-turbomind/ggml-turbomind-ds4-probe.cu` | fixed six-route reduce kernel |
| `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h` | public ABI declaration |
| `kernels/turbomind/ggml-turbomind/test_grouped_gate_up_fusion.cpp` | focused correctness/timing harness |
| `tools/ds4-v100-run-appliance.sh` | opt-in launcher allowlist only |
| `logs/from-cluster/sprint213-routed-ffn-split-reduce/` | V100 evidence |
| `docs/sprints/SPRINT-213.md` | plan and outcome |

## Non-Goals

- No TP4/TP8 runtime integration.
- No PP scheduler changes.
- No `ds4_v100_scheduler.*` changes.
- No default promotion without served A/B evidence.
- No model-weight logs.
- No broad refactor of TurboMind dispatch.

## Definition Of Done

- [ ] Sprint plan exists and is committed before execution evidence is staged.
- [ ] Current dirty split-reduce implementation is audited and either fixed or
      explicitly rejected.
- [ ] V100 TurboMind build passes with `sm_70`.
- [ ] Symbol export for `ggml_turbomind_ds4_reduce6_half_to_float` is verified.
- [ ] Focused split-reduce correctness passes on V100.
- [ ] Focused timing compares materialized split-reduce against atomic
      down-reduce epilogue.
- [ ] Full scheduler smoke passes or the failure is root-caused and recorded.
- [ ] Served 16-slot/256K A/B is run with generated and continuation tok/s.
- [ ] Default serving mode remains unchanged unless the candidate clears the
      promotion gate.
- [ ] Logs are copied to `logs/from-cluster/sprint213-routed-ffn-split-reduce/`.
- [ ] Sprint 213 records validation, decision, and next action.
- [ ] `docs/sprints/STATUS.md` and `docs/sprints/VISION.md` are updated.
- [ ] Changes are committed with explicit `git add` paths.

## Decision Gate

Promote `fused6_split_reduce` only if all are true:

- focused correctness passes;
- full scheduler smoke passes;
- served continuation/decode tok/s beats same-binary `fused6_reduce + graph`
  by at least 3% or repeats clearly above run noise;
- graph capture remains functional with zero launch/capture failures;
- the implementation does not increase VRAM or context/slot constraints.

Reject or leave diagnostic-only if:

- it is correct but served performance is flat/regressed;
- it only improves focused microbenchmarks;
- it breaks graph capture;
- it requires scheduler complexity beyond the routed-FFN hot path.

## Risks

- The materialized half down-route buffer may add a memory transaction that
  erases any benefit from avoiding atomic down-reduce.
- Focused timing may be positive while served graph replay stays flat.
- Existing dirty code may need small correctness fixes before it can be fairly
  benchmarked.

## Security

No new service exposure beyond the existing local appliance benchmark. No model
weights or prompts should be copied into logs.

## Dependencies

- Sprint 199 promoted baseline.
- Sprint 212 TP4 rejection and decision to return to routed-FFN work.
- Existing V100 pod `llm/llamacpp-build-8gpu` and persistent appliance pack.
