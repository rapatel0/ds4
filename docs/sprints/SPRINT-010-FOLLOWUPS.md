# SPRINT-010 Follow-Ups

These are not blockers for the Sprint 010 `SHIP` verdict, but they are blockers
before deployment or throughput work.

## Full Source-Format Projection Path

- **What:** Wire real source FP8/BF16 projection tensors into the V100 layer
  slice instead of bounded synthetic compressor inputs.
- **Why:** Sprint 010 proved arena views and compressor recurrence, not full
  source-model dense projection correctness.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 011.
- **Files:** `ds4_cuda.cu`, `ds4_gpu.h`, layer scheduler files, source-format
  helpers.

## Attention And Layer Output

- **What:** Connect raw/compressed KV writes to attention heads, attention
  output projections, HC post-processing, and residual output for one bounded
  source-layout layer pair.
- **Why:** Deployment needs a coherent layer output, not isolated KV and
  compressor smokes.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 011.
- **Files:** `ds4_cuda.cu`, `ds4_v100_context_cuda.cu`, future scheduler files.

## Router, Expert, And Shared-Expert Baseline

- **What:** Add the first source-faithful router, shared expert, and routed
  expert execution baseline behind diagnostic gates.
- **Why:** DS4 quality depends on MoE routing and expert math; throughput kernel
  selection should wait until this path is correct.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 011+.
- **Files:** `ds4_cuda.cu`, TurboMind/tc-grid integration surfaces, scheduler
  files.

## Bounded Logits/Top-K Comparison

- **What:** Compare a V100 bounded logits or selected-token result against the
  guarded CPU source oracle.
- **Why:** This is the real deployment gate. KV/compressor correctness is
  necessary but not sufficient for serving.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 011.
- **Files:** `tools/ds4-source-oracle-vector.c`, V100 diagnostic test/tool.

## Deployment Re-Sequencing

- **What:** Move public appliance deployment after the logits-producing V100
  source-layout gate.
- **Why:** Sprint 010 confirmed deployment is premature without dense,
  attention, MoE, and output-head correctness.
- **Severity:** Important.
- **Suggested sprint:** Sprint 012 after Sprint 011 correctness.
- **Files:** `docs/sprints/VISION.md`, next sprint plans.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Full source-format projection path | Critical | Sprint 011 | `ds4_cuda.cu`, `ds4_gpu.h`, scheduler files |
| Attention and layer output | Critical | Sprint 011 | `ds4_cuda.cu`, context/scheduler files |
| Router/expert/shared-expert baseline | Critical | Sprint 011+ | `ds4_cuda.cu`, kernel integration surfaces |
| Bounded logits/top-k comparison | Critical | Sprint 011 | oracle and V100 diagnostic tool |
| Deployment re-sequencing | Important | Sprint 012 | vision and sprint plans |
