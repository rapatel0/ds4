# SPRINT-011 Follow-Ups

These are not blockers for the Sprint 011 `SHIP` verdict, but they are blockers
before deployment or throughput work.

## Full Layer Output Gate

- **What:** Connect source projection, attention output projection,
  residual/HC update, and layer boundary relay for at least one bounded
  source-layout layer slice.
- **Why:** Sprint 011 proves projection-fed attention/compressor surfaces, not
  a coherent post-attention layer output.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 012.
- **Files:** `ds4_cuda.cu`, `ds4_v100_context_cuda.cu`, future scheduler files.

## Router And Expert Correctness

- **What:** Add source-faithful router, shared expert, and routed expert
  execution behind diagnostic gates.
- **Why:** DS4 quality depends on MoE routing and expert math; deployment
  cannot rely only on attention/KV correctness.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 012.
- **Files:** `ds4_cuda.cu`, TurboMind/tc-grid integration surfaces, scheduler
  files.

## Output Head And Selected Token

- **What:** Produce a bounded logits or selected-token comparison against the
  guarded source oracle.
- **Why:** This is the real serving readiness gate before public CLI/server
  exposure.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 012.
- **Files:** `tools/ds4-source-oracle-vector.c`, V100 diagnostic test/tool.

## Production F8 Projection Kernel

- **What:** Replace the scalar diagnostic F8 row-matmul with the intended V100
  production tile path once correctness is anchored.
- **Why:** Sprint 011 validates source-format math and residency, but the
  diagnostic kernel is not the desired performance path.
- **Severity:** Important.
- **Suggested sprint:** Sprint 012 or Sprint 014 depending on measured impact.
- **Files:** `ds4_cuda.cu`, kernel registry/scheduler files.

## Deployment Re-Sequencing

- **What:** Keep public appliance deployment behind the logits-producing V100
  source-layout gate.
- **Why:** Sprint 011 moved correctness forward but still lacks MoE and output
  head execution.
- **Severity:** Important.
- **Suggested sprint:** Sprint 013 after Sprint 012 ships.
- **Files:** `docs/sprints/VISION.md`, next sprint plans.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Full layer output gate | Critical | Sprint 012 | `ds4_cuda.cu`, context/scheduler files |
| Router and expert correctness | Critical | Sprint 012 | `ds4_cuda.cu`, kernel integration surfaces |
| Output head and selected token | Critical | Sprint 012 | oracle and V100 diagnostic tool |
| Production F8 projection kernel | Important | Sprint 012/014 | `ds4_cuda.cu`, scheduler files |
| Deployment re-sequencing | Important | Sprint 013 | vision and sprint plans |
