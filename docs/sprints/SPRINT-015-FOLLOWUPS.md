# SPRINT-015 Follow-Ups

These are not blockers for the Sprint 015 `SHIP` verdict, but they block a
deployable appliance.

## Real Router Scheduling

- **What:** Use descriptor-bound `ffn_gate_inp`, `ffn_gate_tid2eid`, and
  `exp_probs_b` inputs to select routed experts for real layer execution.
- **Why:** Sprint 015 proves fixed-expert FFN compute from real bytes, not
  model-selected experts.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 016.
- **Files:** `ds4_cuda.cu`, `ds4_v100_context.*`, FFN/layer scheduler tests.

## Descriptor-Bound Layer State

- **What:** Introduce a layer execution struct that carries bound descriptors,
  activation tensors, scratch tensors, KV views, and relay metadata.
- **Why:** The current smoke owns bindings locally. Production decode needs a
  scheduler-owned structure that can run all layer stages.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 016.
- **Files:** new scheduler/context module, `ds4_v100_context.*`.

## Attention, Residual, And Norm Integration

- **What:** Extend descriptor-bound execution beyond FFN into attention
  projections, compressed KV/indexer updates, RMSNorm/control tensors,
  residual add, and HC transforms.
- **Why:** Serving requires full layer output, not only FFN output.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 016+.
- **Files:** `ds4_cuda.cu`, V100 context/scheduler, CUDA smokes.

## Selected-Token Real-Model Gate

- **What:** Drive a bounded real-model path to output-head logits and
  selected-token comparison.
- **Why:** The appliance should not unlock serving until it can produce a real
  selected token from real model bytes.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 017+.
- **Files:** scheduler, output-head path, gate script.

## Memory Reuse

- **What:** Reuse the production resident GPU arena in descriptor-bound compute
  instead of allocating a partial test arena with the highest descriptor offset.
- **Why:** Sprint 015's arena allocation is acceptable for the smoke, but the
  appliance must avoid duplicate resident weight storage.
- **Severity:** Important.
- **Suggested sprint:** Sprint 016+.
- **Files:** arena residency/context wiring.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Real router scheduling | Critical | Sprint 016 | CUDA/scheduler/context |
| Descriptor-bound layer state | Critical | Sprint 016 | scheduler/context |
| Attention, residual, and norm integration | Critical | Sprint 016+ | CUDA/scheduler/tests |
| Selected-token real-model gate | Critical | Sprint 017+ | scheduler/output/gate |
| Memory reuse | Important | Sprint 016+ | residency/context |
