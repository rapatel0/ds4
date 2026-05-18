# SPRINT-016 Follow-Ups

These are not blockers for the Sprint 016 `SHIP` verdict, but they still block
a deployable DS4 V100 appliance.

## Scheduler-Owned Layer State

- **What:** Move descriptor-bound router/FFN execution out of the standalone
  smoke and into a reusable layer execution state that owns bindings, source row
  views, route metadata, scratch tensors, and arena references.
- **Why:** Sprint 016 proved real router-selected FFN compute, but serving needs
  a scheduler-owned object that can compose attention, FFN, residual, norm, and
  layer-to-layer relay.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 017.
- **Files:** `ds4_v100_context.*`, new scheduler/runtime module, descriptor
  FFN tests.

## Attention, Residual, And Norm Integration

- **What:** Extend descriptor-bound execution to attention projections,
  compressed KV/indexer updates, RMSNorm/control tensors, residual adds, and HC
  transforms.
- **Why:** Router-selected FFN alone is not a layer output. The appliance should
  remain not-ready until a real layer slice can produce the next hidden state.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 017+.
- **Files:** `ds4_cuda.cu`, V100 context/scheduler, CUDA smokes.

## Real-Model Selected-Token Gate

- **What:** Drive a bounded real-model path far enough to produce output-head
  logits and a selected-token comparison from real descriptor-bound bytes.
- **Why:** The current selected-token gate is synthetic. Serving should not
  unlock until the gate proves a real model path.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 018+.
- **Files:** scheduler, output-head path, `tools/ds4-v100-gate.sh`.

## Production Arena Reuse

- **What:** Run descriptor-bound compute against the production resident GPU
  shard arena instead of allocating a partial test arena sized to the highest
  touched source offset.
- **Why:** The smoke proves addressing and math, but production must avoid
  duplicate resident weight storage.
- **Severity:** Important.
- **Suggested sprint:** Sprint 017+.
- **Files:** residency/context wiring, scheduler runtime.

## Representative Router Coverage

- **What:** Add representative coverage for bias-router or non-hash router layer
  classes if the pack index exposes different router semantics outside layer 2.
- **Why:** Sprint 016 validates the layer-2 hash-router path only.
- **Severity:** Important.
- **Suggested sprint:** After scheduler-owned layer state exists.
- **Files:** descriptor gate, router smoke, scheduler tests.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Scheduler-owned layer state | Critical | Sprint 017 | scheduler/context/tests |
| Attention, residual, and norm integration | Critical | Sprint 017+ | CUDA/scheduler/tests |
| Real-model selected-token gate | Critical | Sprint 018+ | scheduler/output/gate |
| Production arena reuse | Important | Sprint 017+ | residency/context |
| Representative router coverage | Important | After layer state | descriptor/router tests |
