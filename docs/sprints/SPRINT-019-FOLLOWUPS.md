# SPRINT-019 Follow-Ups

These are not blockers for the Sprint 019 `SHIP` verdict, but they block a
deployable DS4 V100 appliance.

## Compressor And Indexer Descriptor Binding

- **What:** Add attention compressor and ratio-4 indexer descriptors to
  `ds4_v100_layer_state`, then execute those paths inside
  `ds4_v100_layer_execute_decode` instead of passing prebuilt compressed KV.
- **Why:** Sprint 019 proves semantic attention over raw/compressed inputs, but
  real decode must generate and select compressed rows from source descriptors.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 020.
- **Files:** `ds4_v100_layer_state.*`, `ds4_v100_layer_execute.*`,
  `ds4_cuda.cu`, integrated layer smoke.

## HC Pre/Post Layer Scheduler

- **What:** Wrap the hidden-vector executor with DS4 HC attention and FFN
  pre/post composition so the runtime operates on `[4 x 4096]` HC state.
- **Why:** The real model layer path is HC-state based; Sprint 019 validates the
  hidden-vector body but not HC scheduling.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 020.
- **Files:** layer executor, HC CUDA helpers, scheduler tests.

## Full 43-Layer Selected-Token Gate

- **What:** Walk all layer classes through output head logits and compare a
  selected token against the source oracle.
- **Why:** Public usage should wait for real model selected-token evidence from
  the layer-scheduled V100 path.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 021+.
- **Files:** scheduler, output head, gate script.

## Production Arena Reuse

- **What:** Execute the integrated layer path against resident pack arenas and
  stage-owned KV arenas rather than bounded test arenas.
- **Why:** Sprint 019 still uploads selected source bytes into a bounded arena
  for validation.
- **Severity:** Important.
- **Suggested sprint:** Sprint 020+.
- **Files:** V100 context/residency wiring, scheduler runtime.

## Timing And Throughput Counters

- **What:** Add per-phase timing to `ds4_v100_layer_execute_report`.
- **Why:** The executor now has enough shape to start measuring attention, FFN,
  router, and memory-transfer costs separately.
- **Severity:** Important.
- **Suggested sprint:** Sprint 020+.
- **Files:** `ds4_v100_layer_execute.*`, gate/reporting tools.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Compressor/indexer descriptor binding | Critical | Sprint 020 | layer state/executor/CUDA/tests |
| HC pre/post layer scheduler | Critical | Sprint 020 | executor/scheduler/HC helpers |
| Full 43-layer selected-token gate | Critical | Sprint 021+ | scheduler/output/gate |
| Production arena reuse | Important | Sprint 020+ | context/residency/runtime |
| Timing and throughput counters | Important | Sprint 020+ | executor/reporting |
