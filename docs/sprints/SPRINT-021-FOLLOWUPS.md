# SPRINT-021 Follow-Ups

These block a deployable DS4 V100 appliance after the Sprint 021 `SHIP`
verdict.

## Full 43-Layer Single-Slot Scheduler

- **What:** Allocate per-layer decode caches, walk layers 0-42 through HC
  state, transfer HC at GPU boundaries, and run the output head/top-k.
- **Why:** Sprint 021 proves one representative ratio-4 layer. Usage requires a
  selected token from the whole model.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 022.
- **Files:** runtime scheduler files, `ds4_v100_layer_execute.*`, gate script.

## Production Indexer Threshold Stress

- **What:** Exercise ratio-4 indexed attention at the production
  `indexer_top_k=512` threshold with more than 512 compressed rows.
- **Why:** Sprint 021 validates indexed-attention plumbing by forcing
  `indexer_top_k=1`; the long-context production threshold still needs a
  hardware stress.
- **Severity:** Important.
- **Suggested sprint:** Sprint 022 or 023.
- **Files:** integrated smoke or long-context smoke.

## Reusable Scratch And Timing Counters

- **What:** Move per-call compressor/indexer/attention/FFN scratch allocation
  into reusable per-GPU scratch arenas and add phase timing counters.
- **Why:** The layer path is now correct enough to measure; allocation churn
  will distort throughput and latency.
- **Severity:** Important.
- **Suggested sprint:** after selected-token correctness begins.
- **Files:** `ds4_v100_layer_execute.*`, scheduler/runtime metrics.

## CPU Reference For HC Output

- **What:** Add a bounded CPU HC reference for the layer-2 HC entrypoint.
- **Why:** HC execution currently validates finite output and route ranges, not
  vector-level semantic equivalence.
- **Severity:** Important.
- **Suggested sprint:** Sprint 022+.
- **Files:** `tests/cuda_v100_integrated_layer_smoke.c`.

## Public Serving, MTP, And Multi-Slot Throughput

- **What:** Add the appliance server path, MTP draft/verify/commit, slot
  admission, and aggregate tok/s benchmarks.
- **Why:** These are deployment requirements, but they should follow full
  single-slot selected-token correctness.
- **Severity:** Critical for deployment, deferred for sequencing.
- **Suggested sprint:** after Sprint 022 selected-token gate.
- **Files:** server/runtime/scheduler/benchmark tools.
