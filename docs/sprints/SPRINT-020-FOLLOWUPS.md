# SPRINT-020 Follow-Ups

These block a deployable DS4 V100 appliance after the Sprint 020 `EXTEND`
verdict.

## Executor-Owned Compressor And Indexer Rows

- **What:** Move attention compressor, ratio-4 indexer compressor, indexer
  scoring, and compressed-row visibility into `ds4_v100_layer_execute`.
- **Why:** Sprint 020 binds the real descriptors but the integrated executor
  still accepts test-supplied compressed KV rows.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 021.
- **Files:** `ds4_v100_layer_execute.*`, `tests/cuda_v100_integrated_layer_smoke.c`.

## Indexed Compressed Attention Path

- **What:** Use `ds4_gpu_attention_indexed_mixed_batch_heads_tensor` for
  ratio-4 decode once `n_comp > 512`, with top-k indices from the indexer.
- **Why:** Long-context ratio-4 attention should not scan all compressed rows
  through a dense additive mask.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 021.
- **Files:** executor, CUDA attention call glue, integrated smoke.

## CPU Reference For HC Output

- **What:** Add a bounded CPU HC reference for the layer-2 HC entrypoint.
- **Why:** Sprint 020 proves the HC path executes on V100 and returns finite
  state, but vector-level HC correctness still needs a reference compare.
- **Severity:** Important.
- **Suggested sprint:** Sprint 021 or 022.
- **Files:** `tests/cuda_v100_integrated_layer_smoke.c`.

## Full Layer Scheduler

- **What:** Walk layer classes 0-42 through HC state, KV ownership, and output
  handoff.
- **Why:** Layer-2 ratio-4 execution is not enough to claim selected-token
  decode.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 022+.
- **Files:** scheduler/runtime files, gate script.

## Throughput And Timing Counters

- **What:** Add phase timing for HC pre/post, attention projection/softmax,
  compressor/indexer, router, routed experts, shared expert, and relay.
- **Why:** Optimization should start from measured V100 bottlenecks after the
  row-generation path is real.
- **Severity:** Important.
- **Suggested sprint:** after Sprint 021 correctness.
- **Files:** `ds4_v100_layer_execute.*`, gate/reporting tools.
