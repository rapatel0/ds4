# Sprint 060 Follow-ups

## Shared Expert Batch Path

- **Severity**: Critical
- **Target sprint**: Sprint 061
- **Files**: `ds4_v100_layer_execute.c`, `ds4_cuda.cu`, `ds4_gpu.h`
- **Issue**: The routed MXFP4 path now batches active slots without the
  per-slot input copy, but shared expert gate/up/down still executes per slot.
- **Evidence**: Sprint 060 improves two-slot generated tok/s to `3.915266`,
  but average GPU utilization remains `12.265%`.
- **Next step**: Add a batched shared F8 expert path or profile whether shared
  expert launches dominate after routed input-copy removal.

## Persistent Batch Views

- **Severity**: Important
- **Target sprint**: Sprint 061
- **Files**: `ds4_v100_layer_execute.c`, `ds4_v100_layer_execute.h`
- **Issue**: The batch executor still allocates lightweight tensor views for
  router-output and routed-output slices per layer call.
- **Evidence**: The pointer-input work removed major D2D input staging, leaving
  view churn and shared expert per-slot work as the next non-MTP overhead.
- **Next step**: Persist router and routed-output views inside
  `ds4_v100_layer_batch_scratch`.

## Higher Slot-Tier Retest

- **Severity**: Important
- **Target sprint**: Sprint 061+
- **Files**: `tools/ds4-v100-sustained-decode-bench.sh`,
  `logs/from-cluster/`
- **Issue**: The default path has only been re-benchmarked at one and two slots
  after pointer-input routing.
- **Evidence**: The two-slot path improved, but the practical serving target
  depends on higher active concurrency.
- **Next step**: Re-run 4-slot and 8-slot tiers at 128K/256K context before
  deciding whether to prioritize shared expert batching or MTP commit.
