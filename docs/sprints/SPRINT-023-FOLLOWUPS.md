# SPRINT-023 Followups

## Critical

1. **All-stage scheduler chain**

   Extend the current stage0 -> stage1 chain across stages 2-7 and validate a
   full 43-layer HC walk. Keep output-head selected-token separate unless the
   full chain is stable quickly.

2. **Output-head selected-token gate**

   After full stage chaining, collapse gpu7 HC through the output-head path and
   compare top-1 token against the source oracle vector.

3. **Failure-local reports**

   Populate scheduler reports before layer execution starts so failed stages
   still report stage id, gpu, resident bytes, and layer range.

## High

4. **Relay optimization**

   Replace the synchronous `cudaMemcpyPeer` handoff with the existing relay
   stream/buffer contract after correctness is proven across all stages.

5. **KV/cache ownership decision**

   The scheduler still uses executor-native F32 cache tensors. Decide whether
   to keep that for the selected-token milestone or bridge to the context-owned
   F16 KV arena first.

6. **Device-aware cache regression**

   Add a narrow unit/smoke that catches cross-device reuse of CUDA model-range
   cache entries without needing the full two-stage scheduler.

## Deferred

7. **MTP**

   Keep deferred until the base 43-layer selected-token path passes.

8. **Throughput**

   Measure only after selected-token correctness. The current peer copy and
   scheduler execution are correctness-first.
