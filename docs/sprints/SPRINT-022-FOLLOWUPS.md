# SPRINT-022 Followups

## Critical

1. **Cross-GPU scheduler handoff**

   Stage 0 now produces a resident HC output. The next scheduler step is a real
   handoff API that copies `[4 x 4096]` HC from the source stage device to the
   destination stage device and resumes execution at that stage's first layer.

2. **All-stage resident scheduler**

   Open resident arenas for gpu0-gpu7, upload each GPU shard, initialize all 43
   layer states and caches, and execute the complete layer chain for one slot.

3. **Output-head selected-token gate**

   Reuse the bounded BF16 logits path after stage 7. The minimum gate is top-1
   selected token comparison against an official source oracle vector, not full
   logit equivalence.

## High

4. **Executor cache ownership bridge**

   The current stage scheduler allocates executor-native F32 cache tensors.
   Decide whether Sprint 023 should continue with that contract for selected
   token first, or bridge the executor to the context-owned F16 KV arena before
   full 43-layer validation.

5. **Longer stage scheduler cache progression**

   Add a multi-step stage scheduler smoke so ratio-4 layers emit compressed
   attention and indexer rows in the scheduler path, not only in the single
   layer integrated smoke.

6. **Production stage scheduler input**

   `decode_token` is correct for stage 0. Later stages need a scheduler API
   that accepts incoming HC tensors from relay instead of reseeding from token
   embedding.

## Deferred

7. **MTP**

   Keep MTP off until the base 43-layer selected-token path is verified.

8. **Throughput and multi-slot wavefront**

   Do not optimize aggregate tok/s until the single-slot selected-token gate
   exists. After that, add slots to increase effective expert GEMM M and measure
   the scheduler overhead.

9. **Output-head tensor parallelism**

   Keep gpu7-owned output head for the first selected-token gate. Revisit vocab
   splitting only if output-head memory or latency becomes a measured blocker.
