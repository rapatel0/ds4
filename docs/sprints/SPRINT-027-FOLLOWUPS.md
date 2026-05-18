# SPRINT-027 Followups

## Critical

1. **Ship a public one-slot serving surface**

   The selected-token correctness gate now passes, so the next blocker to
   usability is an appliance entrypoint that can load the resident pack and
   serve a prompt on the V100 cluster.

   - **Why**: Readiness now blocks on `public_serving`.
   - **Suggested sprint**: Sprint 028.
   - **Files**: `ds4_server*.c`, `ds4_v100_scheduler.*`, deployment scripts.

2. **Add throughput timing and counters**

   Measure upload time, decode step time, output-head time, and selected-token
   end-to-end latency for the single-slot path before optimizing.

   - **Why**: We have correctness evidence but no performance baseline for the
     working path.
   - **Suggested sprint**: Sprint 028 or Sprint 029.
   - **Files**: `ds4_v100_scheduler.*`, `tools/ds4-v100-gate.sh`, benchmark
     harness.

## High

3. **Investigate layer-4 FFN numeric drift**

   The checkpoint smoke shows seed, layers 0-3, and layer-4 after-attention
   matching CPU, while layer-4 final HC diverges. Route ids and route weights
   match, so the likely area is FFN expert/shared-expert accumulation.

   - **Why**: The official selected-token passes, but this drift can affect
     robustness on other prompts and longer contexts.
   - **Suggested sprint**: Parallel diagnostic work during serving sprint.
   - **Files**: `ds4_v100_layer_execute.c`, `ds4_cuda.cu`,
     `tests/cuda_v100_scheduler_checkpoint_parity_smoke.c`.

4. **Add explicit FP8 KV validation**

   The default correctness path is F16 KV. FP8 KV is now an explicit scheduler
   option and needs separate correctness/performance evidence before use.

   - **Why**: FP8 KV may be useful for context or slot pressure, but should not
     silently replace the source-layout F16 baseline.
   - **Suggested sprint**: After serving baseline or in parallel if a worker is
     available.
   - **Files**: `ds4_v100_scheduler.*`, `ds4_v100_layer_execute.c`,
     checkpoint smoke.

## Deferred

5. **MTP**

   Keep MTP behind the one-slot serving baseline and throughput counters.

6. **Multi-slot scheduling**

   Keep multi-slot behind a measured single-slot path so slot batching is
   optimizing a known baseline rather than masking scheduler overhead.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| Public one-slot serving | Critical | Sprint 028 | server/runtime/deploy |
| Throughput timing/counters | Critical | Sprint 028-029 | scheduler/gate/bench |
| Layer-4 FFN numeric drift | High | Parallel diagnostic | layer executor/CUDA/checkpoint smoke |
| Explicit FP8 KV validation | High | After serving baseline | scheduler/layer/checkpoint smoke |
| MTP | Deferred | After serving and timing | runtime/server |
| Multi-slot scheduling | Deferred | After single-slot baseline | scheduler/server |
