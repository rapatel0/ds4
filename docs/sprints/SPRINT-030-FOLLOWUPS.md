# SPRINT-030 Followups

## Critical

1. **Implement K=1 MTP runtime parity**

   The sidecar now validates. The next milestone is a single-token MTP forward
   path that matches a known oracle before speculative serving is enabled.

   - **Why**: Gate readiness is now specifically `missing=mtp_runtime`.
   - **Suggested sprint**: Sprint 031.
   - **Files**: MTP tensor binding/upload code, `ds4_v100_replay.*`,
     scheduler/runtime files, MTP CUDA kernel bridge.

2. **Do not route MTP Q4_K through the MXFP4 TurboMind path**

   The MTP routed experts are Q4_K, while the main DS4 Flash expert path is
   MXFP4/F8 source-layout work. Treat MTP as a separate kernel family until
   parity proves otherwise.

   - **Why**: Prior MTP investigations localized mismatch risk to the MTP
     quantized FFN/output path.
   - **Suggested sprint**: Sprint 031.
   - **Files**: MTP kernel selection, sidecar binding, CUDA Q4_K/Q8_0 wrappers.

## High

3. **Add MTP sidecar memory to the V100 runtime planner**

   The validator reports 3.807600108 GB of sidecar tensor bytes. The planner
   should reserve and place that explicitly instead of relying on the old rough
   3.6 GiB placeholder.

   - **Why**: GPU7 has enough headroom, but runtime admission should use the
     measured sidecar bytes.
   - **Suggested sprint**: Sprint 031 or Sprint 032.
   - **Files**: `tools/ds4-v100-plan.c`, `ds4_v100_context.*`.

4. **Parallelize resident stage open/upload**

   Full-gate evidence still shows startup/upload in the 4-5 minute range. This
   is now the largest deployment usability cost after correctness.

   - **Why**: The HTTP process stays resident, but restarts are expensive.
   - **Suggested sprint**: Sprint 031-032.
   - **Files**: `ds4_v100_replay.c`, `ds4_v100_scheduler.c`.

5. **Run longer resident decode baselines**

   Current throughput evidence is one prompt plus one or two generated tokens.
   Longer resident loops are needed before optimizing aggregate tok/s.

   - **Why**: Short correctness smokes are not stable throughput benchmarks.
   - **Suggested sprint**: After K=1 MTP parity or in parallel.
   - **Files**: `tools/ds4-v100-replay.c`, benchmark scripts.

## Deferred

6. **MTP speculative serving**

   Defer until K=1 MTP draft top-1 matches the oracle and target verification
   state mutation is proven.

7. **Multi-slot scheduling**

   Defer until one-slot MTP and longer resident decode baselines are stable.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| K=1 MTP runtime parity | Critical | Sprint 031 | MTP runtime/replay/kernels |
| Keep Q4_K MTP separate from MXFP4 path | Critical | Sprint 031 | kernel selection |
| MTP memory planner update | High | Sprint 031-032 | planner/context |
| Parallel upload | High | Sprint 031-032 | replay/scheduler |
| Longer decode baselines | High | Sprint 031+ | replay/bench |
| Speculative serving | Deferred | After K=1 parity | runtime/server |
| Multi-slot scheduling | Deferred | After one-slot MTP | scheduler/server |
