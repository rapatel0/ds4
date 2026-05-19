# Sprint 057: Deterministic Token-Step Coalescing

## Status

Complete.

## Overview

Sprint 057 fixes the benchmark/request-loop race that made multi-slot sustained
decode unreliable. Sprint 056 proved the two-slot server could be configured for
tensor batching, but the benchmark often reported `tensor_batched_groups=0`
because the first HTTP handler drained the pending queue before peer requests
were enqueued.

This sprint adds a small server-side microbatch rendezvous so concurrent
same-token-count requests reliably enter the existing
`ds4_v100_replay_generate_batch` path. It also prototypes a deeper batched FFN
layer slice behind an opt-in environment variable, but does not enable that path
by default because the first V100 benchmark was slower.

## Goals

1. Make two-slot sustained decode deterministically exercise token-step batch
   generation.
2. Preserve one-slot behavior and bounded latency by using a short rendezvous
   deadline.
3. Keep selected-token correctness at hex `3136`.
4. Prototype a batched routed MXFP4 FFN layer slice for active slots, but ship
   it off by default if it does not improve throughput.
5. Capture cluster artifacts that distinguish the default coalescing path from
   the opt-in batched FFN experiment.

## Out of Scope

- Enabling the batched FFN layer slice by default without a speedup.
- Full attention/KV tensor batching across active slots.
- Persistent MoE kernels.
- Shared F8 expert batching.
- MTP draft commit.

## Implementation Notes

- `tools/ds4-v100-replay.c` now has a pending-request condition variable.
- `process_pending_generation_batch` waits up to roughly 5 ms when
  `active_microbatch > 1`, MTP is off, and the pending queue has not filled the
  active microbatch yet.
- The rendezvous exits as soon as enough peer requests arrive, so single-slot
  and sparse traffic remain bounded.
- `ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_batch_f32` generalizes the
  Sprint 056 grouped MXFP4 route primitive to `[tokens x routes]`.
- `ds4_v100_layer_execute_hc_decode_batch` and the scheduler batch call are
  available only when `DS4_V100_BATCH_LAYER_FFN` is set. The default path keeps
  the proven per-slot layer executor because the opt-in benchmark regressed.

## Definition of Done

- `cc -fsyntax-only -I. tools/ds4-v100-replay.c` passes.
- `cc -fsyntax-only -I. ds4_v100_layer_execute.c` passes.
- `cc -fsyntax-only -I. ds4_v100_scheduler.c` passes.
- `cc -fsyntax-only -I. tests/cuda_v100_mxfp4_moe_smoke.c` passes.
- Local object builds pass for touched C files.
- V100 `sm_70` build passes for `tests/cuda_v100_mxfp4_moe_smoke` and
  `tools/ds4-v100-replay`.
- Focused V100 MXFP4 smoke passes.
- Real 8-GPU replay still selects first token hex `3136`.
- Default sustained decode artifacts are captured under
  `logs/from-cluster/sprint057-coalescing-default`.
- Default two-slot sustained decode reports nonzero `tensor_batched_groups`.
- Opt-in batched FFN artifacts are captured separately under
  `logs/from-cluster/sprint057-batched-ffn-optin`.
