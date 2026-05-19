# Sprint 053: Continuous Token-Step Microbatching

## Status

Complete.

## Overview

Sprint 053 extends the request-loop batching shipped in Sprint 048 from
first-token-only execution to same-length multi-token decode. The objective is
to keep active non-MTP requests resident across continuation steps so the V100
stage scheduler can advance multiple slots together instead of resetting and
replaying each request serially.

This is a practical-use sprint, not a final throughput claim. It should produce
code, a sustained benchmark comparison, and clear evidence about whether the
current batch scheduler actually improves utilization.

## Goals

1. Add a reusable replay API for same-length batched generation.
2. Route HTTP pending batches with equal `tokens` through that API.
3. Preserve existing fallback behavior for MTP, one-request batches, mixed token
   counts, and invalid requests.
4. Report tensor-batched capability in `/v100/status` when
   `active_microbatch > 1`.
5. Re-run sustained decode on the V100 cluster with at least one multi-slot
   case and compare against the Sprint 052 one-slot baseline.

## Scope

- `ds4_v100_replay.c` / `ds4_v100_replay.h`:
  - add `ds4_v100_replay_generate_batch`;
  - keep `ds4_v100_replay_generate_first_token_batch` as a wrapper for existing
    callers.
- `tools/ds4-v100-replay.c`:
  - batch same-token-count non-MTP pending requests;
  - copy per-request output rows back to the owning HTTP request;
  - keep serial fallback for unsupported cases;
  - advertise tensor batching in status for active microbatch modes.
- `docs/operations/DS4-V100-APPLIANCE.md` and `docs/sprints/VISION.md`:
  - update the serving contract and roadmap with measured Sprint 053 evidence.

## Out of Scope

- MTP draft commit.
- Mixed-length continuous batching.
- Streaming or OpenAI-compatible endpoints.
- Replacing hot-path FFN/attention kernels.
- Persistent grouped-MoE scheduling.

## Definition of Done

- `cc -fsyntax-only -I. ds4_v100_replay.c` passes.
- `cc -fsyntax-only -I. tools/ds4-v100-replay.c` passes.
- `make ds4_v100_replay.o tools/ds4-v100-replay.o` passes locally.
- `CUDA_ARCH=sm_70 make tools/ds4-v100-replay` passes on the V100 build pod.
- A sustained cluster benchmark runs with `tokens > 1` and `slots > 1`.
- Benchmark artifacts are copied under `logs/from-cluster/sprint053-*`.
- The sprint report records:
  - aggregate generated tok/s;
  - aggregate continuation tok/s;
  - GPU utilization;
  - token correctness;
  - whether throughput improved versus Sprint 052.
- The vision and runbook state what is now shipped and what remains.
