---
sprint: 009
title: V100 Prefill And Compressed KV Execution
status: completed
date: 2026-05-18
verdict: SHIP
---

# SPRINT-009 Report: V100 Prefill And Compressed KV Execution

## Verdict

`SHIP`

Sprint 009 shipped the first bounded V100 F16 KV execution surface. The runtime
now derives deterministic stage-local KV arena offsets from the Sprint 008
budget, allocates those arenas in the CUDA V100 context, and has a CUDA
diagnostic smoke that updates raw SWA, compressed attention KV, ratio-4 indexer
KV, and F32 diagnostic state surfaces on `sm_70`.

Normal source-layout generation remains fail-closed.

## What Shipped

- Added `ds4_v100_kv_arena_plan` to publish raw SWA, compressed attention,
  indexer, and compression-state offsets/sizes per V100 stage.
- Made the CUDA V100 context allocate/free the derived per-stage KV arena.
- Extended `ds4_v100_context_print_report` with stable KV arena offsets.
- Added model-less assertions for stage-local KV offsets and arena totals.
- Added `ds4_gpu_v100_prefill_kv_update_f16_tensor`, a diagnostic V100 F16 KV
  update primitive with explicit ratio, slot, raw-row, compressed-row, and
  indexer bounds.
- Added `tests/cuda_v100_prefill_kv_smoke.c`, covering ratio-128 raw/compressed
  KV, ratio-4 raw/compressed/indexer KV, F32 state updates, invalid ratio,
  invalid slot, row bounds, and missing indexer surfaces.
- Fed an F8_E4M3_B128 source row through the Sprint 008 CUDA row-decode probe
  before using it as the bounded KV input tile.
- Kept all dequantization bounded to scratch/output rows in the diagnostic
  smoke.

## Evidence

Local validation:

- `docs/sprints/drafts/SPRINT-009-PHASE1-LOCAL.log`
  - Built the context smoke/tool and ran `v100_context_smoke: ok`.
- `docs/sprints/drafts/SPRINT-009-KV-ARENA.log`
  - Shows 1M single-slot stage-local KV arena offsets and totals.
- `docs/sprints/drafts/SPRINT-009-LOCAL-VALIDATION.log`
  - Builds model-less context targets and the CUDA prefill/KV smoke object.
  - Runs `v100_context_smoke: ok`.
- `docs/sprints/drafts/SPRINT-009-FINAL-DIFF-CHECK.log`
  - Empty output from final `git diff --check`.

Cluster validation:

- `docs/sprints/drafts/SPRINT-009-CUDA-PREFILL-KV-FINAL.log`
  - Builds `tests/cuda_v100_prefill_kv_smoke` with `CUDA_ARCH=sm_70`.
  - `cuda_v100_prefill_kv_smoke: ok`.
- `docs/sprints/drafts/SPRINT-009-CUDA-CONTEXT-256K.log`
  - `cuda_v100_context_smoke --production --kv-ctx 262144 --kv-slots 1`
    passes on 8 V100-SXM2-32GB GPUs.
- `docs/sprints/drafts/SPRINT-009-CUDA-CONTEXT-1M.log`
  - `cuda_v100_context_smoke --production --kv-ctx 1048576 --kv-slots 1`
    passes on 8 V100-SXM2-32GB GPUs.
- `docs/sprints/drafts/SPRINT-009-CUDA-CONTEXT-OVERBUDGET.log`
  - `--kv-ctx 1048576 --kv-slots 64` fails closed below reserve.
- `docs/sprints/drafts/SPRINT-009-GUARDS-ONLY.log`
  - `tools/ds4-source-oracle-vector --guards-only` passes against
    `/models/DSv4-Flash-256e-fixed.gguf`.

## Deviations

- The CUDA prefill/KV primitive is diagnostic and bounded. It validates F16 KV
  storage/update semantics and state-surface bounds, but it does not yet run
  production attention projection, compressor projection, RoPE, RMS norm, or
  full layer prefill.
- The F32 state surfaces are represented as combined diagnostic KV+score state
  values in the smoke. Production code should split or view them according to
  the compressor implementation that consumes them.
- The sprint did not expose serving, logits, MTP, tensor parallelism, or
  throughput benchmarking.

## Sprint 010 Handoff

Sprint 010 should integrate the bounded KV surfaces into a real single-slot
layer execution path before public deployment. The next useful gate is a
source-layout V100 decode/prefill slice that consumes actual projection outputs,
updates F16 KV/indexer state through the layer scheduler, and compares a
bounded result against the Sprint 007/008 CPU source oracle.
