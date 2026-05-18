# SPRINT-010 Intent: V100 Single-Slot Decode Integration

## Seed Prompt

Continue the DS4 V100 appliance sprint sequence after Sprint 009. The vision
now places Sprint 010 between bounded KV execution and deployment: integrate the
validated KV surfaces into a real single-slot V100 prefill/decode slice that
consumes actual projection/compressor outputs and compares a bounded result
against the source oracle.

## Orientation Summary

- Sprint 009 shipped deterministic F16 KV arena planning/allocation and a V100
  `sm_70` diagnostic prefill/KV smoke for ratio-128 and ratio-4/indexer state.
- The diagnostic KV update still uses host-provided F32 rows and synthetic
  state values; it does not consume production projection/compressor outputs.
- The stage-local CUDA context allocates one `kv_arena`, but production kernels
  do not yet have named subviews for raw SWA, compressed attention KV, indexer
  KV, and the separate KV/score state buffers.
- Normal source-layout generation remains guarded and must stay guarded while
  integration expands.
- The updated vision defers appliance deployment to Sprint 011; Sprint 010 is
  the correctness bridge from bounded row/state smokes to a real single-slot
  V100 decode/prefill slice.

## Relevant Code Areas

| Area | Files |
|---|---|
| V100 topology, KV budgets, and stage ownership | `ds4_v100_context.[ch]` |
| CUDA context allocation and stage resources | `ds4_v100_context_cuda.cu` |
| CUDA source-format, compressor, indexer, attention, and KV APIs | `ds4_gpu.h`, `ds4_cuda.cu` |
| Source-layout oracle and guard validation | `tools/ds4-source-oracle-vector.c`, `ds4.c`, `ds4_source_formats.[ch]` |
| Current KV smoke | `tests/cuda_v100_prefill_kv_smoke.c` |
| Architecture anchor | `docs/architecture/DS4-V100-LAYOUT.md` |

## Constraints

- V100 has no native BF16, FP8, or FP4 tensor-core compute. BF16/F8/MXFP4
  source data must feed explicit conversion, low-bit kernels, or FP16 HMMA with
  FP32 accumulation where appropriate.
- Do not persistently materialize dequantized copies of large source weights.
- Keep F16 KV as the correctness baseline.
- Keep source-layout generation fail-closed outside explicit diagnostic/oracle
  paths.
- Stay single-slot and bounded; do not use MTP, throughput scheduling, server
  deployment, or tensor parallelism as a shortcut.

## Success Criteria

- Stage-local KV arena subviews expose raw SWA, compressed attention KV,
  indexer KV, attention KV state, attention score state, indexer KV state, and
  indexer score state with deterministic offsets and sizes.
- A V100 diagnostic path writes into the stage-owned `kv_arena` rather than
  standalone smoke tensors.
- At least one ratio-128 and one ratio-4/indexer bounded path consumes actual
  device projection/compressor input tensors and uses the real compressor/state
  recurrence, not the Sprint 009 synthetic state formula.
- A source-oracle or CPU helper provides the bounded reference needed to compare
  the V100 slice.
- V100 `sm_70` cluster validation passes, source-layout guards still pass, and
  logs are archived under `docs/sprints/drafts/`.

## Verification Strategy

- Local model-less build/tests for context/subview math.
- `git diff --check`.
- V100 `sm_70` CUDA smoke covering stage-owned arena writes and real
  compressor/indexer recurrence.
- Cluster guard validation against `/models/DSv4-Flash-256e-fixed.gguf`.
- Bounded V100-vs-CPU/source comparison for the integrated single-slot slice.

## Uncertainty Assessment

| Dimension | Level | Notes |
|---|---|---|
| Correctness | High | Real compressor/indexer recurrence and source-oracle intermediate capture are not yet wired together. |
| Scope | Medium | The sprint is bounded, but "actual projection/compressor outputs" may expose missing source-format projection kernels. |
| Architecture | Medium | Stage-owned arena views are straightforward; the right oracle comparison surface needs care. |

## Open Questions

1. Should the first integrated slice use source-derived F8 decoded rows as
   projection stand-ins, or require a real bounded FP8 dense projection kernel?
2. Should the source oracle expose intermediate KV/state rows directly, or
   should Sprint 010 add a narrow CPU reference helper independent of the full
   oracle runner?
3. Should the first ratio pair be layers 2/3 on gpu0, or a nonzero stage to
   exercise stage-owned arena views away from the embedding stage?

## Vision Context

North Star: a DS4 V100 appliance that runs the high-intelligence source
quantized model from pure device-resident packs by default, preserves model
quality, and reaches verified deployment before throughput tuning.

Sprint 010 sits after Sprint 009's bounded KV execution and before Sprint 011
deployment. Its job is to turn the validated allocation/update surfaces into a
real single-slot correctness slice. Deployment, throughput scheduling, MTP, and
tensor-parallel variants stay deferred.
