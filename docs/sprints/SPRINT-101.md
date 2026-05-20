# Sprint 101 - Batch Attention Projection Semantic Repair

Date: 2026-05-20

## Objective

Repair and re-evaluate the opt-in batch attention projection path before using
it as a practical-serving optimization candidate.

## Context

Sprint 099 left `DS4_V100_ENABLE_BATCH_ATTN_PROJ=1` disabled because it was
flat to slightly slower. While planning the next throughput sprint, we found a
more important issue: the batch path was not semantically equivalent to the
single-slot path. The single path applies `attn_norm` before Q-A and KV
projection and passes that normalized row into compressed-KV preparation; the
batch path projected from the raw hidden row and passed the raw hidden row into
`prepare_decode_cache_attention()`.

## Plan

1. Fix `execute_attention_output_batch()` so it:
   - computes weighted attention RMS norm per active slot;
   - builds the projection pointer table from those normalized rows;
   - passes the normalized row into `prepare_decode_cache_attention()`;
   - preserves the existing opt-in gate.
2. Keep the path opt-in while measuring; do not default it unless V100 evidence
   clears the current production path.
3. Validate on the cluster:
   - `sm_70` replay build;
   - 4-slot, 1M context opt-in versus default;
   - 8-slot, 256K opt-in versus default.

## Definition of Done

- The opt-in batch path matches the single-path projection ordering.
- `DS4_V100_ENABLE_BATCH_ATTN_PROJ=1` still passes token-match correctness.
- Production default remains correct and at least as fast as Sprint 100.
- Sprint results and artifacts are recorded before commit.

## Implementation

- Repaired `execute_attention_output_batch()` so active rows first pass through
  the same weighted attention RMS norm as the single-slot path.
- Built the Q-A and KV projection pointer table from normalized attention rows,
  not raw hidden rows.
- Passed the normalized attention row into `prepare_decode_cache_attention()`
  so compressed-KV/indexer preparation sees the same input semantics as the
  single path.
- Added persistent batch-attention scratch tensors/views to
  `ds4_v100_layer_batch_scratch` to avoid reintroducing allocator churn.
- Tightened batch compatibility checks to reject mixed shard-offset/source-offset
  model maps in one batched projection call.
- Fixed CUDA smoke link rules that use `ds4_v100_context.o` so they include the
  TurboMind pack parser dependency.

## V100 Validation

Cluster target: `llamacpp-build-8gpu` on `gpu-01`, using k8s-local
`/workspace`, appliance `/workspace/ds4-appliance-full-tm-s090`, and copied
TurboMind library
`/workspace/ds4-sprint082/build/turbomind-v100/libggml-turbomind.so`.

Passed:

- `make tools/ds4-v100-replay ... CUDA_ARCH=sm_70 -j8`
- `./tests/cuda_v100_projection_attention_smoke`
- `DS4_V100_ENABLE_BATCH_ATTN_PROJ=1 ./tests/cuda_v100_stage_scheduler_smoke --stage 0 --slots 4`
- `DS4_V100_ENABLE_BATCH_ATTN_PROJ=1 ./tests/cuda_v100_full_scheduler_smoke --slots 4`
- `DS4_V100_ENABLE_BATCH_ATTN_PROJ=1 ./tests/cuda_v100_full_scheduler_smoke --slots 8`
- `./tests/cuda_v100_selected_token_smoke ... --expected-token-hex 3136`

Residual:

- `cuda_v100_scheduler_checkpoint_parity_smoke` still diverges at layer 12
  (`after_attn max_abs=0.0130593777`, `rms_abs=0.00201772526`) even on the
  default single-sequence path. This is not isolated to the Sprint 101 opt-in
  batch projection path and remains a separate parity debt item.

## Throughput

| Mode | Context | Slots | Batch Projection | Generated tok/s | Continuation tok/s | Token Match |
|---|---:|---:|---|---:|---:|---:|
| long default | 1,048,576 | 4 | off | 18.102742 | 16.971321 | 4/4 |
| long opt-in | 1,048,576 | 4 | on | 17.503345 | 16.409386 | 4/4 |
| throughput default | 262,144 | 8 | off | 26.402101 | 24.751970 | 8/8 |
| throughput opt-in | 262,144 | 8 | on | 26.432087 | 24.780082 | 8/8 |

Artifacts:

- `logs/from-cluster/sprint101-batch-attn-repair/soak-4slot-default`
- `logs/from-cluster/sprint101-batch-attn-repair/soak-4slot-batch`
- `logs/from-cluster/sprint101-batch-attn-repair/soak-8slot-default`
- `logs/from-cluster/sprint101-batch-attn-repair/soak-8slot-batch`

## Decision

`DS4_V100_ENABLE_BATCH_ATTN_PROJ=1` remains opt-in. The semantic repair is worth
keeping because the opt-in path is now equivalent to the single-slot projection
ordering, but the measured 4-slot long-context case regresses and the 8-slot
case is only noise-level faster. Production default remains the Sprint 100
path.
