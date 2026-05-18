# SPRINT-011 Intent: V100 Source Projection And Attention Slice

## Seed Prompt

Continue after Sprint 010. The project now trusts stage-owned KV views,
diagnostic KV writes, and real compressor recurrence on V100. The next risk is
source-format math before full logits: prove that source FP8/BF16 tensors can
feed a bounded V100 projection and attention slice without BF16-native or FP32
GEMM fallbacks.

## Orientation Summary

- Sprint 010 shipped per-layer KV/state subviews inside each stage `kv_arena`.
- Stage-owned KV writes pass for ratio-4 and ratio-128 at 1M context on 8 V100s.
- Existing compressor recurrence kernels pass bounded CPU-reference checks for
  ratio-128, ratio-4 attention, and ratio-4 indexer-shaped paths.
- Normal source-layout generation remains guarded.
- The runtime still lacks a trusted source FP8 dense projection path for V100
  and does not yet execute a coherent source-layout layer output.

## Relevant Code Areas

| Area | Files |
|---|---|
| Source F8/BF16 decode helpers | `ds4_source_formats.[ch]`, `ds4_cuda.cu`, `ds4_gpu.h` |
| V100 context and KV ownership | `ds4_v100_context.[ch]`, `ds4_v100_context_cuda.cu` |
| Attention/compressor/indexer kernels | `ds4_cuda.cu`, `ds4_gpu.h` |
| Source oracle/guards | `tools/ds4-source-oracle-vector.c`, `ds4.c` |
| Architecture anchor | `docs/architecture/DS4-V100-LAYOUT.md` |

## Constraints

- V100 has no native BF16, FP8, or FP4 tensor-core execution.
- Do not add a broad FP32 GEMM fallback for model execution.
- Do not persistently materialize large dequantized source weights.
- Keep new execution paths diagnostic-only until bounded source comparison
  passes.
- Stay single-slot and bounded; no MTP, public serving, throughput scheduling,
  or tensor parallelism.

## Success Criteria

- Add a bounded source F8_E4M3_B128 dense projection primitive or diagnostic
  tile path that runs on V100 and compares against a CPU source-format
  reference.
- Add BF16 source-to-FP16/F32 diagnostic handling only where needed for small
  control/global tensors; no native BF16 claim and no FP32 large-matmul path.
- Feed real device projection-equivalent tensors into an attention/compressor
  slice for at least one ratio-4 layer and one ratio-128 layer.
- Compare bounded V100 outputs against CPU/source references with documented
  tolerance.
- Preserve all source-layout guards and existing Sprint 010 V100 smokes.

## Verification Strategy

- Local model-less object builds and context smokes.
- `git diff --check`.
- V100 `sm_70` CUDA smoke for source F8 dense projection/tile math.
- V100 `sm_70` CUDA smoke for bounded projection-to-attention/compressor slice.
- Real-model source-layout `--guards-only` validation.

## Open Questions

1. Should the first F8 dense primitive be a small diagnostic F32-accumulating
   kernel or a direct FP8-decode-to-FP16-HMMA tile?
2. Which layer tensors are the narrowest useful source-model projection gate:
   `attn_kv_latent`, `attn_q_a`, or an attention-output projection?
3. Should source-oracle intermediate comparison be added to the existing oracle
   tool, or remain a standalone CPU helper for this sprint?
