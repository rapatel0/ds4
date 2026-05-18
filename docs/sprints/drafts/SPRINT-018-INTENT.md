# SPRINT-018 Intent: V100 Descriptor-Bound Attention Projection/Residual/Norm Gate

## Seed Prompt

Continue sprint-plan and sprint-execute loops until the DS4 V100 appliance
vision is realized, with actual implementation each sprint.

## Orientation Summary

- Sprint 017 shipped `ds4_v100_layer_state`, moving descriptor-bound router/FFN
  ownership out of a standalone smoke and into a reusable runtime surface.
- The appliance gate still reports `ready=false`; the next numerical gap is
  attention/residual/norm integration before real selected-token decode.
- Existing `tests/cuda_v100_projection_attention_smoke.c` proves synthetic
  source-F8 projection and attention/KV primitives, while Sprint 017 proves
  real descriptor-bound FFN bytes. The missing bridge is real descriptor-bound
  attention projection/control bytes.
- `docs/architecture/DS4-V100-LAYOUT.md` remains the topology and dtype
  contract: source FP8/MXFP4/F32 layouts, no persistent dequantized weights,
  layer-sharded ownership, F16 KV first.

## Goal

Extend `ds4_v100_layer_state` to own attention/control descriptors and add a
V100 gate that runs real layer-2 attention projection, residual add, and norm
work from source bytes.

## Explicit Non-Claim

This sprint does not prove full attention softmax, compressed-KV visibility, or
a complete next hidden state. It proves the descriptor-bound source-byte
surfaces needed before that full layer-output sprint.

## Constraints

- Do not unlock serving.
- Do not persistently dequantize attention weights.
- Keep all source-F8 matrices in the GPU arena and decode inside the diagnostic
  kernel path.
- Use real pack-index offsets and real model bytes.
- Keep CPU references source-format faithful.

## Success Criteria

- `ds4_v100_layer_state` binds and validates attention descriptors:
  `attn_norm`, `attn_q_a`, `attn_q_a_norm`, `attn_q_b`, `attn_kv_latent`,
  `attn_output_a`, `attn_output_b`, and HC attention controls.
- A new descriptor-bound CUDA attention smoke:
  - maps the real model;
  - uploads real layer-2 attention FP8 source matrices to a device arena;
  - applies real F32 RMSNorm weights;
  - runs q/kv/output projection surfaces through V100 kernels;
  - performs a residual add and FFN pre-norm;
  - compares every step against CPU source-format references.
- The appliance gate includes and passes the new attention slice check.

## Verification Strategy

- Local:
  - `make tests/v100_layer_state_smoke`
  - `make tests/cuda_v100_descriptor_bound_attention_smoke.o`
  - `bash -n tools/ds4-v100-gate.sh`
  - `git diff --check`
- Cluster:
  - Build and run `tests/cuda_v100_descriptor_bound_attention_smoke` on layer 2.
  - Run full `tools/ds4-v100-gate.sh --build --pack-index ...`.

## Uncertainty

- Correctness: Medium. Real source-F8 projection and RMSNorm primitives are
  already tested, but this sprint composes more real descriptors.
- Scope: Medium. Full attention softmax must stay deferred.
- Architecture: Medium. The state API should expose enough attention metadata
  for the next sprint without hard-coding a final serving scheduler.

## Open Questions

1. Should Sprint 019 complete attention softmax/compressed-KV layer output, or
   use this attention slice plus FFN to reach bounded selected logits first?
2. When should production resident arena reuse replace partial smoke arenas?
