# SPRINT-002 Seed: Loader, Pack Manifest, And First V100 Execution Path

Sprint 001 closed `SHIP` for source inventory and static memory planning. The
next implementation sprint should start from the measured source model, not the
older DS4 q2/q4 assumptions.

## Target

Load `/models/DSv4-Flash-256e-fixed.gguf` strictly enough to create an
inventory-backed pack manifest and prepare the first per-GPU runtime shards.

## Required Source Support

- `GGML_TYPE_MXFP4 = 39`
- `GGML_TYPE_F8_E4M3_B128 = 42`
- BF16 token embedding and output head
- F32 HC/control/router tensors
- BF16 compressor/indexer KV and gate tensors
- F8 attention, shared expert, and `indexer.attn_q_b.weight`

## Required Name Mapping Deltas

- `attn_kv_latent.weight`
- `attn_compress_ape`
- `attn_compress_norm.weight`
- `attn_compress_gate.weight`
- `attn_compress_kv.weight`
- `indexer.compress_ape`
- `indexer.compress_norm.weight`
- `indexer.compress_gate.weight`
- `indexer.compress_kv.weight`
- `hc_head_base`, `hc_head_fn`, `hc_head_scale`

## First Implementation Order

1. Add narrow source type/name validation without arbitrary GGUF support.
2. Emit a manifest using the Sprint 001 schema.
3. Add per-GPU ownership fields to manifest descriptors.
4. Add upload/pack stubs for source BF16/F32/F8/MXFP4 families.
5. Implement the routed expert MXFP4 grouped path first.

## Guardrails

- No persistent F16 dequantized copy of large weights.
- No blanket INT8 routed expert expansion.
- No MTP, server batching, or tensor parallelism until a base path is coherent.
