# Sprint 582 - MTP Weight Integration (Converter + Contract, Phase A.1)

Date: 2026-05-29

## Goal

Begin the MTP/B1 workstream (steering priority 1): collapse the V100 MTP sidecar
into the unified pack pipeline so the canonical MTPBlock (layer 43) loads through
the same offline-pack + runtime-load path as layers 0-42. This sprint is
**correctness-only** (no perf): it produces the MTP weights in the pack contract,
not the forward pass or the specdec loop (those are later phases).

Per `MTP_IMPLEMENTATION.md`, the full MTP program is:
1. **(this sprint area)** Weight integration: safetensors->GGUF converter +
   `tp-ep-pack-contract.c` layer-43 extension + `runtime_pack.cu` binding +
   sidecar delete.
2. MTPBlock.forward in `engine/` (Phase A, later sprint).
3. TP/EP speculative-decode accept/reject loop (Phase B / the actual throughput
   sprint; this is what banks the EP-fill win the Sprint 581 gap attribution
   pointed at -- EP is 65% because of sub-1-token-per-expert).

## Scope (this sprint)

Phase A.1 of weight integration:
1. **Confirm the canonical MTP tensor set** in the HF safetensors cache
   (`/models/deepseek-v4-flash-safetensors-cache/model-00046-of-00046.safetensors`):
   the 32 tensor families (`research/ds4/ds4.c:3068-3104` `mtp_weights_bind`),
   their shapes, and source dtypes (`f8_e4m3_b128`, `mxfp4`, `bf16`, `f32`).
2. **Author the safetensors->GGUF converter** (~200 LoC, new tool) that emits an
   MTP-only GGUF fragment in the naming/packing convention the pack contract
   expects (256 routed experts stacked into `ffn_*_exps`).
3. **Extend `tools/tp-ep-pack-contract.c`** (~50-100 LoC) to emit layer-43 MTP
   rows using the existing EP8/TP8 sharding rules.

Deferred to Phase A.2+ (later sprints, recorded in steering): re-run the pack
pipeline against the unified GGUF, `runtime_pack.cu` binding, sidecar deletion,
MTPBlock.forward, and the specdec loop.

## Constraints

- Correctness-only; no launcher/default changes; the served path is unaffected
  until the forward + specdec phases land.
- The sidecar stays in place until its replacement is validated (do not delete
  this sprint).
- Reuse the existing format support in `ds4_source_formats.h`
  (`f8_e4m3_b128`/`mxfp4`/`bf16`/`f32`) -- no new dtype paths.

## Definition of Done

- The 32 MTP tensor families confirmed in the safetensors cache with shapes +
  dtypes recorded.
- The converter is implemented and produces an MTP GGUF fragment that round-trips
  the tensor set (validated against the sidecar GGUF's tensor list /
  `mtp_weights_bind` requirements).
- `tp-ep-pack-contract.c` emits layer-43 MTP rows; the contract parses and the
  rows shard correctly under EP8/TP8.
- Phase A.2+ items recorded in steering with prerequisites.
- Steering and vision updated.
- All repo changes committed, excluding user-owned
  `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`.

## Results (scoping + tensor inventory)

### MTP tensor set confirmed

`/models/deepseek-v4-flash-safetensors-cache/model-00046-of-00046.safetensors`
(3.6 GB) holds `1575` tensors, all `mtp.0.*` (the 256 routed experts are unpacked
as individual tensors -- `1575` = the HF unpacking of the 32 GGUF-convention
families). Source dtypes confirmed:

- Attention/projection weights (`wkv`, `wo_a`, `wo_b`, `wq_a`, `wq_b`, `e_proj`):
  **F8_E4M3** weight + **F8_E8M0** block scale.
- Norms (`attn_norm`, `enorm`, `q_norm`, `kv_norm`): **BF16**. `attn_sink`: **F32**.
- Routed experts `ffn.experts.{0..255}.w1/w2/w3`: **I8** weight + **F8_E8M0**
  block scale (shapes e.g. weight `[2048,2048]` / scale `[2048,128]`).

### Key finding: expert dtype mismatch reshapes the converter

`MTP_IMPLEMENTATION.md` assumed the converter needs "no new dtype paths," but the
expert weights are **I8+E8M0**, while the pack pipeline's `ds4_source_formats`
supports `bf16` / `f8_e4m3_b128` / `mxfp4` and the main-model experts are
**MXFP4**. There is no raw-I8 source path. So the converter must either
re-quantize I8+E8M0 -> MXFP4 (to match the pipeline) or a new I8 source path must
be added.

**Decision: re-quantize I8+E8M0 -> MXFP4 in the converter.** MTP is a *draft*
model -- the main model verifies every emitted token, so draft-side precision
loss only affects the speculative acceptance rate, not output correctness. MXFP4
(4-bit) is comparable to the sidecar's existing Q4_K MTP path, so draft quality is
preserved at parity with today. The attention/proj F8_E4M3+E8M0 and BF16/F32
tensors map directly to existing pipeline formats (no re-quant). This keeps the
converter on the existing `ds4_source_formats` surface (no new runtime dtype
path) at the cost of an offline I8->MXFP4 re-quant step in the converter.

## Status

Sprint 582 delivered the weight-integration scoping: tensor set confirmed with
dtypes/shapes, and the I8+E8M0 expert format reconciled to an MXFP4 re-quant
converter design. This corrects `MTP_IMPLEMENTATION.md`'s dtype assumption. The
converter implementation (now including the I8+E8M0 -> MXFP4 re-quant) plus the
`tp-ep-pack-contract.c` layer-43 extension move to Sprint 583; `runtime_pack.cu`
binding + sidecar delete to Sprint 584; then MTPBlock.forward and the specdec
loop. The served path is unchanged.
