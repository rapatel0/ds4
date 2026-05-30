# Sprint 583 - MTP Converter Design + Re-Quant Spec (Phase A.1 cont.)

Date: 2026-05-29

## Goal

Implement the MTP safetensors->GGUF converter (with the I8+E8M0 -> MXFP4 expert
re-quant decided in Sprint 582) and extend `tools/tp-ep-pack-contract.c` for the
layer-43 MTP rows. Correctness-only; served path unaffected.

## Converter design (established this sprint)

Source: `/models/deepseek-v4-flash-safetensors-cache/model-00046-of-00046.safetensors`
(1575 `mtp.0.*` tensors). Target: an MTP-only GGUF fragment in the pack-contract
naming convention (256 routed experts stacked into `ffn_gate/up/down_exps`), on
the existing `ds4_source_formats` surface (no new runtime dtype path).

### Per-family mapping

| Source family | Source dtype | Target GGUF dtype | Transform |
| --- | --- | --- | --- |
| `attn.wkv/wo_a/wo_b/wq_a/wq_b`, `e_proj` | F8_E4M3 + F8_E8M0 | `f8_e4m3_b128` | direct (re-block scales to 128 if needed) |
| `attn_norm`, `enorm`, `q_norm`, `kv_norm`, `*norm.weight` | BF16 | `bf16` | direct |
| `attn.attn_sink`, `exp_probs_b.bias` | F32 | `f32` | direct |
| `ffn.experts.{i}.w1/w2/w3` | **I8 + F8_E8M0** | **`mxfp4`** | **re-quant** |
| `ffn_gate_inp` (router) | (check) | `f32`/`bf16` | direct |

### Expert re-quant (I8+E8M0 -> MXFP4)

MXFP4 (`ds4_source_formats.c`): 32-elem blocks, one E8M0 power-of-2 scale per
block, nibbles drawn from `{0,±0.5,±1,±1.5,±2,±3,±4,±6}` (`fp4_table`). Source
I8+E8M0 uses 16-elem scale blocks (`scale [2048,128]` for `weight [2048,2048]`).

Per expert weight row:
1. **Dequant** source: `f32[j] = i8[j] * ds4_src_e8m0_to_f32(e8m0[block16(j)])`.
2. **Re-quantize to MXFP4** in 32-elem blocks: per block, `amax = max|f32|`;
   `scale_exp = ceil(log2(amax / 6.0))` (6.0 = max fp4 magnitude); E8M0 byte =
   `scale_exp + 127`; each elem -> nearest `fp4_table` entry of `f32/2^scale_exp`,
   packed two nibbles/byte.
3. Emit MXFP4 row bytes matching `ds4_src_mxfp4_row_bytes(ncols)`.

This mirrors the `ds4_src_mxfp4_nibble_to_f32` / `ds4_src_e8m0_to_f32` decode so
the round-trip is consistent with how the runtime reads MXFP4. Draft-model
precision loss (8-bit -> 4-bit) is acceptable and ~Q4_K parity with the sidecar.

### Validation

- Tensor list of the emitted GGUF must satisfy `mtp_weights_bind()`'s 32-family
  requirement (`research/ds4/ds4.c:3068-3104`).
- Spot-check dequant: emitted MXFP4 expert rows decode (via
  `ds4_src_mxfp4_row_to_f32`) to within expected 4-bit error of the I8+E8M0
  dequant.
- The contract parses the layer-43 rows and they shard under EP8/TP8 with the
  same per-rank expert assignment as layers 0-42.

## Implementation status

Design and re-quant spec complete (this document). The converter tool
(`tools/mtp-safetensors-to-gguf.c`, modeled on `tools/pack.c` /
`tools/appliance-pack.cu` GGUF emission) and the `tp-ep-pack-contract.c`
layer-43 extension are the implementation work; they build and validate on the
pod against the real safetensors. This is the substantial code chunk of the MTP
weight-integration phase.

## Definition of Done

- Converter implemented; emits an MTP GGUF fragment that satisfies
  `mtp_weights_bind` and round-trips within 4-bit tolerance.
- `tp-ep-pack-contract.c` emits layer-43 MTP rows that parse and shard.
- Steering and vision updated.
- All repo changes committed, excluding user-owned
  `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`.
