# DS4 V100 TP/EP Layer-2 Communication Boundaries

Date: 2026-05-23

This note captures the current TP/EP mental model for layer `2`, based on the
Sprint 228 pack contract and Sprint 239 full-layer smoke. It is a planning
reference, not a claim of final DS4 logits equivalence.

Assumptions for the bandwidth estimates:

- `slots = 32`
- `TP = 8`
- `hidden = 4096`
- `hidden_shard = 512`
- V100 NVLink budget is treated as roughly `150 GB/s` bidirectional per GPU
  for a well-scheduled peer/collective path.
- Ring all-reduce per-GPU wire traffic is estimated as
  `2 * (TP - 1) / TP * logical_payload = 1.75 * logical_payload`.
- Ring all-gather per-GPU receive traffic is estimated as
  `(TP - 1) * local_shard_payload`.
- These are payload estimates, not latency or kernel launch estimates.

The important point: **dtype does not define the communication boundary**.
The boundary is defined by the tensor-parallel sharding contract.

- Output-sharded matmul: usually needs the input available on each rank, then
  produces a local output shard. No all-reduce after the matmul.
- Input-sharded matmul: each rank computes a partial sum for the same output
  rows. Needs all-reduce or reduce-scatter after the matmul.
- Shard-local consumer: no all-reduce if it consumes the same hidden shard
  layout.
- Full-hidden consumer: needs all-gather or replicated computation.
- RMSNorm over hidden shards: needs a small all-reduce for norm statistics.
- EP experts: use dispatch/return or all-to-all style movement, not all-reduce.

## Layer-2 Table

| Stage / tensor | Native dtype | Shape | TP allocation | Current compute | Communication boundary |
|---|---:|---:|---|---|---|
| Hidden input | fp16/float fixture today | `[slots x 4096]` | intended TP8 hidden shard, 512 dims/GPU | fixture/full input in smoke | production needs hidden all-gather before output-sharded dense unless kernels consume sharded input |
| `blk.2.attn_norm.weight` | f32 | `[4096]` | replicated | not semantically implemented yet | RMSNorm over sharded hidden needs all-reduce for norm stats |
| `blk.2.attn_q_a.weight` | fp8 | `[4096 x 1024]` | output-sharded, 128 rows/GPU | FP8 decode in CUDA, FP32 accumulation | likely input all-gather before matmul; no all-reduce after |
| `blk.2.attn_q_a_norm.weight` | f32 | `[1024]` | replicated | control/check only | if q_a output is sharded, norm may need small all-gather/all-reduce depending final layout |
| `blk.2.attn_q_b.weight` | fp8 | `[1024 x 32768]` | output-sharded, 4096 rows/GPU | FP8 decode in CUDA, FP32 accumulation | no all-reduce if attention consumes local head shard |
| `blk.2.attn_kv_latent.weight` | fp8 | `[4096 x 512]` | output-sharded, 64 rows/GPU | FP8 decode in CUDA, FP32 accumulation | no all-reduce; produces local latent shard |
| `blk.2.attn_compress_gate.weight` | bf16 | `[4096 x 1024]` | output-sharded, 128 rows/GPU | BF16 to FP32 in CUDA | no all-reduce caused by BF16; boundary depends on compressor dataflow |
| `blk.2.attn_compress_kv.weight` | bf16 | `[4096 x 1024]` | output-sharded, 128 rows/GPU | BF16 to FP32 in CUDA | no all-reduce caused by BF16; boundary depends on compressor dataflow |
| `blk.2.indexer.compress_gate.weight` | bf16 | `[4096 x 256]` | output-sharded, 32 rows/GPU | BF16 to FP32 in CUDA | no all-reduce caused by BF16 |
| `blk.2.indexer.compress_kv.weight` | bf16 | `[4096 x 256]` | output-sharded, 32 rows/GPU | BF16 to FP32 in CUDA | no all-reduce caused by BF16 |
| `blk.2.indexer.proj.weight` | bf16 | `[4096 x 64]` | output-sharded, 8 rows/GPU | BF16 to FP32 in CUDA | no all-reduce caused by BF16 |
| `blk.2.indexer.attn_q_b.weight` | fp8 | `[1024 x 8192]` | output-sharded, 1024 rows/GPU | FP8 decode in CUDA, FP32 accumulation | no all-reduce if indexer attention consumes local shard |
| KV cache: `kv.attn.blk.2` | fp8 packed | `[rows_per_slot x 512]` | KV-dim sharded | TP runtime writes local shard | no all-reduce for write; attention read should remain local by head/KV shard |
| KV cache: `kv.indexer.blk.2` | fp8 packed | `[rows_per_slot x 128]` | KV-dim sharded | TP runtime writes local shard | no all-reduce for write |
| Compression state: `kv.comp_state.blk.2` | f32 | `[state]` | state-dim sharded | TP runtime alloc/check | no all-reduce for storage; compression update semantics still need final scheduling |
| Attention softmax/output | mixed | head-local / compressed KV | TP head/latent shard | not fully implemented yet | possible gather/reduce before output projection depending final attention layout |
| `blk.2.attn_output_a.weight` | fp8 | `[4096 x 8192]` | output-sharded, 1024 rows/GPU | FP8 decode in CUDA, FP32 accumulation | likely input all-gather if input is sharded |
| `blk.2.attn_output_b.weight` | fp8 | `[8192 x 4096]` | output-sharded, 512 rows/GPU | used in Sprint 239 compose gate | produces hidden shard; no all-reduce after |
| Residual add | fp16/fp32 working | `[slots x 512/GPU]` | hidden shard | CUDA compose | no all-reduce if all terms share hidden-shard layout |
| `blk.2.ffn_norm.weight` | f32 | `[4096]` | replicated | not semantic yet | RMSNorm over hidden shard needs all-reduce for norm stats |
| Router `blk.2.ffn_gate_inp.weight` | f32 | `[4096 x 256]` | replicated today | control/check only | either all-gather hidden for router or compute replicated small router path |
| Router `blk.2.ffn_gate_tid2eid` | i32 | `[6 x 129280]` | replicated | control/check only | no all-reduce; routing metadata |
| Shared FFN `blk.2.ffn_gate_shexp.weight` | fp8 | `[4096 x 2048]` | output-sharded, 256 rows/GPU | FP8 decode in CUDA, FP32 accumulation | local if arranged shard-to-shard; otherwise gather/reduce boundary |
| Shared FFN `blk.2.ffn_up_shexp.weight` | fp8 | `[4096 x 2048]` | output-sharded, 256 rows/GPU | FP8 decode in CUDA, FP32 accumulation | local if arranged shard-to-shard; otherwise gather/reduce boundary |
| Shared FFN `blk.2.ffn_down_shexp.weight` | fp8 | `[2048 x 4096]` | output-sharded, 512 rows/GPU | used in Sprint 239 compose gate | produces hidden shard; no all-reduce after |
| Expert `blk.2.ffn_gate_up_exps.weight` | mxfp4 | `[4096 x 4096 x 256]` | EP8, 32 experts/GPU | TurboMind MXFP4 grouped gated SiLU | expert dispatch to owning GPU; not all-reduce |
| Expert `blk.2.ffn_down_exps.weight` | mxfp4 | `[2048 x 4096 x 256]` | EP8, 32 experts/GPU | TurboMind MXFP4 grouped down | expert return/peer copy to destination hidden shards; not all-reduce |
| Next hidden shard | fp32 today in smoke | `[slots x 512/GPU]` | TP8 hidden shard | compose kernel | no all-reduce after compose |

## Current Sprint 239 Behavior

Sprint 239 does not fuse FP8 and BF16 layers. It validates dataflow:

```text
production packed bytes
  -> FP8/BF16 decode inside CUDA kernels
  -> TurboMind MXFP4 EP expert execution
  -> explicit EP contribution peer return
  -> resident next-hidden shard composition
```

The dense FP8/BF16 kernels are correctness-oriented scalar CUDA reductions,
not final HMMA/CUTLASS tensor-core kernels. The optimized production path
should replace those dense kernels and some standalone composition passes with
fused kernels where the layout permits.

## Boundary Summary

There is no automatic all-reduce boundary between FP8 and BF16. The likely
communication points are:

| Boundary | Likely operation | Reason |
|---|---|---|
| Sharded hidden into replicated/full-input dense | all-gather, or avoid by using sharded-input kernels | output-sharded dense needs full input unless redesigned |
| RMSNorm over hidden shards | small all-reduce | norm denominator spans hidden dimension |
| Input-sharded matmul | all-reduce or reduce-scatter | ranks hold partial sums |
| Head/KV-local attention | none or local gather | depends on final head/KV ownership |
| EP routed experts | dispatch + return / all-to-all | tokens move to expert owners and back to hidden owners |
| Hidden-shard residual/compose | none | all terms already in `[slots x 512/GPU]` layout |

## Expected Communication Payloads

The table below estimates communication at the target `32`-slot TP8 shape.
The "current dtype" column describes the Sprint 239 smoke where applicable.
The "candidate dtype" column describes the likely production direction if we
optimize communication.

| Boundary | Operation | Tensor shape per step | Current dtype | Candidate dtype | Logical payload | Per-GPU wire estimate | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| RMSNorm stats, attention norm | all-reduce | `[slots]` sum/sumsq | f32 | f32 | `32 * 4 B = 128 B` | `224 B/GPU` | Too small to justify quantization. Keep f32. |
| RMSNorm stats, FFN norm | all-reduce | `[slots]` sum/sumsq | f32 | f32 | `128 B` | `224 B/GPU` | Same as attention norm. |
| Hidden shard to full hidden | all-gather | local `[32 x 512]` | fp16/float fixture mixed | fp16 | local `32768 B`; full hidden `262144 B` | receive `229376 B/GPU` | Avoid if kernels can consume sharded input; otherwise cheap relative to dense GEMM. |
| Hidden shard to full hidden, conservative | all-gather | local `[32 x 512]` | fp32 in some smoke dense paths | fp16 preferred | local `65536 B`; full hidden `524288 B` | receive `458752 B/GPU` | Current smoke dense helpers use f32 inputs; production should not require f32 all-gather. |
| Output-sharded FP8 dense | none after matmul | output shard `[32 x rows/GPU]` | f32 output in smoke | fp16 or fp32 accum -> fp16 store | varies | `0` | Dtype transition FP8 to f32/fp16 is local. No all-reduce caused by FP8. |
| Output-sharded BF16 dense | none after matmul | output shard `[32 x rows/GPU]` | f32 output in smoke | fp16/fp32 depending accuracy | varies | `0` | No all-reduce caused by BF16. |
| Input-sharded matmul producing full hidden | all-reduce | full `[32 x 4096]` | not implemented | fp16 or fp32 | fp32 `524288 B`; fp16 `262144 B` | fp32 `917504 B/GPU`; fp16 `458752 B/GPU` | Prefer reduce-scatter to keep hidden sharded if the next consumer is sharded. |
| Input-sharded matmul producing hidden shard | reduce-scatter | full `[32 x 4096]` reduced to `[32 x 512]` | not implemented | fp16 or fp32 | same reduction volume as full output | fp32 about `917504 B/GPU`; fp16 about `458752 B/GPU` | Output stored as shard; wire volume similar to all-reduce but avoids replicated output. |
| Router input from hidden shards | all-gather or replicated small path | full hidden `[32 x 4096]` | control/check only | fp16 gather or local replicated router | fp16 full `262144 B` | receive `229376 B/GPU` | Router output is small; quantizing hidden gather below fp16 may risk routing quality. |
| Router logits/top-k | all-gather or broadcast if centralized | `[32 x 256]` logits/top-k | not implemented | f32 logits, i32 ids | logits `32768 B`; ids much smaller | depends on ownership | Usually cheaper than hidden gather. Accuracy-sensitive; avoid int8 logits unless validated. |
| EP dispatch activations | all-to-all / peer copies | `routes x 4096`, aggregate routes `192` | half in Sprint 239 EP input | fp16 | aggregate `1572864 B` | about `196608 B/GPU` avg payload before topology factor | Existing route schedule is balanced: `24` routes/GPU. |
| EP return hidden contributions | peer copies / all-to-all | source to dest `[32 x 512]` per pair | f32 in Sprint 239 compose | fp16 likely | f32 aggregate `4194304 B`; fp16 aggregate `2097152 B` | f32 `524288 B/GPU` send and receive; fp16 `262144 B/GPU` | This is the most obvious quantization candidate after correctness. |
| Hidden-shard residual/compose | none | local `[32 x 512]` | f32 in Sprint 239 compose | fp16/fp32 mixed | local only | `0` | Fuse with EP return reduction and dense output epilogues if possible. |
| KV write | none | local KV shard | fp8 packed | fp8 packed | local only | `0` | KV is already sharded by KV/head dim. |
| Attention over local heads/KV | local or small gather | head/KV local | not implemented | fp16/fp32 accum, fp8 KV | layout-dependent | layout-dependent | Keep attention head ownership aligned with KV shards to avoid full hidden collectives. |

At `150 GB/s`, even a `1 MiB` per-GPU collective has a raw bandwidth floor of
about `0.007 ms`. In practice, latency, synchronization, NCCL/peer scheduling,
and kernel boundaries dominate these small messages. The communication payloads
are therefore not the main concern by bytes alone; the main concern is whether
we force extra synchronization between many small kernels.

## Quantization Guidance For Boundaries

| Boundary | Quantize? | Rationale |
|---|---|---|
| RMSNorm all-reduce stats | No | Payload is tiny and accuracy-sensitive. Keep f32. |
| Hidden all-gather | Maybe fp16, not lower initially | Payload is modest; fp16 is natural on V100. Int8 hidden gather risks quality unless calibrated. |
| Input-sharded matmul reduction | Maybe fp16 output, keep fp32 accum inside kernel | V100 HMMA accumulates fp32 internally. Wire/storage can be fp16 if validation passes. |
| Router hidden/logits | Be conservative | Routing quality can dominate model quality. Use fp16 hidden and f32 logits/top-k first. |
| EP dispatch | fp16 | Expert inputs are already half in the TurboMind path. |
| EP return | Correct but not useful as a standalone pass yet | Sprint 241 validated fp16 return and halved `4 MiB -> 2 MiB` aggregate at 32 slots, but the extra cast/expand kernels slowed the resident loop. Revisit only when fused into EP reduction or next-hidden compose. |
| Next-hidden compose | fp16 storage with fp32 local accumulation is plausible | Compose can accumulate locally in fp32 and store fp16 for the next layer if quality holds. |

## Sprint 241 FP16 Return Measurement

Sprint 241 tested the EP return quantization hypothesis directly at
`32` slots / `256K`, MTP off, `50` resident decode-loop steps.

| Mode | Return bytes | ms/step | Slot-step tok/s | Compose ms/step | Result |
|---|---:|---:|---:|---:|---|
| FP32 return | 4194304 | 1.788149 | 17895.603225 | 0.713836 | PASS |
| FP16 return | 2097152 | 1.937399 | 16516.992775 | 0.859697 | PASS |

Conclusion: raw EP return bandwidth is not the limiter at this shape. FP16
return is correct, but standalone conversion adds more synchronization/kernel
cost than the peer-copy payload reduction saves. Keep FP32 return as the
default until conversion is fused into the reduction or compose kernel.
