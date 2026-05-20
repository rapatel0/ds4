# DS4 V100 Layout And Scheduling Sketch

This note captures the current working design for running a DeepSeek V4 Flash
style quantized model on an 8x V100-SXM2-32GB host. It is not a final runtime
contract. It is a starting point for choosing formats, topology, and kernels
while keeping VRAM residency and data movement explicit.

## Ground Rules

- V100 does not have native BF16, Blackwell FP4, or FP8 tensor-core compute.
  BF16/FP4/FP8 are source and packed runtime formats that feed explicit
  conversion, custom unpack/dequant, integer, or FP16-HMMA kernels.
- When this note says FP16 in the runtime columns, it usually means activation,
  cache, or accumulator-adjacent data on V100. It does not mean the source
  DeepSeek V4 Flash tensor is necessarily FP16.
- Do not materialize persistent dequantized weight buffers. If a kernel needs
  dequant, it must happen inside the kernel or in a bounded scratch tile.
- Load exactly one runtime pack variant at a time for each tensor family. Do
  not keep MXFP4, FP8, and INT8 copies resident together.
- Keep activations FP16 through the layer. Use FP32 internally for reductions,
  softmax, routing scores, and debug paths.
- Convert source BF16 weights to FP16 runtime storage, or decode BF16 into
  FP16 scratch tiles, before any production V100 GEMM. Use FP32 only for the
  CPU oracle and small control/reduction paths, not for large BF16 matmuls.
- Start with F16 KV cache. Add F8 KV only after decode correctness and
  long-context accounting are stable.
- Keep cross-GPU payloads small. The layer-sharded baseline only moves hidden
  context at stage boundaries.

## Baseline A: Layer-Sharded 8-GPU Runtime

This is the first topology to implement because it minimizes cross-GPU traffic
and preserves a simple ownership model.

| Stage | GPU | Layers | Layer Types | Persistent Ownership |
|---:|---:|---|---|---|
| 0 | gpu0 | 0-5 | 0-1 SWA-only, 2/4 ratio-4, 3/5 ratio-128 | token embedding, layers 0-5 weights/KV |
| 1 | gpu1 | 6-11 | 3 ratio-4, 3 ratio-128 | layers 6-11 weights/KV |
| 2 | gpu2 | 12-17 | 3 ratio-4, 3 ratio-128 | layers 12-17 weights/KV |
| 3 | gpu3 | 18-23 | 3 ratio-4, 3 ratio-128 | layers 18-23 weights/KV |
| 4 | gpu4 | 24-29 | 3 ratio-4, 3 ratio-128 | layers 24-29 weights/KV |
| 5 | gpu5 | 30-34 | 3 ratio-4, 2 ratio-128 | layers 30-34 weights/KV |
| 6 | gpu6 | 35-39 | 2 ratio-4, 3 ratio-128 | layers 35-39 weights/KV |
| 7 | gpu7 | 40-42 | 2 ratio-4, 1 ratio-128 | layers 40-42, output head |

gpu7 intentionally owns fewer transformer layers because the output head and
future MTP state will pressure it.

## Per-GPU Arenas

| Arena | Contents | Format |
|---|---|---|
| weight arena | packed layer weights owned by the GPU | source-faithful kernel-native packs |
| KV arena | layer-local raw SWA, compressed KV, indexer KV, slot state | F16 first, F8 later |
| scratch arena | bounded GEMM/attention/router temporaries | kernel-specific, reused |
| relay arena | boundary hidden-context transfer | `[2][active_slots][4][4096]` FP16 normal, FP32 debug |

Hard planner budget should include weights, KV for configured slots/context,
scratch, relay, CUDA/cuBLAS overhead, output head, and MTP if enabled. Target
planned peak should leave several GiB free per GPU instead of filling 32 GiB.

## Source Dtype And Memory Estimate Notes

The "native/source dtype" column below means the high-intelligence DSv4 Flash
source model layout measured from `/models/DSv4-Flash-256e-fixed.gguf`, not the
older antirez q2/q4 GGUF family and not the Q8_0/Q4_K MTP sidecar. The Sprint
001 inventory recorded this source mix: 129 MXFP4 tensors, 365 F8_E4M3_B128
tensors, 147 BF16 tensors, 684 F32 tensors, and 3 I32 tensors.

Memory estimates use fixed DS4 dimensions and these planning bytes:

| Format | Planning Bytes |
|---|---:|
| F16/BF16 | 2.000 B/value |
| F32 | 4.000 B/value |
| F8_E4M3_B128 | about 1.008 B/value including one scale per 128 values |
| Q8_0 | about 1.063 B/value |
| MXFP4 | about 0.531 B/value including one scale per 32 values |
| INT8 candidate | about 1.000 B/value plus scale metadata |

MiB values are approximate and refer to persistent resident bytes unless the row
explicitly says cache/scratch. Runtime pack overhead can move these numbers by a
few percent; the planner should use exact packed byte counts once packers exist.

## Global Tensors

| Tensor Family | Dimensions | Expected Native/Source Dtype | Starting Runtime Layout | Est. Resident Bytes |
|---|---:|---|---|---:|
| token embedding | `[4096 x 129280]` | BF16 | FP16 row-gather table on gpu0, converted from source BF16 | about 1010 MiB |
| output HC control | `hc_head_fn [16384 x 4]`, base `[4]`, scale `[1]` | F32 | F32 small tensors on output GPU | about 0.25 MiB |
| output norm | `[4096]` | F32 | F32 | 0.016 MiB |
| output head | `[4096 x 129280]` | BF16 | FP16 projection on gpu7 converted from source BF16; later FP8/Q8 or vocab TP only after quality gate | about 1010 MiB |

## Transformer Layer Schedule

Kernel names below are kernel families, not final C symbol names.

| Step | Path | Tensor Dimensions | Expected Native/Source Dtype | Starting Runtime Layout | Est. Resident Bytes / Layer | Expected Kernel Family | Output |
|---:|---|---:|---|---|---:|---|---|
| 1 | HC attention pre | `hc_attn_fn [16384 x 24]`, base `[24]`, scale `[3]` | F32 | F32 small tensors | about 1.50 MiB | DS4 HC pre kernel | FP16 sublayer vector |
| 2 | attention RMSNorm | `attn_norm [4096]` | F32 | F32 | 0.016 MiB | DS4 RMSNorm kernel | FP16 normed hidden |
| 3 | Q low-rank A/B | `attn_q_a [4096 x 1024]`, `attn_q_b [1024 x 32768]` | F8_E4M3_B128 expected | source FP8 blocked pack | about 37.1 MiB | FP8 dequant + FP16 HMMA dense kernel | FP16 Q |
| 4 | KV projection | `attn_kv_latent [4096 x 512]` | F8_E4M3_B128 | source FP8 blocked pack | about 2.1 MiB | FP8 dequant + FP16 HMMA dense kernel | FP16 KV row |
| 5 | RoPE + KV append | KV row `[512]`; raw SWA `[128 x 512]` | cache, not source weight | F16 cache first | 0.125 MiB raw SWA per layer per slot | DS4 RoPE/cache append kernel | updated raw SWA KV |
| 6 | compressed attention | ratio-4 `attn_kv [262272 x 512]` at 1M; ratio-128 `[8320 x 512]` at 1M | cache, not source weight | F16 cache first | ratio-4 about 256.1 MiB per slot; ratio-128 about 8.1 MiB per slot | DS4 attention kernel | FP16 attention heads |
| 7 | attention compressor | ratio-4 `attn_compress_ape [1024 x 4]`, KV/gate `[4096 x 1024]`; ratio-128 APE `[512 x 128]`, KV/gate `[4096 x 512]` | APE/norm F32, KV/gate BF16 | FP16 compressor projections converted from BF16 plus F32 control tensors | ratio-4 about 16.0 MiB; ratio-128 about 8.3 MiB | DS4 compressor kernels | compressed KV rows |
| 8 | ratio-4 indexer | `indexer.attn_q_b [1024 x 8192]`, proj `[4096 x 64]`, compressor KV/gate `[4096 x 256]` | attn_q_b F8_E4M3_B128, proj/KV/gate BF16, APE/norm F32 | source FP8 plus FP16 projections converted from BF16 and F32 control tensors | about 12.6 MiB on ratio-4 layers only; indexer KV cache adds about 64 MiB per slot at 1M | DS4 indexer score + top-k kernels | selected compressed rows |
| 9 | attention output A/B | `attn_output_a [4096 x 8192]`, `attn_output_b [8192 x 4096]` | F8_E4M3_B128 expected | source FP8 blocked pack | about 66.0 MiB | FP8 dequant + FP16 HMMA dense kernel | FP16 attention output |
| 10 | HC attention post | uses HC attention control state | F32 | F32 small tensors | included in step 1 | DS4 HC post kernel | FP16 HC |
| 11 | FFN RMSNorm | `ffn_norm [4096]` | F32 | F32 | 0.016 MiB | DS4 RMSNorm kernel | FP16 FFN input |
| 12 | router | `ffn_gate_inp [4096 x 256]`, optional bias `[256]`, hash `tid2eid [6 x 129280]` on layers 0-2 | F32; I32 hash metadata | F32/I32 | about 4.0 MiB; hash table adds about 3.0 MiB on layers 0-2 | small dense + top-k/router kernel; replay hot path keeps selected ids/weights on device after Sprint 058 | expert ids/weights |
| 13 | routed gate/up experts | two tensors `[4096 x 2048 x 256]` | MXFP4 / FP4 expert source expected | source MXFP4 grouped pack first | about 2176 MiB MXFP4; about 4096 MiB INT8 candidate | Sprint 060 pointer-input grouped selected-route MXFP4 gate+up+SwiGLU kernel | FP16/FP32 expert mid scratch |
| 14 | SwiGLU | routed mid `[active_routes x 2048]` | activation only | FP16/FP32 internal | scratch only | fused into grouped routed gate/up for MXFP4 path | FP16/FP32 expert mid scratch |
| 15 | routed down experts | `[2048 x 4096 x 256]` | MXFP4 / FP4 expert source expected | same expert pack | about 1088 MiB MXFP4; about 2048 MiB INT8 candidate | Sprint 056 grouped selected-route MXFP4 down-sum kernel | FP16/FP32 routed output |
| 16 | shared expert | gate/up `[4096 x 2048]`, down `[2048 x 4096]` | F8_E4M3_B128 | source FP8 dense pack first | about 24.2 MiB | safe dense/shared-expert kernel | FP16 shared output |
| 17 | HC FFN pre/post + combine | `hc_ffn_fn [16384 x 24]`, base `[24]`, scale `[3]` | F32 | F32 small tensors | about 1.50 MiB | combine + DS4 HC post kernel | FP16 next HC |

Layer type differences:

| Layers | Extra Work | Scheduling Note |
|---|---|---|
| 0-1 | SWA-only attention | no compressed KV/indexer growth |
| even 2-42 | ratio-4 compression + indexer | dominates long-context KV and bandwidth |
| odd 3-41 | ratio-128 compression, no indexer | much lighter long-context path |

Approximate persistent weight totals by layer class, assuming FP8 dense tensors
and MXFP4 routed experts:

| Layer Class | Count | Est. Weight Bytes / Layer | Major Additions |
|---|---:|---:|---|
| SWA-only layers 0-1 | 2 | about 3397 MiB | no compressor/indexer; layers 0-1 also carry hash table |
| ratio-4 layers 2,4,...,42 | 21 | about 3433 MiB | attention compressor + ratio-4 indexer |
| ratio-128 layers 3,5,...,41 | 20 | about 3405 MiB | attention compressor only |

At 1M context with F16 KV, per-slot cache adds roughly:

| Layer Class | Est. KV Bytes / Layer / Slot |
|---|---:|
| SWA-only | 0.125 MiB |
| ratio-4 | 256.1 MiB `attn_kv` + 64.0 MiB `indexer_kv` |
| ratio-128 | 8.1 MiB `attn_kv` |

Compression-state buffers are not included in the table above; budget an
additional 0.5-1.5 GiB aggregate per 1M slot until exact source inventory and
runtime allocation are measured.

## Proposed Shard Memory Estimate

This is a planning estimate for the baseline layer map above, not a substitute
for exact packer byte counts. It assumes FP8 dense, MXFP4 routed experts, F16
KV, one 1M slot, and no MTP.

| GPU | Layers | Est. Weight Bytes | Est. 1M F16 KV / Slot | Global Extra | Est. Total Before Scratch |
|---:|---|---:|---:|---:|---:|
| gpu0 | 0-5 | about 20.0 GiB | about 0.64 GiB | token embedding about 0.99 GiB | about 21.6 GiB |
| gpu1 | 6-11 | about 20.0 GiB | about 0.96 GiB | none | about 21.0 GiB |
| gpu2 | 12-17 | about 20.0 GiB | about 0.96 GiB | none | about 21.0 GiB |
| gpu3 | 18-23 | about 20.0 GiB | about 0.96 GiB | none | about 21.0 GiB |
| gpu4 | 24-29 | about 20.0 GiB | about 0.96 GiB | none | about 21.0 GiB |
| gpu5 | 30-34 | about 16.7 GiB | about 0.95 GiB | none | about 17.7 GiB |
| gpu6 | 35-39 | about 16.7 GiB | about 0.65 GiB | none | about 17.4 GiB |
| gpu7 | 40-42 | about 10.0 GiB | about 0.63 GiB | output head about 0.99 GiB | about 11.6 GiB |

The planner should reserve several GiB beyond these estimates for scratch,
allocator fragmentation, CUDA/cuBLAS overhead, compression state, relay buffers,
and optional MTP. INT8-expanded expert packs should be rejected unless this
table still fits with that reserve.

## Weight Pack Choices

| Tensor Family | First Pack To Try | Alternatives | Gate |
|---|---|---|---|
| dense attention Q/KV/output | source FP8 blocked pack | INT8 dense pack | no persistent dequant, coherent decode |
| routed experts | source MXFP4 grouped pack | INT4 grouped, INT8 expanded, FP8 grouped | must fit VRAM and pass decode-quality checks |
| shared expert | source FP8 dense pack | INT8 dense/shared pack | prior unsafe dense path must stay disabled until tested |
| router | F32 | INT8 only if fused and exact enough | routing quality |
| output head | FP16 projection converted from source BF16 | vocab TP, FP8/Q8/INT8 after quality gate | memory/latency and top-k correctness |
| KV cache | F16 | F8 | long-context correctness |

The expert path is the main fork. MXFP4 best preserves VRAM and source
semantics. INT8 may be easier for existing integer kernels, but it expands
expert weight bytes and must beat that extra HBM traffic.

### Copied Low-Bit Kernel Policy

Prior tc-grid and TurboMind experiments in `~/repos/deepseek` are design
evidence only until their source is copied into this repository and built from
`ds4`. Sprint 080 copied tc-grid's V100 INT8 `v13_rf_v6` proof under
`kernels/tc-grid/`. Sprint 081 copied TurboMind's C ABI wrapper and required
lmdeploy `turbomind` support tree under `kernels/turbomind/`.

The current evidence favors TurboMind as the next routed-expert adapter target:
it builds from the copied tree, passes grouped MXFP4 compare on DS4 gate/up and
down shapes, keeps the expert source format MXFP4, and now passes a DS4
routed-output adapter smoke that packs source bytes, groups selected route rows
by expert, applies DS4 SwiGLU/route weights, and matches the existing
source-MXFP4 arena reference. Sprint 083 also wires this through the DS4 CUDA
wrapper behind `DS4_V100_TURBOMIND_ROUTED_FFN=1`, but that bridge transiently
repacks one expert matrix family at a time and is therefore a correctness
bridge rather than the final throughput layout. The production format should
store TurboMind-ready expert packs offline or use a planner-admitted cache so
the runtime does not keep both source MXFP4 and TurboMind-packed experts
resident for every layer. tc-grid remains useful for INT8 benchmarking and
possible future quality-gated INT8 expert packs.

Sprint 084 adds the first offline sidecar producer for that production format:
`tools/ds4-v100-turbomind-pack` reads the normal V100 `pack-index.tsv`, pulls
source MXFP4 expert bytes from the DS4 GGUF, and writes `gpuN.turbomind` plus
`turbomind-pack-index.tsv`. The runtime should later load this sidecar as a
separate acceleration artifact, reconstruct device pointer tables after upload,
and account for its bytes separately in the memory planner.

Any chosen production kernel must avoid persistent duplicate MXFP4, FP8, and
INT8 resident packs unless the planner explicitly admits the memory cost.

## Baseline Execution Schedule

| Mode | Schedule |
|---|---|
| correctness | run active slot batch gpu0 -> gpu7; copy HC only at stage boundaries |
| throughput | wavefront slot batches so gpu0 works on batch N while gpu1 works on N-1 |
| batching | batch active slots inside each stage to raise effective M for grouped experts; Sprint 057 makes request coalescing deterministic, Sprint 058 removes replay-only router readback sync, Sprint 059 enables scratch-backed multi-slot layer batching by default, Sprint 060 removes routed FFN input-copy staging with a pointer-input MXFP4 batch primitive, and Sprint 061 keeps shared F8 batching opt-in after V100 evidence showed flat slot scaling |
| transfer | boundary payload is `[active_slots][4][4096]`, FP16 normal or FP32 debug |

## Tensor-Parallel Version To Evaluate

Full tensor parallelism is not the first implementation target because it
replaces rare HC boundary copies with collectives inside every layer. It is
still worth evaluating when memory pressure, output-head latency, or expert
utilization requires it.

The best first tensor-parallel topology to inspect is a 2-way TP inside four
pipeline stages:

| TP Stage | GPUs | Layers | Ownership |
|---:|---|---|---|
| 0 | gpu0,gpu1 | 0-10 | embedding replicated or row-sharded; TP layer weights/KV |
| 1 | gpu2,gpu3 | 11-21 | TP layer weights/KV |
| 2 | gpu4,gpu5 | 22-32 | TP layer weights/KV |
| 3 | gpu6,gpu7 | 33-42 | TP layer weights/KV, output head TP |

This halves per-GPU weight bytes inside a stage but introduces per-layer
communication. It should be compared against the layer-sharded baseline only
after the planner can estimate both memory and communication.

### TP Per-Layer Schedule

| Path | TP Split | Communication | Notes |
|---|---|---|---|
| HC pre/post | replicated small weights | optional all-reduce if split | probably keep replicated |
| RMSNorm | hidden split or replicated | all-reduce sumsq if split | replicated hidden is simpler |
| Q low-rank A | column split | q_a norm needs all-reduce or all-gather | communication-sensitive |
| Q low-rank B | row/head split | all-reduce or head-sharded attention | design depends on Q layout |
| KV projection | replicated or column split | all-gather if shared KV needed | MLA single-KV makes this awkward |
| attention heads | head split | all-gather before output or row-parallel output | possible but complex |
| attention output A/B | column then row split | all-reduce after output B | viable later |
| router | replicated | no collective except top-k consistency | small, keep replicated |
| routed gate/up experts | split FFN intermediate dim | none before SwiGLU | clean TP candidate |
| routed down experts | row split FFN intermediate dim | all-reduce 4096 output | clean TP candidate |
| shared expert | same as routed FFN | all-reduce 4096 output | clean TP candidate |
| output head | vocab split | local top-k then small top-k merge | best TP exception |

### Selective TP Variant

Before full 2-way TP, evaluate selective tensor parallelism as exceptions inside
the layer-sharded runtime:

| Exception | GPUs | Why It May Pay |
|---|---|---|
| vocab-parallel output head | gpu6,gpu7 or all 8 | avoids final GPU output-head latency/memory concentration |
| routed expert intermediate split | owner GPU + partner GPU | splits the largest FFN GEMMs with one all-reduce per expert block |
| shared expert split | owner GPU + partner GPU | same pattern as routed FFN, but always active |

Selective TP keeps the rest of the graph layer-owned and only adds collectives
where the memory or compute savings are obvious.

## Starting Recommendation

Start with the layer-sharded design:

```text
dense weights:     source FP8 packed
routed experts:    source MXFP4 grouped packed
shared expert:     source FP8 packed
embedding/output:  FP16 converted from source BF16
activations:       FP16
KV cache:          F16
HC relay:          FP16 normal, FP32 debug
topology:          8-stage contiguous layer shard
scheduler:         correctness first, then slot wavefront
```

Then evaluate tensor parallelism in this order:

1. vocab-parallel output head;
2. 2-way TP routed/shared FFN intermediate split;
3. 2-way TP full stage topology;
4. larger TP groups only if NVLink topology and communication measurements
   justify them.
