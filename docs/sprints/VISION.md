---
created: 2026-05-17
last_updated: 2026-05-23
last_updated_by: vision
revision: 242
archived_previous: docs/sprints/archive/VISION-2026-05-23-pre-tp-hard-cut.md
---

# Vision: DS4 V100 TP/EP Appliance

## North Star

Build a DeepSeek V4 Flash appliance for the 8x V100-SXM2-32GB cluster that
runs the source quantized model from pure device-resident packs, preserves
quality, and reaches practical high-throughput serving through a native
TP/EP topology.

Hard cut: from this revision forward, no new work is spent on PP/layer-split
variants. The old layer-scheduled appliance remains only a frozen correctness
and throughput baseline. All new implementation work targets TP/EP. MTP is
deferred until TP/EP serving is operational and benchmarked.

Target topology:

```text
8x V100:
  pipeline parallel = 1
  tensor parallel   = 8
  expert parallel   = 8
  KV cache          = sharded
  slots target      = 32
  context target    = 256K minimum
  model path        = source quantized, device resident
```

Every GPU should participate in every layer. Dense paths are tensor-parallel.
Routed MoE paths are expert-parallel, using the existing low-bit TurboMind /
CUTLASS kernel work where it helps. The execution goal is to make decode look
like batched mat-mat work over active slots, not single-slot mat-vec work and
not a serial layer-chain.

## Current State

- The PP/layer-scheduled appliance is deployed and useful as a baseline, but
  it is no longer the optimization target.
- Sprint 225 fixed the immediate MTP reset/snapshot blocker:
  `long_memory_archive` full-prompt reset parity and target-block restore now
  pass.
- Sprint 225 also corrected the benchmark contract:
  single-slot replay is diagnostic only, while practical throughput must be
  measured with multi-slot serving and `active_microbatch == slots`.
- The current frozen production-shaped PP baseline from Sprint 225 is:
  `32` slots / `256K`, `64/64` token match, `50.434232` generated tok/s,
  `47.282093` continuation tok/s, average GPU utilization `47.076%`, max
  GPU utilization `96%`.
- Existing TP work is prototype-only. TP is not operational in production
  serving.
- Sprint 226 converted the TP planner into a TP8/EP8-only contract. It no
  longer exposes PP/layer-split topology modes. Against the real production
  pack bytes, the target `32` slots / `256K` / F8-KV shape fits at about
  `27.00 GiB` per GPU including a `2.00 GiB` reserve, with `5.00 GiB`
  headroom.
- Sprint 227 built the TP8 collective workbench. The doubling all-reduce
  boundary is correct and density-sensitive: `1189` overhead-only tok/s at
  32 tokens, `2119` at 64, and `3332` at 128 for the 43-layer,
  two-collective proxy. Root/direct RS+AG is correct but slower and is not the
  first runtime boundary candidate.
- Sprint 228 emitted the TP/EP pack contract from the real production pack.
  The contract has dense TP rows, replicated control/router rows, EP expert
  ownership, and KV/state descriptors, with a balanced `27.024 GiB` per-GPU
  estimate at `32` slots / `256K` / F8 KV.
- Sprint 229 added the first separate TP runtime skeleton. It opens all eight
  GPUs, enables peer access, allocates target hidden/KV/compression/scratch
  arenas for `32` slots / `256K`, runs a fixture pass, and tears down cleanly.
- Sprint 230 added explicit per-layer sharded KV row ownership to the separate
  TP runtime. Ratio-4/indexer and ratio-128 dense/KV slices pass on the V100
  pod at `32` slots / `256K` / F8 KV with `max_abs=0`.
- Sprint 231 added the bounded EP routed-expert slice. A new TP/EP-only smoke
  runs the real TurboMind MXFP4 grouped gated-SiLU and grouped down kernels on
  all eight V100s at the `32` slot / `top_k=6` target, with finite exact repeat
  output and explicit route/latency reporting.
- Sprint 232 added the first one-layer TP/EP fixture gate. The same process
  opens the target TP runtime, verifies a ratio-4 sharded KV row, and runs
  real TurboMind MXFP4 EP experts on all eight GPUs at `32` slots / `256K` /
  `top_k=6`.
- Sprint 233 validated real TP/EP contract ownership for layer `2`: dense TP,
  replicated control/router, EP experts, sharded KV, and compression state are
  present and balanced across all eight GPUs with zero ownership mismatches.
- Sprints 239-242 now run a representative layer-2 TP/EP resident loop from
  production packed bytes at `32` slots / `256K`, MTP off. Sprint 242 fused
  the FP32 EP remote-sum into next-hidden compose, improving the 50-step
  layer-loop metric from `1.784008 ms/step` to `1.641832 ms/step` and from
  `17937.138290` to `19490.418145` slot-step tok/s while preserving checksum.
- Prior TP evidence remains useful:
  - TP8 sharded KV at `32` slots / `256K` fits, while replicated KV does not.
  - TP8 one-layer synthetic and FP16 fixture probes proved resident TP work can
    live inside an all-GPU boundary.
  - The current TurboMind MXFP4 TP8 shard-256 path failed correctness; TP4
    controls were correct but did not justify production integration.
  - Routed-only overlays and PP scheduler TP patches are rejected.

## Non-Negotiable Constraints

- No new PP/layer-split optimization sprints.
- No generic scheduler abstraction to support both PP and TP.
- TP/EP code uses separate files and a separate runtime ownership model.
- PP code may be read for reference and used as a frozen baseline, but not
  extended as the forward path.
- Single-slot tests are correctness/latency diagnostics only.
- Throughput evidence must use multi-slot server mode, report prompt tok/s,
  generated tok/s, continuation tok/s, GPU utilization, and confirm
  `active_microbatch == slots`.
- MTP stays out of the critical path until TP/EP serving is correct and
  measured.

## Sprint Sequence

### Sprint 226 - TP/EP Planner And Topology Contract [complete]

Goal: Create a TP-only planner and topology report for `PP1/TP8/EP8` at
`32` slots / `256K`.

Rationale: The PP planner carries legacy assumptions that will fight the new
topology. The TP path needs its own memory, KV, expert, collective, and slot
admission contract before runtime work starts.

Outcome: Complete. `tools/ds4-v100-plan-tp.c` is now a TP8/EP8-only planner
with sharded KV, expert ownership, route-density, admission-tier, and
collective/EP traffic reporting. The real-pack V100 run reports `145.42 GiB`
total resident weight bytes, `27.00 GiB` per-GPU total at `32` slots / `256K`
/ F8 KV, and admission of `63` slots at `256K` under current assumptions.

### Sprint 227 - TP8 Collective Workbench [complete]

Goal: Build TP-only collective smokes for hidden all-reduce, reduce-scatter,
all-gather, and expert-output reduction across all eight V100s.

Rationale: The suspected TP risk is not raw NVLink bandwidth alone; it is
latency, synchronization, and whether collectives can stay resident and
overlapped inside the layer boundary.

Outcome: Complete. `tools/ds4-v100-tp8-collective-workbench` now measures
`allreduce`, `reduce-scatter`, `allgather`, `rs-ag`, and `ep-reduce` modes.
At 32 tokens, the hidden all-reduce proxy is `26.904544 ms` and the EP reduce
proxy is `27.436756 ms`; both pass correctness. At 128 tokens they improve to
`3332.257` and `3253.920` overhead-only tok/s respectively.

### Sprint 228 - TP/EP Pack Contract [complete]

Goal: Emit a TP/EP pack layout with dense TP shards, EP expert ownership, KV
shard descriptors, and per-GPU memory accounting.

Rationale: Runtime work should not reinterpret PP pack metadata. The pack
format must encode the TP/EP ownership model directly.

Outcome: Complete. `tools/ds4-v100-tp-ep-pack-contract` emits
`tp-ep-pack-contract.tsv`, `tp-ep-memory-summary.tsv`, and
`tp-ep-pack-contract.md`. The real-pack contract has `4096` dense TP rows,
`5496` replicated control/router rows, `688` EP expert rows, and `840`
KV/state rows. Per-GPU total is `27.024 GiB` at the target shape.

### Sprint 229 - TP Runtime Skeleton [complete]

Goal: Add a new TP-only runtime skeleton that opens all eight GPUs, allocates
resident hidden/KV/scratch arenas, and executes no-op or fixture layer passes.

Rationale: The runtime must prove ownership, lifecycle, and memory residency
without touching `ds4_v100_scheduler.*` as a shared abstraction.

Outcome: Complete. `ds4_v100_tp_runtime.{h,cu}` and
`tools/ds4-v100-tp-runtime-smoke.cu` now provide a separate TP runtime
skeleton. The V100 smoke allocates `7061329920` runtime bytes per GPU before
weights at the target shape and verifies fixture output with
`fixture_max_abs=0`.

### Sprint 230 - TP Dense And KV Slice [complete]

Goal: Implement a bounded dense-attention/KV slice in the TP runtime, including
sharded DS4 compressed KV at the `32` slot / `256K` target.

Rationale: TP must keep hidden state and KV in native sharded layout across
layers. This sprint answers whether dense paths and KV are viable before MoE
complexity is added.

Outcome: Complete. `ds4_v100_tp_runtime_dense_kv_slice` now computes
per-layer, per-slot sharded KV offsets and writes/reads deterministic resident
KV rows on all eight GPUs. At the target `32` slots / `256K` / F8 KV shape,
the runtime allocates `7122628608` bytes per GPU before weights. Layer 2
ratio-4 with indexer KV passes at `attn_row=384`, `indexer_row=256`,
`attn_row_bytes=65`, `indexer_row_bytes=17`, and `max_abs=0`. Layer 3
ratio-128 without indexer KV passes at `attn_row=192`, `attn_row_bytes=65`,
and `max_abs=0`. This keeps the TP runtime path viable and moves the next
implementation gate to EP routed experts.

### Sprint 231 - EP Routed Expert Slice [complete]

Goal: Implement a bounded EP routed-expert slice using real low-bit expert
kernels and measure expert dispatch, route imbalance, and grouped GEMM density
at `32` active slots.

Rationale: Expert execution dominates the useful work. EP is only valuable if
active slots create dense enough expert batches and dispatch/reduction does not
erase the kernel gains.

Outcome: Complete. `tools/ds4-v100-tp-ep-expert-smoke.cu` models EP8
ownership as `256` global experts and `32` local experts per GPU, then runs
the real TurboMind MXFP4 grouped gated-SiLU and grouped down kernels on all
eight V100s. At `32` slots / `top_k=6`, it reports `192` aggregate routes,
`1.5 MiB` dispatch, `1.5 MiB` return, balanced route imbalance `1.0`,
`repeat_max_abs=0`, `repeat_bad=0`, `repeat_nan=0`, and `PASS`. Rank `7` is
the slow rank at `0.249378 ms` versus roughly `0.059 ms` on ranks `0-6`, so
per-rank timing must remain visible in Sprint 232.

### Sprint 232 - One-Layer TP/EP Correctness Gate [complete]

Goal: Execute one TP/EP fixture layer that combines the separate TP runtime,
sharded KV, and real low-bit EP expert kernels.

Rationale: This is the first point where the separate TP runtime lifecycle,
sharded KV, and EP experts meet in one process before descriptor-backed real
layer data is introduced.

Outcome: Complete as a fixture gate. `tools/ds4-v100-tp-ep-layer-smoke.cu`
links the separate TP runtime with the TurboMind MXFP4 ABI in one process. At
`32` slots / `256K` / `top_k=6`, it opens the target runtime arenas, verifies
layer-2 ratio-4 KV with `max_abs=0`, executes `192` aggregate EP routes,
reports `1.5 MiB` dispatch and `1.5 MiB` return, and passes finite deterministic
repeat output. The fixture one-layer envelope is `1.321812 ms`, with
`1.078032 ms` in the dense/KV fixture and `0.243780 ms` worst-rank EP time.
Next: replace fixture weights/routes with descriptor-driven one-real-layer
TP/EP correctness while preserving the separate codepath.

### Sprint 233 - Descriptor Driven TP/EP Layer Gate [complete]

Goal: Validate real production-pack TP/EP contract descriptors for one
representative layer.

Rationale: Sprint 232 proved fixture execution. Before running real layer data,
the TP/EP path must prove that the production pack contract contains the dense,
control/router, EP expert, KV, and compression rows needed by the separate
runtime.

Outcome: Complete as a descriptor ownership gate. Layer `2` resolves to
`288` rows: `112` dense TP, `136` replicated control/router, `16` EP expert,
`16` KV shard, and `8` compression-state rows. Each GPU owns `36` rows and
`711945176` estimated bytes, with expert spans `0..31` through `224..255` and
zero ownership mismatches. This does not yet bind real bytes into execution;
that is the next sprint.

### Sprint 234 - Descriptor-Backed One-Layer Execution [complete]

Goal: Bind the layer-2 TP/EP descriptor rows to actual production-pack byte
spans and feed descriptor-derived expert pointers into the one-layer TP/EP
smoke.

Rationale: Descriptor ownership is now proven, but the runtime still executes
synthetic MXFP4 fixtures. The next gate must load real descriptor-backed
weights for at least the routed expert path before scaling layers.

Outcome: Complete for routed experts. `tools/ds4-v100-tp-ep-layer-smoke.cu`
now has a descriptor-backed expert mode that parses the production
`turbomind-pack-index.tsv`, loads layer-2 real packed expert weight/scale bytes,
and feeds descriptor-derived pointer tables into the TurboMind MXFP4 EP
kernels on all eight V100s. At `32` slots / `256K` / `top_k=6`, the run passes
with `192` aggregate routes, `641728512` descriptor bytes read,
`worst_ep_ms=0.246647`, `dense_kv_ms=1.121624`, `one_layer_ms=1.368271`,
KV `max_abs=0`, and deterministic finite repeat output. This is still not
serving and not logits-equivalent; dense/control/router/attention descriptor
execution is the next gate.

### Sprint 235 - Descriptor-Backed Full-Layer TP/EP Scaffold [complete]

Goal: Expand from descriptor-backed routed experts to a full layer-2 TP/EP
scaffold that parses, loads, and device-checks dense/control descriptors,
preserves sharded KV correctness, and runs descriptor-backed EP experts with
MTP off.

Rationale: TP is not operational until every layer family has a concrete
descriptor-backed runtime binding. Sprint 234 proved expert bytes; Sprint 235
must prove that the full-layer ownership model can bind real dense/control,
KV/state, and expert rows in the separate TP/EP codepath before replacing
checksum stages with true DS4 math and scaling to all 43 layers.

Outcome: Complete as a scaffold gate. `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
now parses the real TP/EP contract, binds all layer-2 descriptor families,
device-checks real dense/control bytes on the owning V100s, preserves sharded
KV correctness, and runs descriptor-backed TurboMind EP experts. At `32`
slots / `256K` / `top_k=6`, the run passes with `288` total layer rows,
`163102720` dense bytes checked, `84041408` control bytes checked,
`641728512` EP bytes loaded, KV `max_abs=0`, `worst_ep_ms=0.249378`, and
finite deterministic repeat output. This remains a scaffold, not a
logits-equivalent layer; the descriptor load/check time is startup evidence,
not serving throughput.

### Sprint 236 - Descriptor-Backed TP Dense Compute Gate [complete]

Goal: Replace one Sprint 235 dense checksum stage with real low-bit dense
computation for `blk.2.attn_q_a.weight`, using packed F8 source bytes from the
production pack and executing a TP8 row-sharded dense kernel on all V100s.

Rationale: The full-layer scaffold is not a logits-equivalent layer. The next
gate must prove that descriptor-backed packed dense bytes can feed GPU compute
inside the TP/EP path before expanding that pattern to the rest of attention
and shared dense math.

Outcome: Complete for one representative dense tensor. The TP/EP full-layer
smoke now resolves `blk.2.attn_q_a.weight`, loads real packed F8 E4M3 block-128
TP shards from the production pack, expands F8 values inside a CUDA kernel, and
computes `32` slots x `128` local rows x `4096` columns on all eight V100s.
The V100 run passes with `dense_compute_ms=0.081783`, exact repeat,
`dense_compute_oracle_max_abs=0.000000007`, KV `max_abs=0`, and the existing
descriptor-backed EP path still passing. This is not yet optimized HMMA/CUTLASS
dense math and not full-layer logits equivalence, but it proves the packed
dense compute path inside TP/EP.

### Sprint 237 - Layer-2 Dense Coverage Gate [complete]

Goal: Extend the Sprint 236 packed-F8 dense compute gate from one tensor to
all compatible layer-2 F8 dense TP tensor groups, with per-tensor timing,
repeat, and CPU oracle checks.

Rationale: Serving should not start from a path where only one dense tensor can
compute. The TP/EP layer needs broader dense-family coverage before full-layer
decode and serving gates are meaningful.

Outcome: Complete for layer-2 F8 dense tensors. The TP/EP full-layer smoke now
supports `--dense-compute-all-f8`, discovers all compatible layer-2 F8 dense TP
tensor groups, and executes all nine groups from packed production bytes. The
V100 run passes with `141606912` packed bytes loaded, worst dense compute time
`0.654029 ms`, exact repeat, worst CPU oracle error `0.000000015`, KV
`max_abs=0`, EP `worst_ep_ms=0.241766`, and final `PASS`. BF16 dense/control
math and real layer dataflow remain open.

### Sprint 238 - Layer-2 BF16 Dense Coverage Gate [complete]

Goal: Extend dense coverage to layer-2 BF16 compressor/indexer TP tensors,
expanding BF16 inside CUDA kernels and validating repeat plus CPU oracle checks
on all V100s.

Rationale: Sprint 237 covered F8 dense families. BF16 compressor/indexer
tensors are the remaining dense coverage gap before representative full-layer
dataflow can be composed.

Outcome: Complete for layer-2 BF16 dense tensors. The TP/EP full-layer smoke
now supports `--dense-compute-all-bf16` and combined `--dense-compute-all`.
It discovers all compatible layer-2 BF16 `dense_tp` groups, loads production
pack bytes, expands BF16 inside CUDA code, and validates repeat plus bounded
CPU oracle checks on the V100 pod. The BF16-only run covers five tensors with
`21495808` bytes loaded, worst BF16 compute time `0.047206 ms`, exact repeat,
and worst CPU oracle error `0.000000119`. The combined run preserves all nine
F8 dense checks with `dense_compute_pass=1`, reports `bf16_compute_pass=1`,
keeps KV `max_abs=0`, measures `worst_ep_ms=0.250368`, and ends in final
`PASS`. The next gap is no longer dense coverage; it is composing the real
layer dataflow into a next hidden state.

### Sprint 239 - Full-Layer TP/EP Decode [complete]

Goal: Combine descriptor-backed dense coverage, control/router handling,
sharded KV, and EP experts into a representative full layer that produces a
real next hidden state with MTP off.

Rationale: The current path proves bytes, KV, experts, and one dense compute
gate independently. Full-layer decode must connect those pieces into the layer
dataflow before serving.

Outcome: Complete for representative layer-2 next-hidden composition. The
TP/EP full-layer smoke now supports `--compose-next-hidden`, builds route-slot
mapping for the EP schedule, reduces TurboMind routed expert down outputs into
512-wide TP destination hidden shards, peer-copies those contributions across
all eight V100s, and composes resident next-hidden shards from
`blk.2.attn_output_b.weight`, `blk.2.ffn_down_shexp.weight`, returned EP
contributions, and deterministic residual input. The 32-slot/256K V100 run
passes with `ep_contribution_bytes=4194304`, `ep_return_bytes=4194304`,
`attn_dense_ms=0.555213`, `shared_dense_ms=0.153702`, `compose_ms=3.707477`,
checksum `4112649481`, `finite_bad=0`, exact repeat, and `compose_pass=1`.
The same run preserves combined F8/BF16 dense coverage, KV `max_abs=0`,
`worst_ep_ms=0.255590`, and final `PASS`. This is still not production
serving or logits equivalence, but it is the first resident TP/EP layer
composition gate.

### Sprint 240 - TP/EP Resident Decode Loop Gate [complete]

Goal: Convert the Sprint 239 one-shot TP/EP composition path into a resident
repeated decode-loop benchmark at `32` slots / `256K`, MTP off.

Rationale: Before server integration, the TP/EP path needs a benchmarkable
resident loop that avoids pack-byte reloads and per-step allocation.

Outcome: Complete for a representative layer-2 resident loop. The TP/EP
full-layer smoke now supports `--decode-steps N`, keeps the two F8 dense
composition tensors resident, keeps TurboMind EP weights and composition
buffers resident, and repeats EP+dense+peer-return+compose without rereading
pack bytes. The V100 pod run at `32` slots / `256K`, MTP off, `50` steps
passes with `ms_per_step=1.845548`, `slot_step_tok_s=17339.021356`,
`ep_ms_per_step=0.319095`, `dense_ms_per_step=0.756244`,
`compose_ms_per_step=0.770121`, checksum `2382924023`, `finite_bad=0`, and
`decode_pass=1`. Existing F8/BF16 dense coverage, KV check, and Sprint 239
composition still pass. This is not generated tok/s; it is the first resident
TP/EP layer-loop metric.

### Sprint 241 - TP/EP FP16 EP Return A/B [complete]

Goal: Add an opt-in FP16 EP return path and measure whether halving peer
payload improves the Sprint 240 resident loop.

Rationale: Sprint 240 showed compose/peer synchronization is a major stage
cost. FP16 return is the smallest isolated communication optimization.

Outcome: Complete and rejected as a default. `--ep-return-fp16` halves the
reported EP return payload from `4194304` bytes to `2097152` bytes and passes
finite/checksum validation, but it slows the 50-step resident loop from
`1.788149 ms/step` to `1.937399 ms/step`. Compose time rises from
`0.713836 ms/step` to `0.859697 ms/step`, so the added cast and expand kernels
cost more than the reduced peer payload saves. Keep FP32 return as default;
keep FP16 return as an opt-in diagnostic and revisit only if fused into the
EP reduction or next-hidden compose.

### Sprint 242 - TP/EP Fused Remote-Sum Compose [complete]

Goal: Fuse the FP32 EP remote contribution sum into next-hidden compose for
the separate TP/EP full-layer smoke.

Rationale: Sprint 241 showed standalone FP16 EP return is correct but slower.
The bottleneck is extra kernel/synchronization boundaries, not raw peer-copy
payload bytes.

Outcome: Complete. `--fuse-compose-sum` removes the destination `ep_sum` zero
kernel and eight add kernels per destination rank. Same-binary A/B at `32`
slots / `256K`, MTP off, and `50` resident steps: baseline FP32 return passes
at `1.784008 ms/step`, `17937.138290` slot-step tok/s, and
`0.713663 ms/step` compose; fused compose/sum passes with the same checksum at
`1.641832 ms/step`, `19490.418145` slot-step tok/s, and `0.568906 ms/step`
compose. Keep FP32 return and continue fusing TP/EP synchronization boundaries
before server integration.

## Experiment Backlog

These experiments should be run inside the TP/EP sprints, not as PP variants:

- TP8 collective roofline at `M=32/64/128`, hidden `4096`.
- TP8 dense GEMM fixture using FP16/FP8-style low-bit expansion on GPU.
- TP sharded KV allocation/update/read at `32` slots / `256K`, then `512K`
  if memory allows.
- EP routed expert smoke with real TurboMind/CUTLASS low-bit kernels at
  `32` active slots.
- Expert load-balance measurement: active experts, routes per expert, and
  worst-GPU imbalance.
- One-layer TP/EP correctness against frozen PP baseline.
- Full 43-layer TP/EP decode correctness.
- TP/EP serving throughput with generated and continuation tok/s separated.

## Parking Lot

- PP/layer-split scheduling optimizations: archived. Use only as baseline.
- Routed-only TP overlays inside the PP scheduler: rejected.
- Generic PP/TP scheduler abstraction: rejected.
- Single-slot throughput reports: rejected as practical-serving evidence.
- MTP serving: deferred until TP/EP serving is operational.
- PP-oriented MTP block-2 promotion: paused; useful correctness evidence only.

## Pivot Log

| Date | Change | Rationale | Next |
|---|---|---|---|
| 2026-05-23 | Archived the prior PP-era vision to `docs/sprints/archive/VISION-2026-05-23-pre-tp-hard-cut.md`. | The accumulated roadmap still documents history, but it no longer reflects the strategy. | Use this file as the active alignment document. |
| 2026-05-23 | Sprint 230 proved TP sharded KV row ownership at `32` slots / `256K`. | TP/EP needs resident hidden/KV state before EP expert work is meaningful. | Build the bounded EP routed-expert slice in separate TP/EP files. |
| 2026-05-23 | Sprint 231 proved bounded EP8 routed expert execution with real TurboMind MXFP4 kernels. | The EP low-bit kernel path is live outside the PP scheduler, but rank skew is visible. | Build the one-layer TP/EP correctness gate and preserve per-rank timing. |
| 2026-05-23 | Sprint 232 proved the combined TP runtime plus EP expert fixture in one process. | The TP/EP lifecycle works at the target shape, but it is still fixture data. | Move to descriptor-driven one-real-layer TP/EP correctness. |
| 2026-05-23 | Sprint 233 proved descriptor ownership for layer `2` from the real production-pack contract. | The contract has the rows and TP/EP ownership needed, but execution still uses fixture weights. | Bind descriptor rows to actual pack bytes and feed real expert pointers into the one-layer smoke. |
| 2026-05-23 | Sprint 234 proved descriptor-backed routed expert byte binding for layer `2`. | Real packed expert bytes now flow into the separate TP/EP path; the remaining gap is full-layer math and all-layer decode. | Build descriptor-backed full-layer TP/EP decode with MTP off. |
| 2026-05-23 | Sprint 235 proved a descriptor-backed full-layer scaffold for layer `2`. | All descriptor families now have a concrete TP/EP binding outside the PP path, but dense/control rows are checksum scaffolds, not math. | Replace dense/control checksum stages with real low-bit dense execution for representative full-layer decode. |
| 2026-05-23 | Sprint 236 proved real packed-F8 dense compute for `blk.2.attn_q_a.weight` in the TP/EP path. | The runtime can now compute from packed dense bytes, but only for one representative tensor and with a straightforward FP32 dot kernel. | Extend dense compute coverage or replace this gate with fused HMMA/CUTLASS dense blocks. |
| 2026-05-23 | Sprint 237 proved packed-F8 dense compute coverage for all compatible layer-2 F8 dense tensors. | F8 dense families execute from production bytes; BF16 compressor/indexer math and real layer dataflow remain. | Add BF16 compute coverage or compose dense outputs into representative full-layer decode. |
| 2026-05-23 | Sprint 238 proved BF16 compressor/indexer dense coverage and combined F8+BF16 coverage for layer `2`. | Layer-2 dense families now execute from production bytes in the separate TP/EP path. | Compose dense, KV, control/router, and EP expert outputs into representative full-layer decode. |
| 2026-05-23 | Sprint 239 proved representative TP/EP next-hidden shard composition for layer `2`. | Dense outputs, EP returned contributions, KV update/check, and residual composition now run in one separate TP/EP execution. | Move from smoke composition to a TP/EP serving gate at `32` slots / `256K`, MTP off. |
| 2026-05-23 | Sprint 240 proved a resident repeated TP/EP layer-loop benchmark at `32` slots / `256K`. | The path now reports stage costs without per-step pack reloads: dense and compose/sync dominate over EP. | Decide whether Sprint 241 optimizes dense/compose kernels first or starts server-loop integration with known bottlenecks. |
| 2026-05-23 | Sprint 241 proved FP16 EP return is correct but slower as a standalone pass. | Payload bytes are not the limiter; extra cast/expand kernels increase compose time. | Keep FP32 return default and target fused dense/compose kernel boundaries next. |
| 2026-05-23 | Sprint 242 proved fused FP32 remote-sum compose improves the resident layer loop. | Removing zero/add kernels is more valuable than standalone EP return quantization at this shape. | Continue collapsing TP/EP dense, EP return, and compose boundaries, then move to all-layer/server integration. |
| 2026-05-23 | Hard cut to TP/EP-only implementation work. | Sprint 225 showed the frozen PP path is correct but bottlenecked by layer-scheduled pipeline bubbles. User directed zero further PP variant work. | Sprint 226 starts the TP-only planner and topology contract. |
| 2026-05-23 | Deferred MTP until after TP/EP serving. | MTP can be useful only after the serving runtime has the right topology and multi-slot decode behavior. | Revisit after TP/EP serving exists and has multi-slot throughput evidence. |

## Open Questions

1. Does TP8 remain the primary target after the first collective and expert
   gates, or should TP4/EP8 be used as a temporary correctness stepping stone?
2. Should the first TP/EP pack preserve current TurboMind expert layout exactly
   or repack experts for EP ownership immediately?
3. What correctness tolerance is acceptable for TP/EP low-bit reductions versus
   the frozen PP baseline?
4. Should the first serving target be strictly `32` slots / `256K`, or should
   the TP runtime also gate `32` slots / `128K` as a faster iteration target?
