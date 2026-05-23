---
created: 2026-05-17
last_updated: 2026-05-23
last_updated_by: vision
revision: 277
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
- Sprint 243 tested a first HMMA dense replacement in the same TP/EP path. It
  is correct/finite but slower (`3.533215 ms/step`) than the scalar dense
  control (`1.620386 ms/step`), so naive per-tile F8 decode into WMMA
  fragments is rejected.
- Sprint 244 measured the tensor-core dense ceiling for the same path:
  resident FP16/cuBLAS dense reduces dense time from `0.755645 ms/step` to
  `0.175605 ms/step` and improves the representative layer-loop metric to
  `1.050770 ms/step` / `30453.870979` slot-step tok/s. This validates dense
  as the next kernel target, while keeping expanded FP16 as diagnostic only.
- Sprint 245 added real memory admission for turning that diagnostic into a
  runtime option. At `32` slots / `256K` / F8 KV, the TP/EP contract reports
  `27.024 GiB` base per GPU including reserve and `27.701 GiB` per GPU if
  cacheable dense source tensors are replaced by FP16 runtime weights, leaving
  `4.299 GiB` physical headroom. Dense FP16 cache is therefore admissible as a
  runtime fallback/ceiling path, not a source-format change.
- Sprint 246 turned that admission into a real V100 allocation/conversion
  smoke. The separate TP/EP dense-cache tool materializes all `4096` dense TP
  rows into FP16 arenas: `13.459473 GiB` aggregate cache, `1.682434 GiB` per
  GPU, zero nonfinite values, PASS. This is now an executable runtime cache
  path, though not yet wired into the all-layer decode loop.
- Sprint 247 wired dense cache lookup into the representative layer-2 TP/EP
  resident decode loop. Cache-backed FP16/cuBLAS dense passes at `1.015128`
  ms/step and `31523.122614` slot-step tok/s, preserving the private-FP16
  checksum while using cache pointers. The remaining gap is lifting this from
  two composition tensors to a descriptor-selected dense table for every
  layer.
- Sprint 248 added that descriptor-selected dense execution table. The
  all-layer dense-table gate runs `510` transformer-layer groups and `4080`
  cache-backed FP16/cuBLAS GEMMs per 32-slot iteration, passing at
  `51.003671 ms/iteration`, `7.738345` dense-table TFLOP/s, and zero
  nonfinite outputs. The remaining gap is composing dense, EP, KV, and
  hidden-state flow into a resident all-layer TP/EP loop.
- Sprint 249 made the representative TP/EP full-layer smoke layer-parametric.
  Layers `0`, `1`, `2`, `3`, and `42` pass at `32` slots / `256K` with
  cache-backed FP16 dense compose, real TurboMind MXFP4 EP experts, sharded KV,
  and fused next-hidden composition. The representative decode-loop proxy now
  spans SWA-only, ratio-4, ratio-128, and late-layer cases with `0.999333` to
  `1.181511 ms/step`. The remaining gap is a resident all-layer TP/EP loop
  that preserves hidden shards across all 43 layers in one process.
- Sprint 250 added a one-process all-layer scaffold gate. The TP/EP full-layer
  smoke now supports `--all-layers` and passes all `43` transformer layers at
  `32` slots / `256K`. The 10-step gate reports `45.356852 ms/token` summed
  decode proxy and `705.516343` projected slot-step tok/s, with stage sums
  `12.009343 ms` EP, `8.064360 ms` dense, and `25.277469 ms` compose. This is
  still a scaffold because per-layer runtime/cache state is rebuilt; the next
  gap is making the all-layer loop truly resident.
- Sprint 251 hoisted dense FP16 cache materialization out of the per-layer
  runner in `--all-layers` mode. The shared all-layer cache has `4096` dense
  rows and `14451998720` cache bytes, builds once in `7772.591153 ms`, and the
  10-step all-layer gate still passes `43/43` layers. Wall time improves from
  `91879.358460 ms` to `74382.064295 ms`, and projected slot-step tok/s moves
  from `705.516343` to `731.369579`. The next residency targets are
  TurboMind/API handles, route buffers, expert bindings, and TP runtime state.
- Sprint 252 added an opt-in descriptor-check bypass for serving-shaped TP/EP
  scaffold runs. With shared dense cache and `--skip-descriptor-checks`, the
  10-step all-layer gate passes `43/43` layers with `descriptor_checks=0`,
  wall time drops to `46990.435640 ms`, and the projected decode proxy remains
  in the same range at `720.987187` slot-step tok/s. Strict descriptor checks
  remain the default validation gate.
- Sprint 253 repaired the decode-only all-layer harness path. With shared
  dense cache, descriptor checks off, and no one-shot compose validation, the
  10-step all-layer gate passes `43/43` layers at `44.035733 ms/token`
  summed decode proxy and `726.682578` projected slot-step tok/s. Wall time
  drops to `39951.007721 ms`. This is now the lightweight TP/EP scaffold
  benchmark to use after strict validation.
- Sprint 254 added `--skip-predecode-probes` for benchmark-only runs after
  strict validation. The all-layer decode-only gate passes `43/43` layers with
  `descriptor_checks=0` and `predecode_probes=0`, reducing wall time to
  `37819.503379 ms`. The summed decode proxy remains in the scaffold band at
  `44.848746 ms/token` / `713.509362` projected slot-step tok/s.
- Sprint 255 hoisted TurboMind API lifecycle across the all-layer TP/EP loop.
  The gate now records `shared_api=1`, passes `43/43` layers at `32` slots /
  `256K`, and reduces wall time to `35565.756621 ms`. The summed decode proxy
  is `43.957040 ms/token` / `727.983506` projected slot-step tok/s.
- Sprint 256 hoisted fixed rank buffers, route maps, streams/events, and lazy
  compose buffers across the all-layer TP/EP loop. The gate now records
  `shared_rank_buffers=1`, passes `43/43` layers, and reduces wall time to
  `33978.379725 ms`. The summed decode proxy is `43.895297 ms/token` /
  `729.007483` projected slot-step tok/s.
- Sprint 257 hoisted the TP runtime/KV allocator across the all-layer TP/EP
  loop. The gate now records `shared_tp_runtime=1`, passes `43/43` layers, and
  reduces wall time to `28437.257957 ms`. The summed decode proxy regressed to
  `46.024692 ms/token` / `695.278962` projected slot-step tok/s, so this is
  correct residency progress but needs repeat timing before performance
  promotion.
- Sprint 258 repeated the shared TP runtime path with a 50-step all-layer
  gate. The regression persisted at `45.672166 ms/token` /
  `700.645557` projected slot-step tok/s, while checksum stayed fixed. Shared
  runtime is correct residency progress, but Sprint 256 remains the current
  decode-speed base.
- Sprint 259 added a same-binary TP runtime A/B. Local per-layer TP runtime is
  the current decode-speed base at `42.723359 ms/token` /
  `749.004771` projected slot-step tok/s. Shared TP runtime remains opt-in
  because it regresses decode to `681.247356` projected slot-step tok/s.
- Sprint 260 added resident all-layer TurboMind expert bindings. Active MXFP4
  expert bytes now stay in VRAM across the 43-layer scaffold
  (`3449290752` bytes/GPU). The 50-step gate passes `43/43` layers with
  checksum `204721433`, reduces wall time to `14338.419135 ms`, and reports
  `44.131138 ms/token` / `725.111599` projected slot-step tok/s.
- Sprint 261 added EP+dense overlap with a separate dense stream per rank.
  The same-binary 50-step gate passes `43/43` layers and checksum
  `204721433`; projected scaffold throughput improves from `631.273270` to
  `846.062424` slot-step tok/s. Compose/all-to-all is now the dominant
  remaining stage.
- Sprint 262 rechecked FP16 EP return under the resident overlapped schedule.
  It is still rejected: projected throughput regresses from `831.795688` to
  `729.339500` slot-step tok/s because compose time increases.
- Sprint 263 tested direct peer-memory compose. It is rejected: direct remote
  reads regress projected throughput from `840.751688` to `634.454351`
  slot-step tok/s because compose time increases. Keep staged peer copies.
- Sprint 264 changed staged peer-copy scheduling from destination streams to
  source copy streams. It is promoted: projected throughput improves from
  `840.494594` to `999.490407` slot-step tok/s with checksum preserved.
- Sprint 265 added the first token-major serving-order scaffold. It passes
  `172/172` layer invocations for `4` token steps at `32` slots / `256K`,
  reporting `48.840011 ms/token` proxy and `655.200508` projected slot-step
  tok/s. This is closer to serving order, but still not generated-token
  serving throughput.
- Sprint 266 tested all-layer shared dense op residency in token-major mode.
  It remains correct but is not promoted: the shared-op cache regressed the
  token-major proxy from `51.991980` to `56.085843 ms/token`. Keep it as an
  opt-in diagnostic and keep the default dense op lifecycle local per layer.
- Sprint 267 rechecked shared TP runtime in token-major order and promoted it
  for token-major all-layer runs. The 4-step scaffold improves from
  `51.289549` to `47.902324 ms/token` proxy and cuts wall time from
  `34880.753622` to `11661.323548 ms`, with checksum preserved.
- Sprint 268 made token-major runs advance logical position per token step.
  The 4-step scaffold over positions `1024-1027` passes `172/172` invocations
  at `45.770462 ms/token` proxy and `699.140856` projected slot-step tok/s.
- Sprint 269 ran longer continuous token-major gates. The 32-step run passes
  `1376/1376` layer invocations at `39.290219 ms/token` proxy and
  `814.452062` projected slot-step tok/s. Compose/all-to-all is now the
  dominant measured stage: `742.079181 ms` compose versus `514.766496 ms` EP.
- Sprint 270 skipped same-GPU compose copies on the FP32 EP-return path. The
  16-step A/B improves from `40.271428` to `38.503412 ms/token` proxy, and the
  new 32-step topline is `37.912062 ms/token` / `844.058544` projected
  slot-step tok/s.
- Sprint 271 split compose timing into reduce/copy/final buckets and showed
  copy dominates. Sprint 272 tested per-destination copy streams and improved
  the 32-step scaffold topline to `36.911097 ms/token` / `866.947964`
  projected slot-step tok/s.
- Steering update: stop spending the next work cycle on compose/kernel
  micro-optimization. Focus on making TP/EP operational end-to-end with
  generated and continuation tok/s, then return to kernel selection/fusion
  with serving data.
- Sprint 273 added the first serving-shaped TP/EP metric bridge. Decode-only
  rates are now visible: `875.486234` aggregate generated tok/s and
  `931.549518` aggregate continuation tok/s at `32` slots / `256K` /
  `16` generated tokens. Wall throughput is still only `10.6 tok/s` because
  the scaffold calls the heavy per-layer runner for every token/layer.
- Sprint 274 made the TP/EP serving loop resident enough for useful
  operational metrology. With shared dense ops, `32` slots / `256K` /
  `32` generated tokens/request reports `669.222644` wall generated tok/s and
  `690.469286` wall continuation tok/s.
- Sprint 275 wrapped that resident TP/EP backend in a repeatable sustained
  serving artifact harness. The current tool-level V100 result at `32` slots /
  `256K` / `32` generated tokens/request is `749.304439` wall generated tok/s,
  `774.209856` wall continuation tok/s, `963.264018` decode-only generated
  tok/s, and `1000.823072` decode-only continuation tok/s with `32/32` token
  match. This is not yet the HTTP appliance server.
- Sprint 276 added a TP/EP-only resident HTTP harness. It keeps the TP runtime,
  dense cache, shared dense ops, rank buffers, and expert bindings loaded
  across HTTP requests and exposes `/health`, `/v100/status`, `/metrics`, and
  `POST /v100/selected-token`. The first HTTP smoke reports `719.275018` wall
  generated tok/s and `751.645517` wall continuation tok/s at `32` slots /
  `256K` / `32` generated tokens/request. It is operational as a smoke-tested
  server path, but not yet wired into the production launcher/deployment.
- Sprint 277 wired that server into `tools/ds4-v100-run-appliance.sh` via
  `DS4_V100_SERVE_MODE=tp-ep`. The launcher smoke reports `728.744669` wall
  generated tok/s and `753.022651` wall continuation tok/s at the same
  `32` slot / `256K` / `32` token shape.
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

### Sprint 243 - TP/EP Dense HMMA Compose Gate [complete]

Goal: Test a bounded HMMA dense replacement for the two F8 composition tensors
used by the representative TP/EP resident loop.

Rationale: After Sprint 242, scalar F8 dense compute is the largest measured
stage. V100 should compute low-bit dense paths by expanding/dequantizing on GPU
into FP16 HMMA fragments, not by scalar FP32 dot products.

Outcome: Complete and rejected as a default. `--dense-hmma-compose` adds a
32-slot-capable WMMA/HMMA kernel that keeps F8 bytes resident and decodes each
tile into FP16 fragments before FP32 accumulation. It passes finite/repeat
checks, but it slows the fused-compose resident loop from `1.620386 ms/step`
and `19748.386791` slot-step tok/s to `3.533215 ms/step` and
`9056.907248` slot-step tok/s. Dense time rises from `0.753941 ms/step` to
`2.667910 ms/step`. Keep this as a diagnostic only; the next dense path should
reuse/adapt the older shape-specific F8 HMMA kernels or use a prepacked,
software-pipelined low-bit dense design.

### Sprint 244 - TP/EP Resident Dense Tensor-Core Ceiling [complete]

Goal: Measure the best-case dense-stage improvement when the two F8
composition tensors are expanded once into resident FP16 buffers and executed
with cuBLAS FP16 Tensor Core GEMM.

Rationale: Sprint 243 rejected the naive HMMA implementation, but did not
answer whether dense tensor-core execution is worth pursuing. A resident FP16
ceiling separates the value of the compute shape from the cost of low-bit
decode/layout feeding.

Outcome: Complete as a diagnostic ceiling. `--dense-f16-cublas-compose`
expands packed F8 to resident FP16 during setup for the two layer-2
composition tensors, converts resident activations to FP16, and uses
`cublasGemmEx` to produce FP32 output shards. Same-binary A/B at `32` slots /
`256K`, MTP off, fused compose enabled, and `50` resident steps: scalar dense
passes at `1.685018 ms/step`, `18990.892348` slot-step tok/s, and
`0.755645 ms/step` dense; resident FP16/cuBLAS passes at
`1.050770 ms/step`, `30453.870979` slot-step tok/s, and `0.175605 ms/step`
dense. This is a `1.60x` layer-loop improvement and a `4.30x` dense-stage
improvement. Keep the path diagnostic; build a packed low-bit dense production
kernel next.

### Sprint 245 - TP/EP Dense FP16 Cache Admission Gate [complete]

Goal: Decide whether the Sprint 244 resident FP16 dense ceiling can fit inside
the target `32` slot / `256K` TP/EP appliance memory budget.

Rationale: V100 cannot execute BF16/FP8/FP4 natively. The source model should
remain quantized, but a practical runtime can materialize selected dense
execution weights into FP16 if that materially improves tensor-core utilization
and still fits in VRAM.

Outcome: Complete. `tools/ds4-v100-tp-ep-pack-contract` now reports dense
FP16 runtime cache admission from real pack metadata. Against the production
pack at `32` slots / `256K` / F8 KV, base memory is `27.024 GiB` per GPU
including the `2.0 GiB` reserve. F8 dense packed bytes eligible for FP16 cache
are `0.687 GiB` per GPU, the FP16 cache is `1.364 GiB`, BF16 dense shadow is
`0.319 GiB`, and the practical replace-source total is `27.701 GiB` per GPU.
That leaves `4.299 GiB` physical headroom. Dense FP16 cache is memory
admissible as a runtime option; next implement the dense-cache loader/runtime
path for all dense tensors, then benchmark the resident all-layer path.

### Sprint 246 - TP/EP Dense FP16 Cache Runtime Smoke [complete]

Goal: Materialize the dense FP16 runtime cache on the V100 pod from the real
TP/EP contract.

Rationale: Sprint 245 proved the memory budget on paper. The next risk was
whether the runtime can allocate the arenas, stage packed source shards,
convert all dense F8/BF16 tensors on GPU, and keep the cache resident without
bad values.

Outcome: Complete. `tools/ds4-v100-tp-ep-dense-cache-smoke` is a new
TP/EP-only CUDA tool. It allocates one dense FP16 cache arena per GPU and
converts `f8_e4m3_b128` and `bf16` dense shards from the production pack into
that arena. Layer-2 passes with `112` dense rows and `0.281738 GiB` aggregate
cache. The full contract passes with `4096` dense rows, `8.047012 GiB`
aggregate source bytes, and `13.459473 GiB` aggregate FP16 cache. Per GPU:
`512` rows, `1.005877 GiB` source, `1.682434 GiB` FP16 cache, `126.250 MiB`
max temp staging, and zero nonfinite values. Next wire this arena into the
resident TP/EP layer execution path and benchmark all-layer decode.

### Sprint 247 - TP/EP Dense Cache Compose Integration [complete]

Goal: Wire the dense FP16 cache arena into the representative TP/EP resident
decode loop.

Rationale: Sprint 246 proved all dense rows can be cached, but execution still
used private FP16 copies for the two composition tensors. The runtime must
look up cache-resident weights by tensor and GPU if this is going to become a
serving path.

Outcome: Complete. `--dense-f16-cache-compose` builds a layer-local dense
cache from contract rows and makes the resident FP16/cuBLAS dense path use
cache pointers. Same-binary A/B/C at `32` slots / `256K`, MTP off, fused
compose, and `50` resident steps: scalar dense passes at `1.642514 ms/step`
and `19482.326340` slot-step tok/s; private FP16/cuBLAS passes at
`1.056807 ms/step` and `30279.894858`; cache-backed FP16/cuBLAS passes at
`1.015128 ms/step` and `31523.122614`. The cache-backed path emits
`dense_f16_cache=1`, preserves checksum `2515001`, and materializes `112`
layer-2 dense rows into `302514176` cache bytes. Next lift this into a
descriptor-selected dense execution table for every layer.

### Sprint 248 - TP/EP All-Layer Dense Execution Table [complete]

Goal: Build and validate a descriptor-selected dense execution table across
the transformer layers.

Rationale: The layer-2 cache-backed decode path still selected two dense
tensors by name. TP/EP serving needs the runtime to enumerate dense work from
the contract across all layers.

Outcome: Complete. `tools/ds4-v100-tp-ep-dense-cache-smoke` now supports
`--execute-table`, which groups complete `dense_tp` rows by `(layer,
tensor_id)` and runs cache-backed FP16/cuBLAS GEMMs for each group on all TP
ranks. The layer-2 gate passes with `14` groups, `112` GEMMs per iteration,
and `1.384323 ms/iteration`. The all-layer gate passes with `510`
transformer-layer groups, `4080` GEMMs per iteration, `394684006400` FLOPs
per iteration, `51.003671 ms/iteration`, `7.738345` dense-table TFLOP/s,
checksum `15841839914005485`, and zero nonfinite outputs. Next compose this
dense table with EP routed experts, KV/update, and hidden-state flow in a
resident all-layer TP/EP loop.

### Sprint 249 - TP/EP Layer-Parametric Resident Loop [complete]

Goal: Remove layer-2 hardcoding from the representative TP/EP full-layer smoke
and validate the DS4 layer families needed for an all-layer loop.

Rationale: Sprint 248 proved all-layer dense table enumeration, but the
resident decode loop still selected layer-2 composition tensors and ratio-4 KV
behavior. The next all-layer loop needs layer-local tensor names and the DS4
SWA/ratio-4/ratio-128 compression schedule to be correct before iterating all
43 layers.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now derives
composition tensors from `--layer N` and selects indexer KV only for ratio-4
layers. The V100 representative gate at `32` slots / `256K`, MTP off,
cache-backed FP16 dense compose, real TurboMind MXFP4 EP experts, and fused
compose passes layers `0`, `1`, `2`, `3`, and `42`. Decode-loop proxy timing
ranges from `0.999333` to `1.181511 ms/step`, or `27083.969701` to
`32021.345429` slot-step tok/s. The final scaffold accepts `comp_rows=0` only
for SWA-only layers and still requires compression rows for ratio-4/ratio-128
layers. Next build the resident all-layer TP/EP loop with hidden shards carried
through all layers in one process.

### Sprint 250 - TP/EP All-Layer Scaffold Gate [complete]

Goal: Add a single-process all-layer scaffold gate for the separate TP/EP path.

Rationale: Sprint 249 proved representative layer families, but the workflow
still required shell orchestration. Before server integration, the TP/EP path
needs one command that exercises all 43 transformer layers and reports an
aggregate decode proxy.

Outcome: Complete as a scaffold. `tools/ds4-v100-tp-ep-full-layer-smoke` now
supports `--all-layers`, emitting one `tp_ep_all_layer_item` row per layer and
a final `tp_ep_all_layer_scaffold` aggregate. On the V100 pod at `32` slots /
`256K`, MTP off, cache-backed FP16 dense compose, real TurboMind MXFP4 EP
experts, and fused compose, both all-layer gates pass `43/43` layers. The
10-step gate reports `45.356852 ms/token` summed decode proxy,
`705.516343` projected slot-step tok/s, `12.009343 ms` summed EP,
`8.064360 ms` summed dense, `25.277469 ms` summed compose, and checksum
`6174401222`. This remains scaffold evidence because runtime/cache/TurboMind
state is still recreated per layer inside the process. Next make the 43-layer
loop truly resident.

### Sprint 251 - TP/EP Shared Dense Cache Residency [complete]

Goal: Hoist dense FP16 cache materialization out of the per-layer all-layer
runner.

Rationale: Sprint 250's all-layer gate was one process, but not resident: each
layer rebuilt dense cache state. Dense cache is both large enough to matter and
already memory-admitted for `32` slots / `256K`, so it is the right first
state-hoist.

Outcome: Complete. In `--all-layers` mode, the full dense contract is parsed
once and materialized into a shared FP16 cache with `4096` rows and
`14451998720` cache bytes. The cache builds in `7772.591153 ms` and is reused
across all 43 layer scaffolds. The 10-step V100 gate passes `43/43` layers,
improves wall time from `91879.358460 ms` to `74382.064295 ms`, and improves
the summed decode proxy from `45.356852 ms/token` to `43.753529 ms/token`
(`731.369579` projected slot-step tok/s). Next hoist TurboMind/API handles,
route buffers, expert bindings, and TP runtime state.

### Sprint 252 - TP/EP Descriptor Check Bypass [complete]

Goal: Add an opt-in way to skip dense/control descriptor byte checks for
serving-shaped all-layer scaffold measurements.

Rationale: Descriptor byte checks are validation work, not serving work. After
the pack has passed strict descriptor validation, the all-layer loop should not
reread and checksum dense/control rows every layer.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now supports
`--skip-descriptor-checks`. The default remains strict. With shared dense
cache, `--compose-next-hidden`, and descriptor checks disabled, the 10-step
V100 gate passes `43/43` layers at `32` slots / `256K`, reports
`descriptor_checks=0`, cuts wall time from `74382.064295 ms` to
`46990.435640 ms`, and reports `44.383590 ms/token` summed decode proxy
(`720.987187` projected slot-step tok/s). A decode-only run exposed a smoke
harness `invalid resource handle` path; keep compose validation enabled until
that is fixed.

### Sprint 253 - TP/EP Decode-Only Harness Repair [complete]

Goal: Restore the decode-only all-layer scaffold benchmark.

Rationale: Sprint 252's descriptor-bypass path still needed
`--compose-next-hidden` enabled to avoid a harness failure. That extra one-shot
compose validation is not serving-shaped and should not be required for the
standard scaffold benchmark.

Outcome: Complete. `prepare_resident_f8_dense()` now drains stale per-device
CUDA error state before launching local dense setup conversion kernels. The
decode-only all-layer V100 gate passes `43/43` layers at `32` slots / `256K`,
shared dense cache, descriptor checks off, and MTP off. It reports
`44.035733 ms/token` summed decode proxy, `726.682578` projected slot-step
tok/s, `11.804094 ms` summed EP, `7.744769 ms` summed dense,
`24.482197 ms` summed compose, and `39951.007721 ms` wall time. Next hoist
TurboMind/API handles, route buffers, expert bindings, and stream/event
lifecycle across the 43-layer loop.

### Sprint 254 - TP/EP Pre-Decode Probe Bypass [complete]

Goal: Add an opt-in benchmark mode that skips pre-decode validation probes.

Rationale: After strict gates pass, the serving-shaped scaffold should not run
extra isolated TurboMind warmup/timing/repeat probes before each layer's decode
loop.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now supports
`--skip-predecode-probes`. The default strict behavior remains unchanged. With
shared dense cache, descriptor checks disabled, predecode probes disabled, and
decode-only all-layer mode, the V100 gate passes `43/43` layers at `32` slots /
`256K`. It reports `predecode_probes=0`, `44.848746 ms/token` summed decode
proxy, `713.509362` projected slot-step tok/s, and `37819.503379 ms` wall
time. Use this only as a lightweight benchmark mode after strict validation.

### Sprint 255 - TP/EP Shared TurboMind API [complete]

Goal: Hoist TurboMind dynamic library and API lifecycle across the all-layer
TP/EP scaffold.

Rationale: Sprint 254 removed benchmark-only probes, but each layer still
performed TurboMind `dlopen`, eight-device init, shutdown, and `dlclose`.
Serving should initialize that state once and reuse it across the decode loop.

Outcome: Complete. `--all-layers` now opens TurboMind once, initializes all
eight devices once, runs all 43 layers through the shared API handle, and
shuts down once. The single-layer path preserves local lifecycle for focused
diagnostics. With shared dense cache, descriptor checks disabled, predecode
probes disabled, and decode-only all-layer mode, the V100 gate passes `43/43`
layers at `32` slots / `256K`. It reports `shared_api=1`,
`43.957040 ms/token` summed decode proxy, `727.983506` projected slot-step
tok/s, and `35565.756621 ms` wall time. Next hoist route buffers,
streams/events, expert bindings, and TP runtime/KV state.

### Sprint 256 - TP/EP Shared Rank Buffers [complete]

Goal: Hoist fixed rank buffers and stream/event lifecycle across the all-layer
TP/EP scaffold.

Rationale: Route offsets, route-to-slot maps, input/gated/down buffers,
streams, events, and compose buffers are invariant for a fixed `slots/top_k`
run. Serving should not allocate and destroy them once per layer.

Outcome: Complete. `--all-layers` now initializes shared rank buffers once and
reuses them across all 43 layers. Per-layer packed expert bindings remain
layer-specific and are still loaded/freed per layer. With shared dense cache,
shared TurboMind API, descriptor checks disabled, predecode probes disabled,
and decode-only all-layer mode, the V100 gate passes `43/43` layers at `32`
slots / `256K`. It reports `shared_rank_buffers=1`, `43.895297 ms/token`
summed decode proxy, `729.007483` projected slot-step tok/s, and
`33978.379725 ms` wall time. Next hoist TP runtime/KV state or expert
descriptor bindings.

### Sprint 257 - TP/EP Shared TP Runtime [complete]

Goal: Hoist the TP runtime/KV allocator across the all-layer TP/EP scaffold.

Rationale: The 256K KV/compression/scratch arenas are serving state. Reopening
them once per layer is setup churn and obscures the cost of the resident
decode loop.

Outcome: Complete. `--all-layers` now opens the TP runtime once, allocates
sharded KV/compression/scratch arenas once, runs `dense_kv_slice()` per layer,
and closes the runtime once. With shared dense cache, shared TurboMind API,
shared rank buffers, descriptor checks disabled, predecode probes disabled,
and decode-only all-layer mode, the V100 gate passes `43/43` layers at `32`
slots / `256K`. It reports `shared_tp_runtime=1`, `46.024692 ms/token` summed
decode proxy, `695.278962` projected slot-step tok/s, and `28437.257957 ms`
wall time. The checksum matches prior gates, but decode timing regressed versus
Sprint 256; repeat before treating this as a performance promotion.

### Sprint 258 - TP/EP Shared Runtime Repeat Gate [complete]

Goal: Repeat the shared TP runtime path with a longer decode loop.

Rationale: Sprint 257 reduced wall time but regressed the decode proxy. A
longer gate is needed before deciding whether that regression is just short-run
noise.

Outcome: Complete. The 50-step all-layer gate passes `43/43` layers at `32`
slots / `256K` with `shared_tp_runtime=1` and checksum `204721433`. It reports
`45.672166 ms/token` summed decode proxy and `700.645557` projected slot-step
tok/s. This confirms the shared-runtime decode regression is persistent enough
to respect. Keep the shared runtime as correct residency work, but use Sprint
256 as the current decode-speed base unless the EP timing interaction is fixed.

### Sprint 259 - TP Runtime A/B Gate [complete]

Goal: Add a same-binary TP runtime sharing toggle and choose the current
decode-speed base.

Rationale: Shared TP runtime reduces setup wall time but appears to disturb
the decode proxy. A same-binary A/B avoids comparing across commits or cluster
conditions.

Outcome: Complete. The tool now supports `--share-tp-runtime` and
`--local-tp-runtime`, with local TP runtime as the default. The V100 50-step
A/B passes `43/43` layers and checksum `204721433` in both modes. Local
per-layer TP runtime reports `42.723359 ms/token` summed decode and
`749.004771` projected slot-step tok/s. Shared TP runtime reports
`46.972659 ms/token` and `681.247356` projected slot-step tok/s. Keep shared
runtime as an opt-in diagnostic; do not use it as the performance base until
the EP/dense timing interaction is fixed.

### Sprint 260 - TP/EP Resident Expert Bindings [complete]

Goal: Hoist active TurboMind expert bindings into an all-layer resident cache.

Rationale: A production appliance cannot reload expert weights for every layer.
Expert weights must be device resident, with only layer selection and execution
changing during decode.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now supports
`--shared-expert-bindings` and `--local-expert-bindings`; shared is the
default. The resident cache loads active gated and down MXFP4 expert bindings
for all 43 layers and all 8 GPUs, reporting `27594326016` aggregate bytes and
`3449290752` bytes/GPU. The V100 50-step A/B at `32` slots / `256K` passes
`43/43` layers and checksum `204721433`. Shared bindings reduce wall time from
`35770.339339 ms` to `14338.419135 ms`; decode proxy is `44.131138 ms/token`
and `725.111599` projected slot-step tok/s.

### Sprint 261 - TP/EP EP-Dense Overlap [complete]

Goal: Overlap routed EP work with dense tensor-core GEMMs inside the TP/EP
decode loop.

Rationale: EP and dense projections are independent until next-hidden compose.
Running them serially leaves available GPU work overlap on the table.

Outcome: Complete. Each rank now has a separate dense stream. Dense cuBLAS
GEMMs run on that stream, while routed EP stays on the existing rank stream.
The tool supports `--overlap-ep-dense` and `--serial-ep-dense`; overlap is now
the default. The V100 50-step A/B at `32` slots / `256K`, resident expert
bindings, and local TP runtime passes `43/43` layers with checksum
`204721433`. Projected scaffold throughput improves from `631.273270` to
`846.062424` slot-step tok/s. The next target is compose/all-to-all.

### Sprint 262 - TP/EP FP16 EP Return Recheck [complete]

Goal: Recheck FP16 EP return in the new resident, overlapped execution regime.

Rationale: Compose/all-to-all is now dominant, so reducing EP return payload
could have become valuable even though it was previously rejected.

Outcome: Complete. The V100 50-step A/B at `32` slots / `256K`, resident
expert bindings, local TP runtime, and EP+dense overlap passes `43/43` layers
with checksum `204721433` in both modes. FP32 return reports
`831.795688` projected slot-step tok/s; FP16 return reports `729.339500`.
FP16 return remains rejected because the cast/expand path increases compose
time from `25.608539 ms` to `31.200853 ms`.

### Sprint 263 - TP/EP Direct Remote Compose Probe [complete]

Goal: Test whether compose can skip staged peer copies and read EP
contributions directly from source GPUs over peer memory.

Rationale: The staged compose path performs explicit peer copies into
destination-local buffers, then launches the compose kernel. Direct remote
reads could remove that staging boundary if NVLink remote reads are fast enough.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now supports
`--direct-remote-compose` as an opt-in diagnostic. The V100 50-step A/B at
`32` slots / `256K`, resident expert bindings, local TP runtime, EP+dense
overlap, and FP32 EP return passes `43/43` layers with checksum `204721433` in
both modes. Staged compose reports `840.751688` projected slot-step tok/s;
direct remote compose reports `634.454351`. Direct remote compose is rejected
because remote reads increase compose time from `25.368965 ms` to
`37.776787 ms`.

### Sprint 264 - TP/EP Source-Scheduled Staged Copies [complete]

Goal: Improve the staged compose/all-to-all schedule without changing math.

Rationale: Direct remote reads lost to staged peer copies, but the staged path
still has scheduling freedom. Destination-scheduled copies may underuse source
copy engines.

Outcome: Complete. Each rank now owns a `copy_stream`. The tool supports
`--source-copy-schedule` and `--dest-copy-schedule`; source scheduling is now
the default. The V100 50-step A/B at `32` slots / `256K`, resident expert
bindings, local TP runtime, EP+dense overlap, FP32 EP return, and staged
compose passes `43/43` layers with checksum `204721433`. Projected scaffold
throughput improves from `840.494594` to `999.490407` slot-step tok/s, and
compose time drops from `25.452322 ms` to `19.513090 ms`.

### Sprint 265 - TP/EP Token-Major Scaffold [complete]

Goal: Add a serving-order TP/EP scaffold that executes layers in token-major
order.

Rationale: Layer-major repeated loops are useful for kernel timing, but serving
decodes as `for token -> for layer`. We need a gate that exposes that schedule
before claiming practical serving.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now supports
`--token-major-all-layers`. The V100 gate runs `4` token steps x `43` layers
at `32` slots / `256K`, using resident expert bindings, EP+dense overlap, and
source-scheduled staged copies. It passes `172/172` layer invocations and
reports `48.840011 ms/token` proxy / `655.200508` projected slot-step tok/s.
This is a serving-order scaffold, not generated-token serving throughput.

### Sprint 266 - TP/EP Shared Dense Ops Probe [complete]

Goal: Test whether token-major setup cost can be reduced by hoisting dense
operation objects across all layers.

Rationale: The token-major scaffold still constructs dense cuBLAS handles,
input buffers, and output buffers per layer invocation. If that setup is a
material part of the token-major gap, a shared dense-op cache should improve
the serving-order scaffold.

Outcome: Complete and rejected as a default. `tools/ds4-v100-tp-ep-full-layer-smoke`
now supports `--shared-dense-ops` as an opt-in diagnostic. On the V100 pod at
`32` slots / `256K`, `4` token steps, resident expert bindings, EP+dense
overlap, and source-scheduled staged copies, both local and shared dense-op
modes pass `172/172` layer invocations with checksum `296236348`. Local dense
ops report `51.991980 ms/token` proxy and `615.479538` projected slot-step
tok/s. Shared dense ops report `56.085843 ms/token` proxy and `570.553966`
projected slot-step tok/s. Shared dense ops slightly reduce wall time but
regress decode timing by `7.3%`, so the default remains local dense ops.

### Sprint 267 - TP/EP Token-Major Shared TP Runtime [complete]

Goal: Recheck shared TP runtime in token-major serving order and promote it
only if the serving-order proxy improves.

Rationale: Shared TP runtime was previously rejected in layer-major mode, but
token-major execution reuses KV/runtime state across token steps. That changes
the cost model enough to warrant a same-binary A/B before moving to generated
serving integration.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now defaults
token-major all-layer runs to shared TP runtime unless `--local-tp-runtime` is
explicitly requested. Layer-major defaults are unchanged. On the V100 pod at
`32` slots / `256K`, `4` token steps, resident expert bindings, EP+dense
overlap, source-scheduled staged copies, and local dense ops, shared TP runtime
improves the token-major proxy from `51.289549` to `47.902324 ms/token` and
projected slot-step throughput from `623.908781` to `668.026047 tok/s`.
Wall time drops from `34880.753622` to `11661.323548 ms`, with checksum
`296236348` preserved. A default one-step check confirms token-major runs now
select `shared_tp_runtime=1`.

### Sprint 268 - TP/EP Token-Major Position Advance [complete]

Goal: Make the token-major scaffold advance context position across token
steps.

Rationale: The first token-major scaffold validated execution order, but every
token step reused the same logical position. Serving decode advances position
each token while keeping the sequence slot fixed, so the scaffold should do
the same before longer continuous gates or generated-token integration.

Outcome: Complete. In `--token-major-all-layers` mode, each layer invocation
now uses `position = start_position + token_step`, and token-major item logs
include the effective position. On the V100 pod at `32` slots / `256K`, `4`
token steps, positions `1024-1027`, shared TP runtime, resident expert
bindings, EP+dense overlap, and source-scheduled staged copies, the scaffold
passes `172/172` layer invocations. It reports `45.770462 ms/token` proxy,
`699.140856` projected slot-step tok/s, `93.872406 ms` summed EP,
`89.157724 ms` summed compose, `11799.119372 ms` wall, and checksum
`296236348`.

### Sprint 269 - TP/EP Continuous Token-Major Gate [complete]

Goal: Run longer token-major gates to reduce early-token noise and expose the
steady scaffold bottleneck.

Rationale: Four token steps are useful for iteration but still include startup
effects. Before bridging to generated serving, the scaffold needs a longer
continuous run at the target `32` slots / `256K` shape.

Outcome: Complete. On the V100 pod, the 16-step and 32-step token-major gates
both pass. The 32-step run covers `1376` layer invocations with shared TP
runtime, resident expert bindings, EP+dense overlap, source-scheduled staged
copies, local dense ops, and advancing positions from `4096`. It reports
`39.290219 ms/token` proxy, `814.452062` projected slot-step tok/s,
`514.766496 ms` summed EP, `742.079181 ms` summed compose, `91515.672970 ms`
wall, and checksum `8297177632`. The bottleneck is now clearly the
compose/all-to-all boundary plus remaining orchestration, not the routed EP
kernel in isolation.

### Sprint 270 - TP/EP Skip Self Compose Copy [complete]

Goal: Remove same-GPU staged compose copies from the FP32 EP-return path.

Rationale: Sprint 269 showed compose/all-to-all dominates the continuous
token-major scaffold. The staged path still copied `src == dst` shards even
though each destination GPU can read its local EP contribution directly.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now supports
`--skip-self-compose-copy` and `--copy-self-compose`; skip-self is the default.
On the FP32 return path, same-GPU copy traffic is skipped and compose reads the
local `d_ep_contrib_all` slice for that source. The V100 16-step A/B at `32`
slots / `256K` passes with checksum `8244145680` in both modes and improves
from `40.271428` to `38.503412 ms/token` proxy. Compose time drops from
`371.558564` to `342.417467 ms`. The 32-step skip-self run passes
`1376/1376` invocations at `37.912062 ms/token` proxy, `844.058544` projected
slot-step tok/s, `522.914003 ms` EP, `689.877521 ms` compose, and checksum
`8297177632`.

### Sprint 271 - TP/EP Compose Stage Breakdown [complete]

Goal: Split token-major compose timing into actionable buckets.

Outcome: Complete. The tool now reports compose reduce, copy, and final
compose timing. At `32` slots / `256K`, `16` token steps, the passing run
reports `327.657087 ms` compose total: `49.805028 ms` reduce,
`242.803068 ms` copy, and `35.048991 ms` final compose. Copy/all-to-all is
the dominant part of compose.

### Sprint 272 - TP/EP Multi Copy Streams Probe [complete]

Goal: Test whether source-scheduled peer copies benefit from multiple copy
streams per source rank.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now supports
`--multi-copy-streams`. The 16-step A/B at `32` slots / `256K` improves from
`39.288036` to `37.395624 ms/token` proxy and reduces copy time from
`248.331836` to `219.221398 ms`. The 32-step opt-in run passes `1376/1376`
invocations at `36.911097 ms/token` proxy and `866.947964` projected
slot-step tok/s. Per steering, the next sprint pivots to end-to-end TP/EP
serving rather than continuing compose micro-optimization.

### Sprint 273 - TP/EP Serving Metric Bridge [complete]

Goal: Expose generated-token and continuation-token metrics from the resident
token-major TP/EP path.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now supports
`--serving-bench`, emitting generated/continuation token counts and tok/s
rates. At `32` slots / `256K`, `16` generated tokens/request, shared TP
runtime, resident expert bindings, source-scheduled multi-copy compose, and
MTP off, the V100 run passes with checksum `8244145680`. Decode-only metrics
are `875.486234` aggregate generated tok/s and `931.549518` aggregate
continuation tok/s. Wall metrics are only `10.612319` generated tok/s and
`10.616412` continuation tok/s because the token-major scaffold still invokes
the heavy per-layer `run_layer()` path for every token/layer. Next build a
resident serving loop that calls the decode body directly without per-layer
scaffold setup.

### Sprint 274 - TP/EP Resident Serving Loop [complete]

Goal: Remove the per-token/per-layer `run_layer()` scaffold from TP/EP
serving-bench mode.

Outcome: Complete. `--serving-bench` now uses a direct resident decode loop
when shared TP runtime, resident expert bindings, shared rank buffers, and the
shared dense cache are available. It parses layer contracts once, binds
resident expert/dense state, skips serving-mode checksum readback, and calls
the decode body directly. At `32` slots / `256K`, shared dense ops are required
for wall throughput. The best V100 run so far uses `32` generated
tokens/request and reports `669.222644` wall generated tok/s,
`690.469286` wall continuation tok/s, `876.524260` decode generated tok/s,
and `910.270244` decode continuation tok/s. Next wrap this backend in the
HTTP sustained-decode harness.

### Sprint 275 - TP/EP Sustained Serving Artifact Wrapper [complete]

Goal: Produce repeatable sustained-serving artifacts from the resident TP/EP
backend before wiring the backend into the HTTP appliance server.

Outcome: Complete. `tools/ds4-v100-tp-ep-sustained-bench.sh` runs the
resident TP/EP serving bench with the promoted `32` slot / `256K` settings,
records stdout/stderr, and writes `sustained_decode.tsv`,
`sustained_decode.json`, and per-case `result.json` artifacts. The V100 pod
run at `32` slots / `256K` / `32` generated tokens per request passes with
`32/32` token match. The current artifact topline is `749.304439` wall
generated tok/s, `774.209856` wall continuation tok/s, `963.264018`
decode-only generated tok/s, and `1000.823072` decode-only continuation tok/s.
This confirms the resident backend can be measured repeatably, but it still
needs the operational HTTP harness.

### Sprint 276 - TP/EP Resident HTTP Harness [complete]

Goal: Expose the resident TP/EP backend through an in-process HTTP harness.

Outcome: Complete as a smoke-tested server path. The TP/EP full-layer tool now
has `--serve-http`, keeps the resident backend loaded across requests, and
serves `GET /health`, `GET /v100/status`, `GET /metrics`, and
`POST /v100/selected-token`. The V100 HTTP smoke used four requests against
one resident server and the generation POST returned `32/32` token match,
`719.275018` wall generated tok/s, `751.645517` wall continuation tok/s,
`926.497242` decode-only generated tok/s, and `974.020201` decode-only
continuation tok/s. Requests are currently serialized and the harness is not
yet wired into the deployment launcher.

### Sprint 277 - TP/EP Appliance Launcher Path [complete]

Goal: Start the TP/EP resident HTTP server through the appliance launcher.

Outcome: Complete. `tools/ds4-v100-run-appliance.sh` now supports
`DS4_V100_SERVE_MODE=tp-ep`, resolves the promoted TP/EP server command, and
fails closed outside the current target shape. The V100 launcher smoke used
the launcher to start the resident TP/EP server, then exercised `/health`,
`/v100/status`, `POST /v100/selected-token`, and `/metrics`. The POST returned
`32/32` token match, `728.744669` wall generated tok/s, `753.022651` wall
continuation tok/s, `939.787471` decode-only generated tok/s, and
`976.290858` decode-only continuation tok/s.

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
| 2026-05-23 | Sprint 243 rejected the first naive TP/EP dense HMMA candidate. | HMMA is not enough by itself; per-tile F8 decode/staging made dense time worse than scalar. | Adapt the older shape-specific HMMA kernels or design a prepacked/software-pipelined dense path. |
| 2026-05-23 | Sprint 244 proved a resident FP16 tensor-core dense ceiling is materially faster. | Dense is removable if low-bit feeding is efficient, but expanded FP16 is not the final memory format. | Implement a packed low-bit dense production kernel that approaches the FP16/cuBLAS ceiling. |
| 2026-05-23 | Sprint 245 proved dense FP16 runtime cache fits the `32` slot / `256K` TP/EP budget when replacing dense source tensors in VRAM. | This gives us a working tensor-core dense fallback while preserving the quantized source pack offline. | Build the TP/EP dense-cache loader/runtime path for all dense tensors and benchmark resident all-layer decode. |
| 2026-05-23 | Sprint 246 materialized all dense TP rows into FP16 cache arenas on the V100 pod. | The dense-cache path is now an executable runtime primitive, not just an estimate. | Wire dense cache lookup into resident layer execution and benchmark all-layer decode. |
| 2026-05-23 | Sprint 247 wired dense cache lookup into the representative TP/EP decode loop. | Execution can now consume cache-resident FP16 dense weights instead of private per-op copies. | Build a descriptor-selected dense execution table across all layers. |
| 2026-05-23 | Sprint 248 built the descriptor-selected all-layer dense execution table. | Dense no longer depends on hardcoded layer-2 tensor selection. | Compose dense, EP, KV, and hidden-state flow in a resident all-layer TP/EP loop. |
| 2026-05-23 | Sprint 249 made the representative TP/EP full-layer smoke layer-parametric across SWA-only, ratio-4, ratio-128, and late layers. | The all-layer loop no longer has layer-2 tensor-name and ratio-4 KV assumptions as blockers. | Build a resident all-layer TP/EP loop that carries hidden shards through all 43 layers in one process. |
| 2026-05-23 | Sprint 250 added a single-process all-layer TP/EP scaffold gate. | The TP/EP path now has a 43-layer correctness/timing gate, but it still recreates per-layer state. | Move runtime/cache/TurboMind state outside the per-layer runner for a truly resident all-layer loop. |
| 2026-05-23 | Sprint 251 hoisted the dense FP16 cache across all layers. | Reusing dense cache cuts all-layer scaffold wall time by about 19% and removes one class of per-layer state churn. | Hoist TurboMind/API, route buffers, expert bindings, and TP runtime state. |
| 2026-05-23 | Sprint 252 added opt-in descriptor-check bypass for serving-shaped scaffold runs. | Descriptor checks are validation work; skipping them cuts all-layer wall time by about 37% after validation has passed. | Fix decode-only harness and hoist TurboMind/API plus rank buffers. |
| 2026-05-23 | Sprint 253 repaired the decode-only all-layer scaffold harness. | The standard TP/EP scaffold benchmark no longer requires an extra one-shot compose validation path. | Hoist TurboMind/API handles, route buffers, expert bindings, and stream/event lifecycle. |
| 2026-05-23 | Sprint 254 added opt-in pre-decode probe bypass for benchmark runs. | Extra isolated TurboMind probes are validation work, not serving work. | Hoist TurboMind/API handles, route buffers, expert bindings, and stream/event lifecycle. |
| 2026-05-23 | Sprint 255 hoisted TurboMind API lifecycle across the all-layer TP/EP loop. | Removing per-layer library/API setup cuts scaffold wall time while preserving decode checksums. | Hoist route buffers, streams/events, expert bindings, and TP runtime/KV state. |
| 2026-05-23 | Sprint 256 hoisted fixed rank buffers and stream/event lifecycle across the all-layer TP/EP loop. | Removing per-layer route/core buffer allocation cuts wall time and keeps checksum stable. | Hoist TP runtime/KV state or expert descriptor bindings. |
| 2026-05-23 | Sprint 257 hoisted TP runtime/KV allocation across the all-layer TP/EP loop. | Correctness holds and wall time drops, but decode proxy regresses and needs repeat timing. | Repeat/longer gate, then decide whether to keep shared TP runtime as the performance base before expert binding hoist. |
| 2026-05-23 | Sprint 258 repeated the shared TP runtime path with a 50-step all-layer gate. | The decode regression persisted while checksum stayed stable. | Investigate EP timing under shared runtime, or keep Sprint 256 as decode-speed base while hoisting expert bindings. |
| 2026-05-23 | Sprint 259 added a same-binary TP runtime A/B and made local TP runtime the default. | Shared TP runtime is correct but slower for decode in the same executable. | Hoist expert descriptor bindings or collapse EP/dense/compose boundaries while preserving the local-runtime performance base. |
| 2026-05-23 | Sprint 260 hoisted active TurboMind expert bindings into a resident all-layer cache. | This matches the production appliance requirement and removes per-layer expert reload churn. | Move toward a real serving loop or reduce the EP/dense/compose boundary now that major setup state is resident. |
| 2026-05-23 | Sprint 261 overlapped routed EP with dense cuBLAS work on separate streams. | EP and dense are independent until compose, and overlap produced a 34% scaffold throughput gain. | Optimize compose/all-to-all or convert the scaffold into a serving loop. |
| 2026-05-23 | Sprint 262 rechecked FP16 EP return under the resident overlapped schedule. | FP16 return still regresses total decode because compose gets slower. | Keep FP32 return and target fused/direct compose-all-to-all instead of standalone cast staging. |
| 2026-05-23 | Sprint 263 tested direct peer-memory compose. | Direct remote reads preserve correctness but regress compose time and total throughput. | Keep staged peer copies; optimize staged-copy scheduling or destination-side reduction. |
| 2026-05-23 | Sprint 264 changed staged peer-copy scheduling to source copy streams. | Source-scheduled copies materially reduce compose time and raise projected scaffold throughput. | Convert scaffold into serving loop or continue destination-side compose kernel optimization. |
| 2026-05-23 | Sprint 265 added a token-major serving-order scaffold. | It exposes the real decode order and shows the next gap is resident token-loop state, not only layer-major kernel speed. | Reduce token-major setup/wall cost and then integrate generated/continuation serving measurement. |
| 2026-05-23 | Sprint 266 tested shared dense-op residency in token-major order. | Correctness holds, but decode proxy regresses despite slightly lower wall time. | Keep dense ops local per layer and target TP runtime/KV orchestration or serving integration next. |
| 2026-05-23 | Sprint 267 promoted shared TP runtime for token-major all-layer runs. | In serving order, TP/KV runtime residency improves both wall/setup and summed decode proxy. | Reduce token-major compose/all-to-all and bridge the scaffold into generated/continuation serving measurement. |
| 2026-05-23 | Sprint 268 added token-major position advance. | The scaffold now progresses logical context position across token steps and remains correct. | Run a longer continuous token-major gate, then bridge to generated/continuation serving measurement. |
| 2026-05-23 | Sprint 269 established the longer continuous token-major scaffold baseline. | At 32 steps the path reaches `814.452062` projected slot-step tok/s and compose dominates EP. | Collapse compose/all-to-all or bridge to generated/continuation serving measurement. |
| 2026-05-23 | Sprint 270 removed same-GPU staged compose copies. | Self-copy traffic was a measurable part of compose cost, but compose remains dominant after removal. | Target destination-side reduction/synchronization or bridge to generated/continuation serving measurement. |
| 2026-05-23 | Sprint 271 split compose timing and Sprint 272 tested multi-copy streams. | Copy/all-to-all dominates compose, and per-destination copy streams improve the scaffold. | Pivot to TP/EP generated/continuation serving before more kernel micro-optimization. |
| 2026-05-23 | Sprint 273 added serving-shaped TP/EP metrics. | Decode-only TP/EP rates are promising, but scaffold wall overhead prevents operational serving. | Build a resident serving loop without per-token/per-layer `run_layer()` setup. |
| 2026-05-23 | Sprint 274 built the resident TP/EP serving loop. | Shared dense ops plus direct decode remove the scaffold wall bottleneck and produce useful serving-shaped wall tok/s. | Integrate the resident TP/EP backend with the HTTP sustained-decode harness. |
| 2026-05-23 | Sprint 275 added a sustained-serving artifact wrapper over the resident TP/EP backend. | We need repeatable serving-shaped metrology before and during HTTP harness integration. | Wire the resident backend into the operational HTTP sustained-decode path. |
| 2026-05-23 | Sprint 276 added a TP/EP-only resident HTTP harness. | The backend now stays loaded across HTTP health/status/metrics/generation requests. | Wire this server mode into the appliance launcher and run sustained HTTP matrices. |
| 2026-05-23 | Sprint 277 wired the TP/EP HTTP server into the appliance launcher. | Operators can now start the TP/EP path with `DS4_V100_SERVE_MODE=tp-ep`. | Build and run sustained HTTP matrix tooling against the launcher path. |
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
