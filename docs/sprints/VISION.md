---
created: 2026-05-17
last_updated: 2026-05-18
last_updated_by: sprint-execute
revision: 40
---

# Vision: DS4 V100 Appliance

## North Star

Build a DeepSeek V4 Flash appliance for the 8x V100-SXM2-32GB cluster that runs
the high-intelligence source quantized model from pure device-resident packs by
default, preserves model quality, and reaches a verified deployed serving path
before broad throughput tuning.

The sprint sequence should keep format, topology, sharding, and scheduling
decisions explicit. The project is not trying to become a generic GGUF runner;
it is a narrow DS4 runtime tuned for this hardware.

## Current State

- Sprints 001-004 proved the memory and pack-residency foundation: the source
  model is inventoried, manifested, packed into per-GPU shards, reconciled, and
  uploaded to all 8 V100s as CUDA device memory.
- The source model generation guard is still active by design for normal
  serving. Sprint 007 added only a bounded CPU diagnostic oracle path for
  source-layout correctness evidence.
- Sprint 005 proved a first resident source-dtype diagnostic: native BF16
  `token_embd.weight` bytes can be gathered from V100 device memory and expanded
  to F32 bit-exactly. This is not native BF16 math; V100 production GEMMs must
  target FP16 tensor cores or the selected low-bit/integer kernels.
- The tightest observed residency case still leaves more than the planned 3 GiB
  reserve on a 32 GB V100. Weight VRAM fit is no longer the primary blocker.
- The main remaining risk is production numerical correctness and kernel
  coverage for the mixed BF16/F32/F8_E4M3_B128/MXFP4 source layout on V100,
  especially attention, compressed KV, routing, and routed expert execution.
- Sprint 006 has shipped that context/skeleton contract. The project now has a
  verified 8-GPU V100 topology check, descriptor policy, HC relay smoke, and
  no-math layer walk over the real pack index, while source-layout generation
  remains guarded.
- Sprint 007 shipped a guarded CPU-only source-layout oracle. The official
  `short_reasoning_plain` fixture now selects the expected first token exactly
  on the cluster. The sprint corrected MXFP4 row layout to match GGML's
  `block_mxfp4` low-half/high-half nibble ordering and reset source-layout KV
  correctness to the default F16 cache contract.
- Sprint 008 shipped the bridge from CPU source oracle to first V100 source
  anchors: automated official-vector validation, source-layout guard checks,
  exact F16 KV admission by stage/context/slot, MXFP4 parity hardening, and a
  bounded CUDA F8_E4M3_B128 source-format row-decode probe on `sm_70`. Full
  V100 source-layout prefill execution is now the next runtime sprint.
- Sprint 009 shipped the first bounded V100 prefill/KV execution surface:
  deterministic F16 KV arena planning/allocation, V100 context allocation for
  256K and 1M single-slot tiers, guarded source-layout validation on the real
  model, and a CUDA `sm_70` diagnostic smoke that bridges F8 source rows into
  raw SWA, compressed KV, ratio-4 indexer KV, and compression-state surfaces.
- Sprint 010 shipped the stage-owned KV integration gate: deterministic
  per-layer KV/state subviews inside each GPU's `kv_arena`, V100 diagnostic
  writes through those subviews for ratio-4 and ratio-128 layers, real
  compressor recurrence smokes for attention and indexer-shaped paths, CPU
  references, and real-model guard validation. It did not ship dense
  projection, MoE, output-head logits, selected-token decode, or serving.
- Sprint 011 shipped the bounded source projection/attention gate: device
  source-F8 projection diagnostics from resident arenas, executable BF16/F32
  V100 policy checks, projection-fed ratio-128 and ratio-4
  attention/compressor smokes, and device-resident writes into stage-owned KV
  views. It did not ship full layer output, MoE, output-head logits,
  selected-token decode, or serving.
- Sprint 012 shipped the bounded source-BF16 output-head/logits gate and a
  runnable V100 appliance gate. The gate passes real-model source guards and
  all implemented CUDA smokes on the 8x V100 pod, but reports `ready=false`
  because full layer/MoE execution, full selected-token decode, public serving,
  MTP, and throughput benchmarks remain missing.
- Sprint 013 shipped the first source-MXFP4 routed expert execution surface and
  a bounded MoE/logits selected-token smoke. The gate now validates router
  selection, MXFP4 gate/up/down expert matmuls, SwiGLU accumulation, BF16
  output-head logits, and selected-token comparison on V100. It still reports
  `ready=false` because the path is synthetic and not yet bound to real
  pack-index layer descriptors or the full layer scheduler.
- Sprint 014 shipped the real pack-index layer descriptor gate. The appliance
  gate now validates layer-2 attention, compressor/indexer, router,
  routed/shared expert, HC control, and output-head descriptors from the real
  pack index on the 8x V100 pod. It still reports `ready=false` because
  descriptors are not yet materialized as runtime bindings and no real
  descriptor-bound layer compute has shipped.
- Sprint 015 shipped runtime tensor bindings and the first descriptor-bound
  real-byte FFN compute gate. The V100 pod now runs layer-2 routed MXFP4 plus
  shared F8 FFN bytes from the source GGUF at real pack offsets and compares
  the output against CPU source-format references. The gate still reports
  `ready=false` because real router scheduling, full attention/residual/norm
  layer execution, selected-token decode, serving, MTP, and throughput remain
  incomplete.
- Sprint 016 shipped descriptor-bound real router scheduling for the layer-2
  FFN slice. The V100 pod now computes router logits from real
  `ffn_gate_inp.weight` bytes, selects experts through the real
  `ffn_gate_tid2eid` hash table, executes all six selected MXFP4 routed experts
  plus the shared F8 expert, and compares the result against CPU source-format
  references. The gate still reports `ready=false` because scheduler-owned full
  layer execution, attention/residual/norm integration, real-model selected
  token decode, serving, MTP, and throughput remain incomplete.
- Sprint 017 shipped the scheduler-owned layer-state surface for the
  descriptor-bound router/FFN slice. The state binds real layer descriptors
  once, validates dimensions and router kind, carries layer/stage/KV metadata,
  exposes source row views, constructs selected routed expert matrices, and
  sizes the FFN arena span. The gate now includes `layer_state` and still
  reports `ready=false` because full layer output, real selected-token decode,
  serving, MTP, and throughput remain incomplete.
- Sprint 018 shipped the first descriptor-bound attention projection,
  residual, and norm gate from real source bytes. The layer state now owns
  attention FP8/control descriptors and the V100 pod validates real layer-2
  q/kv/output projection surfaces, residual add, and FFN pre-norm against CPU
  source-format references. This still is not full attention semantics: softmax
  over raw/compressed KV, semantic layer output, real selected-token decode,
  serving, MTP, and throughput remain incomplete.
- Sprint 019 shipped the first reusable hidden-vector layer execution surface.
  The V100 pod now validates layer-2 semantic attention over explicit raw plus
  compressed KV rows with sinks, grouped F8 attention output, residual,
  FFN pre-norm, real hash-router selected MXFP4 experts, shared F8 expert, and
  final next-hidden residual. The gate now includes `integrated_layer` and
  reports `ready=false` because full HC pre/post scheduling, real compressor/
  indexer descriptor binding, full 43-layer selected-token decode, serving,
  MTP, and throughput remain incomplete.
- Sprint 020 extended the runtime layer bridge with real compressor/indexer
  descriptor ownership and an executable DS4 HC-state layer entrypoint. The
  V100 pod now validates layer-2 `[4 x 4096]` HC attention pre/post and FFN
  pre/post around the hidden-vector body, and the full 8-GPU gate passes with
  `ready=false`. The remaining critical gap is executor-owned compressed-row
  generation and indexed ratio-4 compressed attention.
- Sprint 021 shipped executor-owned compressed-row generation for the
  representative ratio-4 layer. The V100 pod now validates mutable raw KV,
  attention compressor recurrence, emitted attention compressed rows, ratio-4
  indexer recurrence, emitted indexer rows, forced top-k visibility, indexed
  mixed attention, the existing HC layer entrypoint, and the full appliance
  gate. The next critical gap is the full 43-layer single-slot scheduler that
  produces a real selected token.
- Sprint 022 shipped the first resident multi-layer scheduler surface. The
  executor now supports both hash and bias router layers, the V100 gate
  validates a real ratio-128 bias-router layer, and the stage scheduler uploads
  the complete gpu0 shard and executes layers 0-5 from a token embedding seed.
  The next critical gap is cross-GPU HC relay through stages 1-7 and the final
  output-head selected-token gate.
- Sprint 023 shipped the first cross-GPU scheduler handoff. The V100 pod now
  executes layers 0-5 on gpu0, peer-copies HC, and executes layers 6-11 on
  gpu1 with resident arenas. It also fixed CUDA model-range caching so cached
  control tensors are device-local instead of being reused across GPUs. The
  next critical gap is extending the stage chain through gpu7, then attaching
  output-head selected-token validation.
- Sprint 024 shipped the full 8-stage scheduler chain. The V100 pod now
  executes all 43 layers across gpu0-gpu7 with resident arenas and peer HC
  handoffs, producing a finite nonzero final HC state on gpu7. The full gate no
  longer lists `full_43_layer_scheduler`; the next critical gap is collapsing
  final HC through the output head and comparing a selected token against the
  source oracle.
- Sprint 025 extended the scheduler with gpu7 output-head selected-token
  execution. The V100 pod can now replay the short official prompt through all
  43 layers, run HC-head collapse, output norm, BF16 output projection, and
  select top-1. The explicit oracle check fails today: expected token bytes
  `3136`, got `0a0a` at token id 271. The next critical gap is localizing the
  numerical divergence across the 43-layer body.
- Sprint 026 localized the first selected-token failure away from the
  output-head adapter. A deterministic HC parity smoke matches CPU and V100
  output-head top-5 on gpu7, while the prompt replay top-k remains dominated
  by punctuation/newline-like tokens. The next critical gap is finding the
  first divergent layer or stage in the 43-layer scheduler body.
- Sprint 027 shipped the selected-token correctness fix and checkpoint
  diagnostics. The V100 scheduler now matches the official short-prompt
  expected token bytes `3136`; checkpoint replay proves the seed, early layers,
  and layer-4 after-attention match the CPU source-layout oracle, while
  layer-4 final HC still shows FFN numeric drift. The current readiness
  blockers are now public serving, MTP, and throughput benchmarking.
- Sprint 028 extracted the selected-token path into a reusable one-shot V100
  replay runtime and `tools/ds4-v100-replay`. The tool loads all eight resident
  stages, replays prompt tokens, generates greedy continuations, verifies token
  bytes, and emits timing/memory JSON. Throughput/timing evidence now exists;
  the remaining readiness blockers are public serving and MTP.
- Sprint 029 shipped the first resident HTTP appliance surface. The replay
  runtime can now reset all eight stage schedulers between independent one-slot
  loopback requests, `tools/ds4-v100-replay --serve` returns the expected token
  bytes `3136`, and the full gate now reports readiness with only `mtp`
  missing.
- `docs/architecture/DS4-V100-LAYOUT.md` is the architecture anchor for
  sharding, memory layout, kernel selection, tensor-parallel alternatives, and
  context/slot assumptions. Sprint plans should reference it instead of
  re-deriving the topology.

## Sprint Sequence

### Sprint 001 - Baseline DS4 V100 Appliance Planner And Source Inventory [complete]

- **Goal**: Prove the model inventory, source dtype mix, memory envelope, and
  baseline layer-sharded V100 topology.
- **Rationale**: The project first needed to know whether the high-intelligence
  DS4 Flash source model could fit and how it should be mapped before runtime
  work could be safe.
- **Outcome**: `SHIP`. The canonical source model was identified, exact tensor
  inventory was recorded, and the planner showed 8x V100 pure-residency
  feasibility for the baseline topology.

### Sprint 002 - Source Loader And Pack Manifest Baseline [complete]

- **Goal**: Teach the runtime tools to recognize the native source GGUF layout
  and emit an inventory-backed pack manifest.
- **Rationale**: The source layout differs from the older q2/q4 DS4 GGUF family,
  so loader and manifest work had to precede any real execution path.
- **Outcome**: `EXTEND`. The source model is recognized, validated, and
  manifested, while generation remains intentionally guarded until V100
  FP8/MXFP4 execution paths exist.

### Sprint 003 - Manifest-Driven Packer Baseline [complete]

- **Goal**: Convert the Sprint 002 manifest into deterministic per-GPU shard
  offsets and a pack index.
- **Rationale**: Runtime residency needs immutable GPU-owned weight shards, not
  ad hoc GGUF offsets or persistent dequantized copies.
- **Outcome**: `SHIP`. The packer writes deterministic `gpuN.weights` layouts
  and validates source ranges; full real-model shard emission was deferred to
  persistent cluster scratch.

### Sprint 004 - Runtime Pack Loading And V100 Device Residency Smoke [complete]

- **Goal**: Prove that the runtime can reconcile pack metadata and upload all
  source-faithful packed bytes to the 8 V100s.
- **Rationale**: Device residency needed to be proven before compute scheduling,
  KV allocation, or source-model decode could be credible.
- **Outcome**: `SHIP`. Full real-model shards were emitted, both GGUF and shard
  providers loaded all 1328 tensors into CUDA device arenas, spot checks and a
  cross-provider check passed, and all GPUs retained the required reserve.

### Sprint 005 - First Resident BF16 Gather/Expand Probe [complete]

- **Goal**: Execute a diagnostic BF16 row-gather/expand probe from resident
  `ds4_gpu_arena` bytes, using `token_embd.weight` as the first source-format
  tensor family and returning host F32 samples for exact verification.
- **Rationale**: BF16 embedding is the lowest-risk useful source-dtype proof
  after residency. It validates arena pointers, descriptor bounds, dtype
  expansion, and CUDA launch semantics before Sprint 006 introduces production
  multi-GPU execution context.
- **Outcome**: `SHIP`. Host-stub and CUDA probes passed, GGUF and shard
  provider `token_embd.weight` probes passed on the 8x V100 pod, and source
  model generation remains guarded.

### Sprint 006 - Multi-GPU Execution Context And Layer Skeleton [complete]

- **Goal**: Introduce the production 8-GPU execution context and a layer-owned
  no-math layer skeleton with hidden-context relay boundaries and explicit
  fail-closed V100 execution-format policy.
- **Rationale**: Full decode requires streams, handles, scratch, tensor
  descriptors, device ownership, and boundary transfer semantics that are
  shaped by the first resident tensor probe. V100 has no native BF16, FP8, or
  FP4 tensor-core path, so Sprint 006 must encode BF16 as source/probe only,
  FP8/MXFP4 as packed inputs to later registered kernels, FP16 HMMA with FP32
  accumulation as the dense production target, and FP32 as control/debug math
  rather than a broad GEMM fallback.
- **Outcome**: `SHIP`. The sidecar context, descriptor policy, real pack
  skeleton walk, production 8x V100 topology check, CUDA resource ownership, and
  HC relay smoke shipped. Generation remains guarded.

### Sprint 007 - Source-Layout Single-Slot Decode Oracle [complete]

- **Goal**: Build a guarded CPU-only source-layout oracle that proves exact
  BF16/F32/I32/F8_E4M3_B128/MXFP4 semantics and matches at least one short
  official first token before production V100 kernels are trusted.
- **Rationale**: Source dtype and V100 runtime dtype must stay separate; a
  fail-closed correctness oracle must come before prefill, long context,
  multi-slot scheduling, MTP, or server deployment.
- **Outcome**: `SHIP`. The guarded oracle selected the official expected token
  `16` for `short_reasoning_plain`, normal source generation remains guarded,
  MXFP4 row semantics were corrected, and source-layout KV defaults to F16
  before Sprint 008 device-kernel work.

### Sprint 008 - Source Oracle Harness And V100 KV Admission Anchors [complete]

- **Goal**: Turn Sprint 007's manual source-oracle proof into automated
  official-vector validation, add source-layout guard regressions, expose exact
  F16 KV admission/reporting for the layer-owned V100 topology, and land one
  bounded CUDA source-format anchor.
- **Rationale**: Full V100 source-layout prefill should not begin until the
  oracle, guard, memory-admission, and source-format device-anchor contracts are
  executable and tested.
- **Outcome**: `SHIP`. The source oracle runner selected
  `short_reasoning_plain` token bytes `3136`, guard checks passed on the source
  model, exact F16 KV admission is reported by stage and fails closed on
  over-budget slots, MXFP4 source layout parity is hardened, and a bounded CUDA
  F8_E4M3_B128 row-decode anchor passed on V100 `sm_70`.

### Sprint 009 - V100 Prefill And Compressed KV Execution [complete]

- **Goal**: Implement the first layer-owned V100 source-layout prompt prefill
  and compressed KV/indexer state update path, validated against the Sprint 007
  oracle and Sprint 008 KV admission contract.
- **Rationale**: This turns the planning/anchor surfaces into usable prompt
  handling while preserving fail-closed normal serving until correctness is
  demonstrated.
- **Outcome**: `SHIP`. The diagnostic path now allocates derived F16 KV arenas,
  validates V100 allocation/admission at 256K and 1M single-slot tiers, runs
  source-layout guards on the real model, and passes a bounded CUDA prefill/KV
  smoke covering ratio-128 and ratio-4/indexer state updates on `sm_70`.

### Sprint 010 - V100 Single-Slot Decode Integration [complete]

- **Goal**: Wire the bounded Sprint 009 KV surfaces into a real layer-owned
  single-slot V100 prefill/decode slice that consumes projection/compressor
  outputs and compares a bounded result against the source oracle.
- **Rationale**: Sprint 009 proved diagnostic KV allocation and row/state
  updates, but deployment should wait until V100 layer execution reaches
  selected-token or bounded-logit correctness.
- **Outcome**: `SHIP`. Stage-owned KV subviews and updates now pass on V100,
  and real compressor recurrence is validated against CPU references for
  ratio-128, ratio-4 attention, and ratio-4 indexer-shaped paths. Full
  source-format dense projection, MoE, logits, selected-token decode, and
  serving remain deferred.

### Sprint 011 - V100 Source Projection And Attention Slice [complete]

- **Goal**: Prove bounded source F8/BF16 projection boundaries and feed
  projection-equivalent device tensors through ratio-4 and ratio-128
  attention/compressor slices on V100.
- **Rationale**: Sprint 010 proved KV ownership and compressor recurrence, but
  the next untrusted surface is source-format projection math. Full logits
  should wait until this path is correct.
- **Outcome**: `SHIP`. Source-F8 projection diagnostics now run from
  device-resident arenas into device tensors, BF16/F32 policy is executable,
  ratio-128 and ratio-4 attention/compressor slices compare against CPU/source
  references, and stage-owned KV writes can consume device-resident projection
  rows.

### Sprint 012 - V100 Appliance Gate And Bounded Output-Head Logits [complete]

- **Goal**: Add a bounded source-BF16 output-head/logits primitive on V100 and
  a runnable appliance readiness gate that validates real-model guards,
  existing V100 smokes, and the new logits/top-k smoke.
- **Rationale**: Deployment should wait for a coherent logits-producing V100
  path. Sprint 011 proves projection-fed attention/compressor slices; Sprint
  012 fills the output-head/logits surface and makes readiness status explicit
  without pretending full MoE or serving are complete.
- **Outcome**: `SHIP`. Source-BF16 output-head rows can now be reduced into
  bounded logits on V100 and top-k compared against a CPU reference. The
  appliance gate passes implemented checks and correctly reports `ready=false`
  until full MoE/selected-token serving exists.

### Sprint 013 - V100 Source MXFP4 MoE And Selected-Token Gate [complete]

- **Goal**: Add a bounded source-MXFP4 routed expert primitive and a
  single-token router/MoE/output-head fixture that produces a selected-token
  comparison on V100.
- **Rationale**: Sprint 012 proves the output-head/logits surface but the gate
  still reports not-ready. The next concrete blocker is the routed expert path:
  DS4 Flash stores routed gate/up/down experts as MXFP4 source tensors, and
  deployment should wait until MoE and selected-token evidence exist.
- **Outcome**: `SHIP`. MXFP4 source expert matmuls and a bounded router/MoE/
  output-head selected-token smoke pass on V100. The remaining readiness gap is
  real pack-index layer integration and shared-expert/full scheduler wiring.

### Sprint 014 - V100 Real Pack-Index Layer Descriptor Gate [complete]

- **Goal**: Add a fail-closed descriptor gate that validates the real pack-index
  rows needed by a source-layout layer, including attention, compressor/indexer,
  router, routed/shared experts, HC controls, and output head.
- **Rationale**: Sprint 013 proves synthetic MoE composition. Deployment should
  wait until the same kernel surfaces consume real model descriptors. A strict
  descriptor contract is the next integration step before real layer compute.
- **Outcome**: `SHIP`. The descriptor gate validates 35 real layer-2/global
  descriptors, fails closed on missing required rows, and is wired into the
  V100 appliance gate behind `--pack-index`.

### Sprint 015 - V100 Descriptor-Bound FFN Compute Gate [complete]

- **Goal**: Materialize validated pack-index descriptors into runtime bindings
  and consume real source-model bytes at real pack offsets in a
  descriptor-bound FFN compute path.
- **Rationale**: Descriptor validation is necessary but not sufficient; the
  next readiness jump is executing real model bytes through the bounded kernel
  surfaces, including the shared expert path.
- **Outcome**: `SHIP`. Runtime tensor bindings landed, layer-2 binding
  validation passes locally and on the pod, and a descriptor-bound V100 FFN
  smoke executes real routed MXFP4 plus shared F8 bytes from the source GGUF at
  real pack offsets.

### Sprint 016 - V100 Descriptor-Bound Router FFN Gate [complete]

- **Goal**: Upgrade descriptor-bound FFN compute from fixed expert to
  model-selected routed experts using real `ffn_gate_inp.weight` and
  `ffn_gate_tid2eid` descriptors.
- **Rationale**: Sprint 015 proves real-byte FFN compute, but serving requires
  real router scheduling before a coherent layer state machine can be trusted.
- **Outcome**: `SHIP`. Source-F32 arena matmul landed, the descriptor-bound FFN
  smoke computes router logits from real bytes, selects experts through the real
  hash-router table, executes all six selected routed experts plus the shared
  expert, and passes the full V100 appliance gate.

### Sprint 017 - V100 Scheduler-Owned Layer State Gate [complete]

- **Goal**: Introduce a reusable scheduler-owned layer execution state that
  binds real descriptors once and owns the router/FFN scratch needed by later
  attention, residual, norm, and selected-token integration.
- **Rationale**: Sprint 016 still proves router-selected FFN as a standalone
  smoke. The next readiness gap is making descriptor-bound execution a runtime
  surface the appliance scheduler can call instead of a test-local composition.
- **Outcome**: `SHIP`. `ds4_v100_layer_state` now owns descriptor-bound
  router/FFN metadata, route matrix construction, source row views, and FFN
  arena-span sizing. The descriptor-bound FFN smoke uses it, and the V100
  appliance gate includes and passes `layer_state`.

### Sprint 018 - V100 Descriptor-Bound Attention Projection Residual Norm Gate [complete]

- **Goal**: Extend the scheduler-owned layer state from router/FFN ownership to
  descriptor-bound attention projection/control ownership, then run real
  source-byte attention projection, residual add, and norm work on V100.
- **Rationale**: Serving is still blocked by the lack of full layer output.
  Sprint 017 created the state surface; Sprint 018 should bridge existing
  synthetic attention kernels to real descriptor-bound attention source bytes
  without claiming full softmax/compressed-KV layer correctness.
- **Outcome**: `SHIP`. Attention/control descriptors are part of
  `ds4_v100_layer_state`, the layer-state smoke validates real attention
  dimensions and arena span, and the new descriptor-bound attention smoke runs
  real source-byte q/kv/output projection, residual add, and FFN pre-norm
  surfaces on V100 against CPU source-format references.

### Sprint 019 - V100 Integrated Single-Layer Runtime Slice [complete]

- **Goal**: Replace Sprint 018's bounded attention-output proxy with a
  scheduler-owned single-layer executor that produces a real next-hidden vector
  for a representative ratio-4 layer by composing semantic attention,
  residual/norm, and router-selected FFN.
- **Rationale**: The appliance still cannot produce the next hidden state from
  a real layer. Sprint 018 proved real projection/control surfaces; Sprint 019
  should ship a reusable runtime slice instead of another isolated primitive.
- **Outcome**: `SHIP`. `ds4_v100_layer_execute` now composes real
  descriptor-bound projection bytes, semantic raw/compressed attention inputs,
  grouped F8 attention output, residual, FFN pre-norm, router-selected MXFP4
  routed experts, shared F8 expert, and final next-hidden residual. The
  integrated smoke passes on one V100 and in the full appliance gate. Real
  compressor/indexer descriptor binding and HC pre/post scheduling remain the
  next blockers.

### Sprint 020 - V100 Compressor/Indexer And HC Scheduler Bridge [extended]

- **Goal**: Bind real compressor/indexer descriptors into layer state, execute
  compressed row generation/selection inside the layer executor, and wrap the
  hidden-vector body with DS4 HC pre/post scheduling.
- **Rationale**: Sprint 019 proves the hidden-vector body of one layer, but the
  appliance still needs real compressed-KV production, ratio-4 indexer
  visibility, and HC state handling before a full 43-layer selected-token path
  is credible.
- **Outcome**: `EXTEND`. `ds4_v100_layer_state` now binds real
  compressor/indexer descriptors and the executor has an HC-state entrypoint
  that runs DS4 attention and FFN HC pre/post around the hidden-vector body.
  The full 8-GPU V100 appliance gate passes with `ready=false`. Executor-owned
  compressed-row generation and indexed ratio-4 compressed attention move to
  Sprint 021.

### Sprint 021 - Executor-Owned Compressor/Indexer Decode Rows [complete]

- **Goal**: Move attention compressor rows, ratio-4 indexer compressor rows,
  indexer scoring/top-k, and indexed compressed attention into the executor
  instead of passing test-built compressed KV rows.
- **Rationale**: Sprint 020 proved the descriptors and HC layer surface. The
  next correctness blocker is making compressed KV production part of the real
  scheduler-owned layer path.
- **Outcome**: `SHIP`. `ds4_v100_layer_execute` now accepts mutable
  decode-cache state, generates raw KV, attention compressed rows, ratio-4
  indexer rows, indexer top-k visibility, and indexed mixed attention from real
  descriptors. The integrated smoke forces `indexer_top_k=1` to reach indexed
  attention in eight decode steps, while production default remains 512. The
  full V100 gate passes and remains `ready=false` pending full scheduler,
  selected-token decode, serving, MTP, and throughput.

### Sprint 022 - Bias Router And Resident Stage Scheduler [complete]

- **Goal**: Remove the hash-router-only execution limit and introduce a
  scheduler-owned resident stage walk over real pack bytes.
- **Rationale**: The model cannot reach full selected-token decode if the
  executor stops at layer 2 or if scheduling remains a single-layer test
  fixture. Stage 0 is the right first target because it owns token embedding
  and includes SWA-only, ratio-4, and ratio-128 layers.
- **Outcome**: `SHIP`. The layer executor now supports hash and bias routers,
  layer 3 ratio-128 bias routing passes on V100, and
  `ds4_v100_stage_scheduler` uploads the full gpu0 shard into a resident arena
  and executes layers 0-5 from a token embedding seed. The full V100 gate
  passes and remains `ready=false` pending cross-GPU 43-layer scheduling,
  selected-token decode, serving, MTP, and throughput.

### Sprint 023 - Cross-GPU Two-Stage Scheduler Handoff [complete]

- **Goal**: Prove the first real scheduler handoff between resident stage
  owners.
- **Rationale**: Stage-local scheduling is insufficient for a layer-sharded
  appliance. The next risk is whether HC can move between GPUs and whether the
  CUDA backend can safely run the same source-model helpers on more than one
  device in one process.
- **Outcome**: `SHIP`. The scheduler now runs layers 0-5 on gpu0, copies HC to
  gpu1 with `cudaMemcpyPeer`, and runs layers 6-11 on gpu1. CUDA tensor
  allocation/copy paths now track device ownership, and model-range caches are
  device-local to avoid cross-GPU pointer reuse. The full V100 gate passes and
  remains `ready=false` pending full 43-layer scheduling, selected-token
  decode, serving, MTP, and throughput.

### Sprint 024 - Full 8-Stage Scheduler Chain [complete]

- **Goal**: Generalize the scheduler handoff from two stages to the full
  8-GPU, 43-layer model body.
- **Rationale**: Output-head correctness is not meaningful until final HC is
  produced by the real layer-sharded body, not a partial stage fixture.
- **Outcome**: `SHIP`. The full scheduler smoke opens all eight resident stage
  arenas, executes layers 0-42, handoffs HC across every stage boundary, and
  verifies finite nonzero final HC on gpu7. The V100 gate now removes
  `full_43_layer_scheduler` from readiness when this check passes and remains
  `ready=false` pending selected-token decode, serving, MTP, and throughput.

### Sprint 025 - Scheduler Output-Head Selected Token Surface [complete]

- **Goal**: Attach gpu7 output-head selected-token execution to the resident
  scheduler.
- **Rationale**: Full-body traversal is necessary but not sufficient; the
  appliance needs final HC collapse, output normalization, vocab projection,
  and a top-1 token surface before selected-token correctness can be debugged.
- **Outcome**: `EXTEND`. The output-head path runs on V100 and produces a
  finite selected token after replaying the `short_reasoning_plain` prompt, but
  the selected token does not match the official/source oracle. Readiness still
  blocks on `real_model_selected_token`.

### Sprint 026 - Output-Head Divergence Localization [complete]

- **Goal**: Prove or eliminate the gpu7 output-head adapter as the cause of the
  selected-token mismatch.
- **Rationale**: Sprint 025 proved that the scheduler can produce logits, but
  not whether the mismatch comes from final HC collapse/vocab projection or
  from earlier layer execution.
- **Outcome**: `SHIP`. The deterministic HC parity smoke matches CPU and V100
  output-head top-5 exactly enough for the diagnostic tolerance, and the prompt
  top-k diagnostic records the remaining oracle mismatch. The next blocker is
  stage/layer HC divergence localization inside the 43-layer body.

### Sprint 027 - V100 Selected-Token Correctness And HC Checkpoints [complete]

- **Goal**: Localize scheduler-body divergence and make the official
  selected-token oracle pass on V100.
- **Rationale**: Output-head parity passed in Sprint 026, so the next useful
  implementation was checkpoint visibility through the actual 43-layer body
  rather than more output-head work.
- **Outcome**: `SHIP`. The scheduler now decodes native BF16 token embeddings
  correctly, defaults KV/cache mutation to the F16 source-layout contract, and
  passes the selected-token gate for expected bytes `3136`. The gate remains
  `ready=false` only for public serving, MTP, and throughput.

### Sprint 028 - V100 Replay Runtime And Timing Tool [complete]

- **Goal**: Move selected-token replay out of a smoke test and into a reusable
  appliance runtime/tool with timing counters.
- **Rationale**: Correctness alone was not a usable surface. The project needed
  a commandable path that loads the resident 8-stage scheduler, emits tokens,
  and measures where time is going.
- **Outcome**: `SHIP`. `tools/ds4-v100-replay` generates tokens through the V100
  scheduler, verifies expected bytes `3136`, and emits JSON timing/memory data.
  The gate should now remove `throughput_benchmark`; public serving and MTP
  remain open.

### Sprint 029 - V100 Resident HTTP Appliance Smoke [complete]

- **Goal**: Keep the V100 replay runtime resident behind a minimal loopback
  HTTP endpoint and prove selected-token correctness through the served path.
- **Rationale**: A CLI replay tool is useful for measurement, but the appliance
  needs a long-running process that keeps all eight stage schedulers resident
  and handles independent requests without reuploading weights each time.
- **Outcome**: `SHIP`. `tools/ds4-v100-replay --serve` exposes
  `/v100/selected-token`, resets scheduler KV/HC state per request, returns
  expected bytes `3136`, and the full V100 gate now reports
  `missing=mtp`.

## Parking Lot

- See `docs/sprints/SPRINT-004-DEFERRED.md`: first source-format math probe,
  source-model decode correctness, MTP, production multi-GPU context, hidden
  context relay, layer scheduler, tensor-parallel variants, multi-slot
  scheduling, KV residency, JSON reports, upload optimization, and pack-only
  runtime boot.
- See `docs/sprints/SPRINT-004-FOLLOWUPS.md`: model-less default test target,
  direct CUDA arena unit target, and upload timing metrics.
- See `docs/sprints/SPRINT-005-DEFERRED.md`: HC expansion, device-output and
  stream-aware probe variants, source-layout embedding dtype cleanup, F16
  output, F32 control tensor probe, additional BF16 tensors, FP8/MXFP4 compute
  probes, and default model-less `make test` cleanup.
- See `docs/sprints/SPRINT-006-DEFERRED.md`: decode, KV population, real
  FP8/MXFP4/INT kernels, tensor-parallel exceptions, output-head math, MTP,
  serving/deployment, and host-backed or persistent dequantized runtime paths.
- See `docs/sprints/SPRINT-007-DEFERRED.md`: production V100 FP8/MXFP4 kernels,
  device-side oracle reads, prefill/KV, multi-slot scheduling, MTP, public
  CLI/server exposure, tensor-parallel exceptions, and full-logit oracle
  capture.
- See `docs/sprints/SPRINT-007-FOLLOWUPS.md`: MXFP4 parity hardening,
  official-vector automation, source-oracle guard tests, and Sprint 008
  correctness anchors.
- See `docs/sprints/SPRINT-009-FOLLOWUPS.md`: Sprint 010 integration work for
  production projection/compressor outputs, source-oracle comparison, explicit
  KV state subviews, and deployment sequencing.
- See `docs/sprints/SPRINT-010-FOLLOWUPS.md`: Sprint 011 blockers for real
  source-format projection, attention/layer output, router/expert execution,
  bounded logits/top-k comparison, and deployment re-sequencing.
- See `docs/sprints/SPRINT-011-DEFERRED.md` and
  `docs/sprints/SPRINT-011-FOLLOWUPS.md`: Sprint 012 blockers for coherent
  layer output, router/shared/routed expert correctness, output-head or
  selected-token comparison, production F8 projection kernels, and deployment
  re-sequencing.
- See `docs/sprints/SPRINT-014-FOLLOWUPS.md`: runtime descriptor table,
  descriptor-bound layer compute, layer-class descriptor coverage, shared
  expert execution, and readiness-policy cleanup.
- See `docs/sprints/SPRINT-015-FOLLOWUPS.md`: real router scheduling,
  descriptor-bound layer state, attention/residual/norm integration,
  selected-token real-model gate, and production memory reuse.
- See `docs/sprints/SPRINT-016-FOLLOWUPS.md`: scheduler-owned layer state,
  attention/residual/norm integration, real-model selected-token gate,
  production arena reuse, and representative router coverage.
- See `docs/sprints/SPRINT-017-FOLLOWUPS.md`: descriptor-bound attention,
  residual/norm/HC layer slice, real-model selected-token gate, production
  arena reuse, and bias-router coverage.
- See `docs/sprints/SPRINT-018-FOLLOWUPS.md`: full attention softmax over
  raw/compressed KV, combined attention plus FFN layer slice, real-model
  selected-token gate, and production arena reuse.
- See `docs/sprints/SPRINT-019-FOLLOWUPS.md`: compressor/indexer descriptor
  binding, HC pre/post layer scheduling, full 43-layer selected-token gate,
  production arena reuse, and timing/throughput counters.
- See `docs/sprints/SPRINT-021-FOLLOWUPS.md`: full 43-layer single-slot
  scheduler, production indexer-threshold stress, reusable scratch/timing
  counters, HC CPU reference, serving, MTP, and multi-slot throughput.
- See `docs/sprints/SPRINT-024-FOLLOWUPS.md`: output-head selected-token gate,
  full-chain failure-local reports, per-stage upload/memory timing, relay
  optimization, MTP, and throughput.
- See `docs/sprints/SPRINT-025-FOLLOWUPS.md`: selected-token divergence
  localization, top-k diagnostics, output-head CPU parity, prompt replay
  counters, and failure-preserving logs.
- See `docs/sprints/SPRINT-026-FOLLOWUPS.md`: stage/layer HC divergence
  checkpoints, per-layer execution reports, full-gate build guards, parallel
  resident uploads, and continued deferral of serving/MTP/throughput.
- See `docs/sprints/SPRINT-027-FOLLOWUPS.md`: public one-slot serving,
  throughput counters, layer-4 FFN numeric drift, explicit FP8 KV validation,
  and continued deferral of MTP/multi-slot scheduling until the single-slot
  baseline is usable.
- See `docs/sprints/SPRINT-028-FOLLOWUPS.md`: HTTP/process serving around the
  replay runtime, scheduler reset or single-session semantics, open/upload
  reduction, longer decode baselines, and continued MTP/multi-slot deferral.
- See `docs/sprints/SPRINT-029-FOLLOWUPS.md`: MTP implementation/validation,
  parallel resident stage open/upload, longer resident decode baselines,
  serving API hardening, and continued multi-slot deferral.
- See `docs/sprints/SPRINT-001-DEFERRED.md`: q2/q4 fallback, SSD/host-backed
  offload, INT8 default-layout questions, F8 KV mode, and broad TurboMind or
  tc-grid kernel import as conditional paths rather than default strategy.
- See `docs/sprints/SPRINT-001-FOLLOWUPS.md`,
  `docs/sprints/SPRINT-002-FOLLOWUPS.md`, and
  `docs/sprints/SPRINT-003-FOLLOWUPS.md`: earlier follow-ups that are now
  either completed by Sprints 002-004 or carried forward in the Sprint 004
  deferred/follow-up lists.

## Pivot Log

| Date | What Changed | Why | Sprints Affected |
|------|-------------|-----|-----------------|
| 2026-05-17 | Created the first DS4 V100 appliance vision after Sprint 004 residency shipped. | The project has moved from feasibility and pack-residency proof to source-format compute, correctness, deployment, and performance sequencing. | Sprint 005+ |
| 2026-05-17 | Refined Sprint 005 from a generic source-format compute probe to a BF16 resident row-gather probe on `token_embd.weight`. | Planning consensus found BF16 embedding is the smallest useful proof of arena-resident compute and avoids premature FP8/MXFP4, scheduler, or decode work. | Sprint 005-006 |
| 2026-05-17 | Corrected Sprint 005 language from BF16 compute to BF16 gather/expand and shipped the probe. | V100 has no native BF16 tensor-core execution; the useful proof is resident addressing and exact dtype expansion, while production compute must target FP16 or low-bit/integer kernels. | Sprint 005-006 |
| 2026-05-17 | Scoped Sprint 006 to sidecar V100 context, fail-closed execution policy, HC relay, memory reserve, and no-math layer skeleton. | The next risk is not another dtype probe; it is proving the appliance runtime topology without silently promoting BF16/FP8/FP4 to unsupported native V100 compute or defaulting the model to FP32 GEMMs. | Sprint 006-007 |
| 2026-05-17 | Shipped Sprint 006 and moved the next milestone to single-slot decode correctness. | The 8-GPU context, descriptor policy, peer topology, memory reserve, and HC relay contract are now verified; the next unknown is numerical correctness through actual attention, MoE, KV, and output-head math. | Sprint 007+ |
| 2026-05-18 | Refined Sprint 007 into a guarded source-layout oracle sprint. | Planning consensus found that exact FP8/MXFP4/BF16 source semantics and a narrow CPU-only diagnostic unlock are the right next gate before production V100 kernels, prefill, or deployment. | Sprint 007-008 |
| 2026-05-18 | Shipped Sprint 007 source-layout oracle, corrected MXFP4 row ordering, and restored F16 KV as the source correctness baseline. | The official vector exposed both a bad interleaved MXFP4 assumption and an unsafe forced FP8 KV round-trip; matching GGML's low-half/high-half layout and the default F16 KV contract produced the expected first token and gives Sprint 008 a real correctness anchor. | Sprint 007-008 |
| 2026-05-18 | Re-scoped Sprint 008 as oracle automation, F16 KV admission, and one CUDA source-format anchor. | Full V100 source-layout prefill combines too many unproven contracts; making oracle, guard, memory, and source-format device checks executable first reduces risk before runtime KV execution. | Sprint 008-010 |
| 2026-05-18 | Shipped Sprint 008 source oracle harness, F16 KV admission, source dtype hardening, and CUDA F8 source-format anchor. | The project now has executable correctness, memory-admission, and first device source-format contracts for the Sprint 009 V100 prefill/KV implementation. | Sprint 008-009 |
| 2026-05-18 | Shipped Sprint 009 bounded V100 prefill/KV execution and inserted a single-slot decode integration sprint before deployment. | KV arena allocation, source-layout guards, and CUDA ratio-class row/state updates now pass on V100 `sm_70`; the next risk is real projection/compressor integration and oracle comparison, not server packaging. | Sprint 009-011 |
| 2026-05-18 | Shipped Sprint 010 stage-owned KV views/updates and real compressor recurrence smokes, then moved deployment behind a logits-producing V100 source-layout gate. | The project now trusts per-layer KV/state ownership and compressor recurrence on V100, but serving still lacks real source-format dense projection, MoE, output-head logits, and selected-token correctness. | Sprint 010-012 |
| 2026-05-18 | Split Sprint 011 into a source projection and attention-slice gate before the full logits gate. | Planning showed the next concrete risk is source FP8/BF16 projection on V100; full logits remain too broad until projection and bounded attention/compressor slices are trusted. | Sprint 011-013 |
| 2026-05-18 | Shipped Sprint 011 source projection and attention/compressor slice, keeping deployment behind Sprint 012's logits gate. | V100 now has device-resident source-F8 projection diagnostics, BF16/F32 policy checks, projection-fed ratio-4/ratio-128 attention/compressor smokes, and device-row KV writes, but still lacks MoE, output head, and selected-token correctness. | Sprint 012-013 |
| 2026-05-18 | Shipped Sprint 014 real pack-index descriptor validation and moved Sprint 015 to descriptor-bound layer compute. | The appliance gate now proves the real layer-2 descriptor contract on the V100 pod, so the next risk is converting those descriptors into runtime bindings that launch compute on real resident shard bytes. | Sprint 015+ |
| 2026-05-18 | Shipped Sprint 015 descriptor-bound FFN compute from real source bytes and moved Sprint 016 to scheduler-owned layer slicing. | The appliance now proves real pack offsets can feed routed MXFP4 and shared F8 FFN compute on V100; the next blocker is real router/layer state and attention/residual/norm integration. | Sprint 016+ |
| 2026-05-18 | Shipped Sprint 016 descriptor-bound real router FFN compute and moved Sprint 017 to scheduler-owned layer state. | The appliance now proves real layer-2 router logits, hash-router selected experts, all six routed MXFP4 experts, and shared F8 FFN compute on V100; the next blocker is moving this out of a standalone smoke into a scheduler-owned runtime layer surface. | Sprint 017+ |
| 2026-05-18 | Shipped Sprint 017 scheduler-owned layer state and moved Sprint 018 to descriptor-bound attention/layer output. | Router/FFN descriptor ownership is now a reusable runtime surface instead of test-local glue; the next blocker is producing a coherent hidden state through attention, residual, norm, and HC composition. | Sprint 018+ |
| 2026-05-18 | Shipped Sprint 018 descriptor-bound attention projection/residual/norm and moved Sprint 019 to full attention/layer output. | Real source-byte q/kv/output projection, residual add, and FFN pre-norm now pass on V100 through layer state; the next blocker is semantic attention softmax over raw/compressed KV and a coherent next hidden state. | Sprint 019+ |
| 2026-05-18 | Shipped Sprint 019 integrated hidden-vector layer executor and moved Sprint 020 to compressor/indexer plus HC scheduling. | Layer 2 now produces a bounded next-hidden vector through semantic raw/compressed attention inputs and real router-selected FFN on V100; the remaining blocker is generating those compressed rows from real descriptors and running the true HC-state layer scheduler. | Sprint 020+ |
| 2026-05-18 | Extended Sprint 020 with compressor/indexer descriptor binding and a V100 HC-state layer entrypoint. | The runtime now has the true `[4 x 4096]` HC layer surface and real compressor/indexer descriptor ownership, but still needs executor-owned compressed-row generation before selected-token decode. | Sprint 021+ |
| 2026-05-18 | Shipped Sprint 021 executor-owned compressor/indexer decode rows and indexed ratio-4 attention. | The representative layer now owns raw/compressed/indexer cache mutation from real descriptors on V100; the next blocker is wiring all 43 layers into a single-slot selected-token scheduler. | Sprint 022+ |
| 2026-05-18 | Shipped Sprint 022 bias-router execution and a resident stage-0 scheduler. | The runtime now walks layers 0-5 from resident gpu0 pack bytes and validates both router families on V100; the next blocker is cross-GPU HC relay through all stages and output-head selected-token comparison. | Sprint 023+ |
| 2026-05-18 | Shipped Sprint 023 cross-GPU two-stage scheduling. | The runtime now executes layers 0-11 across gpu0 and gpu1 with a peer HC handoff and device-local CUDA model caches; the next blocker is generalizing the stage chain through gpu7. | Sprint 024+ |
| 2026-05-18 | Shipped Sprint 024 full 8-stage scheduling. | The runtime now executes all 43 layers across the 8x V100 body and removes `full_43_layer_scheduler` from gate readiness; the next blocker is output-head selected-token comparison against the source oracle. | Sprint 025+ |
| 2026-05-18 | Extended Sprint 025 with scheduler-owned output-head selected-token execution. | The output-head path now runs and produces finite logits/top-1 on V100, but official-vector comparison fails (`3136` expected, `0a0a` selected), so the next blocker is divergence localization rather than more scheduling structure. | Sprint 026+ |
| 2026-05-18 | Shipped Sprint 026 output-head divergence localization. | Deterministic CPU-vs-V100 output-head parity passes on gpu7, so the selected-token mismatch is now scoped to the 43-layer scheduler body; the next sprint should checkpoint HC after layer/stage boundaries. | Sprint 027+ |
| 2026-05-18 | Shipped Sprint 027 selected-token correctness and HC checkpoint diagnostics. | BF16 embedding decode and F16 KV/cache semantics now match the CPU source-layout oracle closely enough for the official V100 selected-token gate to pass; the next milestone moves from correctness blocking to public serving and measurement. | Sprint 028+ |
| 2026-05-18 | Shipped Sprint 028 V100 replay runtime and timing tool. | The working scheduler path is now callable outside a smoke test and emits machine-readable token/timing/memory evidence; the next milestone is keeping that runtime resident behind an HTTP or process-serving surface. | Sprint 029+ |
| 2026-05-18 | Shipped Sprint 029 resident HTTP appliance smoke. | The one-slot selected-token path is now served through a resident loopback process and `public_serving` is no longer a gate blocker; the next milestone is MTP correctness and then performance work such as parallel upload and longer resident decode baselines. | Sprint 030+ |

## Open Questions

1. What reference should define correctness tolerances for mixed
   BF16/F32/F8_E4M3_B128/MXFP4 execution on V100 after MoE is included?
2. What production serving milestone should follow the loopback smoke:
   OpenAI-compatible API, process supervision, or a narrower internal endpoint?
3. How much MTP work should land before longer resident decode benchmarks and
   upload optimization?
4. Should the persistent `/srv/dev/ds4-sprint004` pack become the seed
   deployment artifact, or should a formal pack release format come first?
