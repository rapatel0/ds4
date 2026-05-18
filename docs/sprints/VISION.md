---
created: 2026-05-17
last_updated: 2026-05-18
last_updated_by: sprint-plan
revision: 26
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

### Sprint 018 - V100 Descriptor-Bound Attention Projection Residual Norm Gate [planned]

- **Goal**: Extend the scheduler-owned layer state from router/FFN ownership to
  descriptor-bound attention projection/control ownership, then run real
  source-byte attention projection, residual add, and norm work on V100.
- **Rationale**: Serving is still blocked by the lack of full layer output.
  Sprint 017 created the state surface; Sprint 018 should bridge existing
  synthetic attention kernels to real descriptor-bound attention source bytes
  without claiming full softmax/compressed-KV layer correctness.
- **Plan**: Bind attention/control descriptors through the layer state, reuse
  source-F8 projection and RMSNorm kernels on real layer-2 bytes, add residual
  and FFN pre-norm composition, and gate the slice against CPU source-format
  references.

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

## Open Questions

1. What is the smallest Sprint 018 layer-output slice that meaningfully
   advances serving: attention/residual/norm only, attention plus FFN output, or
   a narrow path that also reaches output logits?
2. What reference should define correctness tolerances for mixed
   BF16/F32/F8_E4M3_B128/MXFP4 execution on V100 after MoE is included?
3. What minimum serving milestone counts as "usable" before optimization:
   one-slot small context, 256K context, or a deployed endpoint with narrower
   context limits?
4. How long should MTP and multi-slot throughput stay deferred after base
   decode works?
5. Should the persistent `/srv/dev/ds4-sprint004` pack become the seed
   deployment artifact, or should a formal pack release format come first?
