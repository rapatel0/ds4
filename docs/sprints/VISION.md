---
created: 2026-05-17
last_updated: 2026-05-18
last_updated_by: sprint-execute
revision: 7
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
- Sprint 008 is the next correctness milestone: prompt prefill, compressed KV,
  indexer state, and the first production-relevant device-side source-format
  kernel anchors must validate against the Sprint 007 oracle.
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

### Sprint 008 - Prefill, KV, And Compressed Attention [planned]

- **Goal**: Implement prompt prefill plus DS4 SWA, compressed KV, indexer, and
  context-state updates for the layer-owned topology.
- **Rationale**: This turns first-token decode into usable prompt handling and
  establishes the real memory and bandwidth facts for long-context operation.

### Sprint 009 - V100 Appliance Deployment [planned]

- **Goal**: Package the runtime as a cluster-deployed CLI/server path with
  startup residency validation, health checks, and guarded operational defaults.
- **Rationale**: Deployment should wait until source-format decode and prompt
  prefill are correct enough that failures mean serving issues, not basic model
  execution gaps.

### Sprint 010 - Throughput And Context Optimization [tentative]

- **Goal**: Improve aggregate tokens/sec and context-tier admission through
  slot batching, wavefront scheduling, expert kernel selection, relay overlap,
  KV format choices, and output-head tuning.
- **Rationale**: Optimization should be driven by measured bottlenecks from the
  verified decode and prefill path, not by assumptions from the residency sprint.

### Sprint 011 - MTP And Advanced Throughput [tentative]

- **Goal**: Add MTP/speculative decoding and evaluate selective tensor-parallel
  exceptions after the base appliance path is stable.
- **Rationale**: MTP and tensor-parallel variants can amplify a correct runtime,
  but they should not mask correctness issues in the baseline layer-sharded
  scheduler.

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

## Open Questions

1. What reference should define correctness tolerances for mixed
   BF16/F32/F8_E4M3_B128/MXFP4 execution on V100?
2. What minimum serving milestone counts as "usable" before optimization:
   one-slot small context, 256K context, or a deployed endpoint with narrower
   context limits?
3. How long should MTP and multi-slot throughput stay deferred after base
   decode works?
4. Should the persistent `/srv/dev/ds4-sprint004` pack become the seed
   deployment artifact, or should a formal pack release format come first?
