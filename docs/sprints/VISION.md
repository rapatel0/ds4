---
created: 2026-05-17
last_updated: 2026-05-17
last_updated_by: sprint-plan
revision: 2
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
- The source model generation guard is still active by design. No source-format
  decode, prefill, KV path, MTP path, or throughput benchmark has been enabled.
- The tightest observed residency case still leaves more than the planned 3 GiB
  reserve on a 32 GB V100. Weight VRAM fit is no longer the primary blocker.
- The main remaining risk is numerical correctness and kernel coverage for the
  mixed BF16/F32/F8_E4M3_B128/MXFP4 source layout on V100, especially attention,
  compressed KV, routing, and routed expert execution.
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

### Sprint 005 - First Resident BF16 Compute Probe [planned]

- **Goal**: Execute a diagnostic BF16 row-gather probe from resident
  `ds4_gpu_arena` bytes, using `token_embd.weight` as the first source-format
  tensor family and returning host F32 samples for exact verification.
- **Rationale**: BF16 embedding is the lowest-risk useful compute proof after
  residency. It validates arena pointers, descriptor bounds, dtype conversion,
  and CUDA launch semantics before Sprint 006 introduces production multi-GPU
  execution context.

### Sprint 006 - Multi-GPU Execution Context And Layer Skeleton [planned]

- **Goal**: Introduce the production 8-GPU execution context and a layer-owned
  execution skeleton with hidden-context relay boundaries.
- **Rationale**: Full decode requires streams, handles, scratch, tensor
  descriptors, device ownership, and boundary transfer semantics that are
  shaped by the first real compute path.

### Sprint 007 - Single-Slot Decode Correctness [planned]

- **Goal**: Run a guarded one-slot, small-context decode path and compare logits
  or generated tokens against a trusted reference.
- **Rationale**: A correctness gate must come before prefill, long context,
  multi-slot scheduling, MTP, or server deployment.

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
