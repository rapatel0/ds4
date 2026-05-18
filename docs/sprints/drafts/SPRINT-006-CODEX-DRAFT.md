# Sprint 006: Multi-GPU Execution Context And Layer Skeleton

## Overview

Sprint 005 proved one narrow resident source-dtype contract: BF16 bytes can be
gathered from a `ds4_gpu_arena` on V100 and expanded to F32 exactly for
diagnostics. Sprint 006 should turn that proof into a production-shaped
multi-GPU runtime skeleton without pretending decode correctness already
exists.

The deliverable is structural and diagnostic: an opaque V100 execution context
that owns per-GPU weight arenas, scratch budgets, streams, cuBLAS handles,
relay buffers, typed tensor descriptors, and the baseline contiguous 8-stage
layer map from `docs/architecture/DS4-V100-LAYOUT.md`. The context must be able
to initialize from reconciled pack metadata, print a topology/policy report,
run a bounded hidden-context relay smoke, and walk the layer schedule without
executing attention, MoE, KV append, or output-head math.

The execution-format policy must be explicit in both code and reports. V100 has
no native BF16, FP8, or FP4 tensor-core execution. Sprint 006 should therefore
encode these rules:

- BF16 source bytes are valid for exact diagnostics and descriptor binding, but
  not as a claimed native V100 tensor-core compute path.
- FP16 is the default activation and GEMM target for future dense execution on
  V100.
- FP32 is the default control, reduction, softmax-adjacent, and debug format.
- FP8 and MXFP4 remain packed source/runtime formats that feed later
  dequant-to-FP16 or validated low-bit kernels; they are not directly executed
  as native tensor-core formats on V100.

This sprint does not enable normal source-model generation. It does not add
decode, prefill, KV population, routed-expert execution, output projection,
MTP, or server exposure. If the source-layout model can generate tokens through
the normal runtime at the end of Sprint 006, the sprint exceeded scope.

## Use Cases

1. **Context topology report**: as the operator, I can build a V100 context
   from a GGUF plus `pack-index.tsv` and get a fail-closed report of layer
   ownership, arena bytes, reserve/headroom, relay bytes, and peer-access
   facts before any decode work starts.
2. **Typed descriptor binding**: as the runtime, I can resolve semantic tensor
   ids into typed resident descriptors keyed by owning GPU, arena span, layer
   ownership, source dtype, and execution class instead of re-deriving state
   from GGUF offsets at execution sites.
3. **Hidden-context boundary validation**: as the CUDA developer, I can
   allocate the boundary payload shape `[active_slots][4][4096]`, execute a
   bounded relay transfer, and prove that stage handoff semantics exist before
   layer math is wired up.
4. **No-math layer walk**: as the next sprint, I can walk the full contiguous
   8-stage layer schedule and verify descriptor presence, tensor ownership, and
   boundary allocation without executing attention, FFN, or output kernels.
5. **Guarded runtime evolution**: as the maintainer, I can land the production
   context shape beside the legacy global CUDA path while keeping normal
   source-model generation closed until decode correctness is proven later.

## Architecture

Sprint 006 should introduce a V100-specific sidecar module rather than widen
the legacy global CUDA path into premature decode ownership. A concrete shape is
`ds4_v100_context`, implemented in a new context module and consumed by a
dedicated smoke tool plus narrow diagnostic entry points. `ds4.c` remains the
authority for model binding, source-layout recognition, and pack reconciliation,
but it must continue to reject normal source-layout generation.

```text
GGUF + pack-index.tsv
    |
    v
ds4.c source-layout bind + pack reconcile
    |
    v
ds4_v100_context_open(...)
    |
    +--> global descriptors + per-stage descriptors + execution policy
    +--> per-GPU arenas / scratch budgets / streams / handles / relay buffers
    +--> topology + peer matrix + memory report
    |
    v
tools/ds4-v100-context-smoke
    |
    +--> relay smoke
    +--> no-math layer walk
    +--> optional targeted descriptor diagnostics
    |
    v
exit with source-model generation guard still active
```

The context module should own four kinds of state:

- `ds4_v100_context`: global topology, execution-policy settings, pack/model
  provenance, and an array of stage records.
- `ds4_v100_stage`: one record per owning GPU with `gpu_id`, contiguous
  `layer_begin/layer_end`, weight arena reference, scratch budget, stream,
  cuBLAS handle, relay buffers, and peer-access capabilities.
- `ds4_v100_tensor_desc`: a typed resident descriptor keyed by semantic tensor
  id, with owning GPU, layer id or global scope, arena offset, byte length,
  source dtype, runtime layout, and execution class.
- `ds4_v100_layer_desc`: a no-math layer contract that names the tensor
  families required for each layer class so the skeleton can validate presence
  and ownership without launching compute kernels.

### Execution-Format Policy

Sprint 006 should encode execution policy as data, not as comments. A minimal
policy table is:

| Tensor Family | Source Format | Sprint 006 Allowed Operation | V100 Execution Target | Forbidden Claim |
|---|---|---|---|---|
| HC control, norms, router metadata, small globals | F32 / I32 and a few BF16 small tensors | exact resident descriptors, report emission, small diagnostic reads | F32 control/reduction kernels | treating these as part of decode completion |
| token embedding, output head, compressor/indexer BF16 families | BF16 | exact BF16 diagnostic gather/expand and descriptor binding | FP16 data path or F32 control path after explicit conversion in later sprints | native BF16 tensor-core execution |
| dense attention and shared-expert packs | F8_E4M3_B128 | descriptor binding only | kernel-local unpack/dequant feeding FP16 HMMA or another validated low-bit path | native FP8 tensor-core execution on V100 |
| routed-expert packs | MXFP4 / FP4 | descriptor binding only | grouped low-bit kernel or bounded unpack-to-FP16 tile path in later sprints | native FP4 tensor-core execution |
| activations and HC relay payloads | runtime values | relay allocation plus synthetic transfer only | FP16 normal, FP32 debug | full layer compute in Sprint 006 |
| KV cache | runtime cache, not source weight | planner fields and reserved ownership only | F16 first | real prefill/decode KV population |

The code should expose a small execution-class enum, for example
`DS4_V100_EXEC_F32_CONTROL`, `DS4_V100_EXEC_F16_HMMA`,
`DS4_V100_EXEC_LOWBIT_KERNEL`, and `DS4_V100_EXEC_DIAGNOSTIC_ONLY`. The
context report should emit per-class counts and must say plainly that BF16,
FP8, and FP4 are source or diagnostic formats, not native V100 tensor-core
formats.

### Relay Contract

The relay primitive should stay narrow and production-shaped:

- Boundary payload shape is `[active_slots][4][4096]`.
- Relay buffers are owned per stage and double-buffered so the later scheduler
  can overlap without redesigning the ABI.
- FP16 is the normal boundary format. FP32 is debug-only.
- Successful relay paths remain device-to-device. Same-GPU loopback is allowed
  for synthetic local tests. Real multi-GPU validation should use peer copy or
  equivalent direct CUDA device transfer. The sprint should fail closed rather
  than silently succeeding through host-backed relay paths.

### Layer Skeleton Boundary

The no-math layer skeleton should validate the baseline 8-stage contiguous map
from the architecture doc:

- `gpu0`: layers 0-5 plus token embedding ownership.
- `gpu1`: layers 6-11.
- `gpu2`: layers 12-17.
- `gpu3`: layers 18-23.
- `gpu4`: layers 24-29.
- `gpu5`: layers 30-34.
- `gpu6`: layers 35-39.
- `gpu7`: layers 40-42 plus output-head ownership.

The skeleton may enter each stage, validate required descriptor families, emit
small diagnostics, and exercise boundary relay, but it must not perform real
attention, MoE, KV, or output projection math.

## Implementation

### Phase 1: Context Contract And Policy Encoding

**Files:**
- `ds4_v100_context.h`
- `ds4_v100_context.c`
- `ds4_pack.h`

**Tasks:**
- [ ] Define opaque `ds4_v100_context`, `ds4_v100_stage`,
      `ds4_v100_tensor_desc`, and `ds4_v100_layer_desc` types.
- [ ] Define an explicit execution-class enum and report fields that encode the
      V100 format policy rather than relying on comments.
- [ ] Encode the baseline contiguous 8-stage layer map from
      `docs/architecture/DS4-V100-LAYOUT.md` as data.
- [ ] Define context-report output that prints stage ownership, arena bytes,
      relay bytes, scratch budget, reserve/headroom, and policy classification.
- [ ] Reject requested topologies or tensor families that drift from the
      baseline map instead of silently re-planning.

### Phase 2: Descriptor Binding And Guard-Preserving Runtime Wiring

**Files:**
- `ds4.c`
- `ds4.h`
- `ds4_pack.c`
- `ds4_pack.h`
- `ds4_v100_context.c`

**Tasks:**
- [ ] Bind validated pack rows to typed descriptors keyed by semantic tensor id
      instead of raw GGUF offsets at execution sites.
- [ ] Cover global BF16/F32 tensors plus the layer-owned tensor families needed
      by the skeleton, including SWA-only, ratio-4, and ratio-128 differences.
- [ ] Add one narrow diagnostic-only entry point for building the context from
      the reconciled model and pack metadata.
- [ ] Keep the normal `ds4_engine_open()` source-layout generation guard in
      place and unchanged for decode/generation paths.
- [ ] Fail closed on missing descriptors, duplicate ownership, bad layer ids,
      arena-range overflow, or source-dtype/policy mismatches.

### Phase 3: CUDA Resource Ownership And Relay Primitive

**Files:**
- `ds4_gpu.h`
- `ds4_cuda.cu`
- `ds4_v100_context.c`

**Tasks:**
- [ ] Add per-stage CUDA resource ownership for stream, cuBLAS handle, scratch
      budget, relay buffers, and peer-access facts.
- [ ] Keep legacy globals such as `g_cublas` and `g_cuda_tmp` out of the new
      context contract; the sidecar context must own its own resources.
- [ ] Add relay allocation and copy helpers for the boundary payload shape
      `[active_slots][4][4096]`.
- [ ] Support FP16 normal relay and FP32 debug relay as explicit modes.
- [ ] Emit a peer matrix and fail closed when a required real multi-GPU relay
      path cannot be validated as device-to-device.

### Phase 4: No-Math Layer Skeleton And Diagnostic Tooling

**Files:**
- `tools/ds4-v100-context-smoke.c`
- `tests/v100_context_plan_smoke.c`
- `tests/cuda_v100_context_smoke.c`
- `Makefile`

**Tasks:**
- [ ] Add a dedicated smoke tool that builds the context, prints the policy and
      topology report, runs relay smoke, and performs a full no-math layer
      walk.
- [ ] Validate that each stage owns the expected global and layer-local
      descriptors before the walk proceeds.
- [ ] Reuse the Sprint 005 BF16 gather/expand probe only as an optional
      descriptor-backed diagnostic spot check, not as a decode entry point.
- [ ] Add model-less tests for layer-map correctness, execution-policy
      classification, descriptor range checks, and invalid-config failures.
- [ ] Add CUDA synthetic tests for stage initialization, relay buffer
      allocation, same-device loopback, and cross-GPU transfer when at least
      two devices are visible.

### Phase 5: Cluster Validation And Close-Out

**Files:**
- `docs/sprints/drafts/SPRINT-006-CONTEXT-SMOKE.log`
- `docs/sprints/drafts/SPRINT-006-GUARD.log`
- `docs/sprints/VISION.md`

**Tasks:**
- [ ] Build the context-smoke targets with `CUDA_ARCH=sm_70`.
- [ ] Run the model-less and CUDA synthetic context tests.
- [ ] Run the real context smoke on the 8x V100 cluster against the source GGUF
      plus current pack index and archive the report.
- [ ] Confirm that normal source-layout generation still fails closed and
      archive the exact guard output.
- [ ] Update `docs/sprints/VISION.md` after the sprint executes.

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `ds4_v100_context.h` | Create | Define the opaque V100 execution context, stage records, descriptor contract, and execution-policy types. |
| `ds4_v100_context.c` | Create | Implement context construction, layer-map ownership, descriptor binding, reporting, and skeleton validation. |
| `ds4.c` | Modify | Reuse source-layout binding and pack reconciliation for diagnostic context creation while preserving the normal source-model generation guard. |
| `ds4.h` | Modify | Export one narrow diagnostic-only context entry point without widening normal decode APIs. |
| `ds4_pack.c` | Modify | Add helper logic for descriptor-oriented lookup and layer-aware iteration over pack rows. |
| `ds4_pack.h` | Modify | Expose any descriptor-supporting lookup or iteration helpers needed by the context module. |
| `ds4_gpu.h` | Modify | Add the minimal CUDA-side types or helpers needed for per-stage relay and topology reporting. |
| `ds4_cuda.cu` | Modify | Implement per-stage CUDA resource ownership, relay helpers, peer-matrix reporting, and sidecar context support without enabling decode. |
| `tools/ds4-v100-context-smoke.c` | Create | Build and validate the bounded V100 context, print reports, run relay smoke, and perform the no-math layer walk. |
| `tests/v100_context_plan_smoke.c` | Create | Cover model-less layer-map, descriptor-policy, and invalid-config checks. |
| `tests/cuda_v100_context_smoke.c` | Create | Cover synthetic CUDA context initialization and relay semantics on visible GPUs. |
| `Makefile` | Modify | Build the new context smoke targets and tests. |
| `docs/sprints/drafts/SPRINT-006-CONTEXT-SMOKE.log` | Create | Archive topology, memory, peer, policy, relay, and layer-walk output from the cluster run. |
| `docs/sprints/drafts/SPRINT-006-GUARD.log` | Create | Archive proof that normal source-layout generation is still guarded. |

## Definition of Done

- [ ] An opaque V100 execution context exists and can initialize the baseline
      8-stage contiguous layer map from reconciled pack metadata.
- [ ] The context owns per-stage weight-arena references, scratch budgeting,
      stream/handle state, relay buffers, and peer-access facts.
- [ ] Typed resident descriptors exist for the global tensors and the
      layer-owned tensor families required by the no-math skeleton walk.
- [ ] The V100 execution-format policy is encoded in code and emitted in the
      context report, including an explicit statement that BF16, FP8, and FP4
      are not native V100 tensor-core execution formats.
- [ ] A bounded relay primitive validates the boundary payload shape
      `[active_slots][4][4096]` in FP16 normal mode and FP32 debug mode.
- [ ] The no-math layer skeleton can walk all planned stages and layers and
      validate ownership plus descriptor presence without executing decode math.
- [ ] No persistent dequantized weight copies are introduced, and no host-backed
      successful weight-residency path is added.
- [ ] The normal source-layout generation guard remains active for `ds4` runtime
      startup and decode/generation paths.
- [ ] Local model-less checks pass, CUDA synthetic context checks pass on V100,
      cluster smoke artifacts are archived, and `git diff --check` passes.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| The sprint expands into decode, attention, or expert-kernel work. | High | High | Keep the only public deliverable as context construction, relay smoke, and no-math skeleton validation; treat working decode as out of scope by definition. |
| The new context becomes tangled with legacy global CUDA state such as `g_cublas` and `g_cuda_tmp`. | Medium | High | Keep a sidecar module with separate resource ownership and avoid retrofitting the legacy path in this sprint. |
| Descriptor binding is incomplete for one layer class, so the skeleton only proves a partial topology. | Medium | High | Make the layer walk fail closed when any required tensor family is missing and cover SWA-only, ratio-4, and ratio-128 cases in model-less tests. |
| Relay looks valid on one machine but depends on an implicit host path or unsupported peer topology. | Medium | High | Emit the peer matrix, require device-to-device success for real multi-GPU validation, and reject silent host-backed success paths. |
| The execution-format policy stays ambiguous and later work accidentally treats BF16/FP8/FP4 as native V100 compute formats. | High | High | Encode execution class in data structures and reports, and make policy mismatches a hard failure in context construction. |
| Scratch or relay budgeting consumes too much of the V100 reserve on one or more stages. | Medium | Medium | Report per-stage reserve/headroom in the context smoke and keep the first scratch/relay allocations bounded and synthetic. |

## Security

- Treat all context-build inputs as untrusted diagnostics. Validate GPU ids,
  layer ids, descriptor spans, relay sizes, and output buffers before any
  device allocation or copy.
- Keep shard, GGUF, and pack-index inputs read-only. Sprint 006 should not
  mutate model artifacts or produce rewritten shard payloads.
- Do not expose the new context or relay smoke through `ds4-server` or any
  network API in this sprint.
- Do not export raw device pointers or create general-purpose memory-inspection
  interfaces. Reports and bounded diagnostics are sufficient.
- Preserve the source-layout generation guard so successful context bring-up
  cannot be misread as supported decode.

## Dependencies

- Sprint 004 runtime pack loading, reconciliation, shard validation, and device
  residency APIs.
- Sprint 005 BF16 gather/expand probe and the existing `ds4_gpu_arena`
  diagnostic contract.
- `docs/architecture/DS4-V100-LAYOUT.md` as the baseline topology, dtype, and
  relay-shape anchor.
- `pack-index.tsv` and `ds4_pack_*` helpers as the source of semantic tensor
  ownership and shard coordinates.
- V100 cluster access with `CUDA_ARCH=sm_70` for real multi-GPU validation.
- Existing build/test targets in `Makefile`, plus room to add dedicated context
  smoke targets without changing normal generation semantics.

## Open Questions

1. Should the first diagnostic context entry point live in `ds4.h`, or should
   Sprint 006 keep it tool-private until Sprint 007 begins real decode wiring?
2. Should the no-math skeleton require full descriptor coverage for every
   tensor family in every layer, or is one representative family per layer
   class enough for `SHIP`?
3. Should the relay smoke require all seven inter-stage boundaries on the real
   cluster, or is one representative boundary plus a full peer matrix
   sufficient for this sprint?
4. How much future output-head and MTP reserve should be carved out on `gpu7`
   during context initialization, even though neither path is enabled yet?
5. Should real cluster smoke always allocate the full weight arenas again, or
   should Sprint 006 allow a bounded "context plus selected spans" mode when
   the goal is topology and relay validation rather than another full-residency
   proof?
