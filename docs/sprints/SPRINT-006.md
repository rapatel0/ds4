---
sprint: 006
title: Multi-GPU Execution Context And Layer Skeleton
status: planned
date: 2026-05-17
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-006-INTENT.md
merge_notes: drafts/SPRINT-006-MERGE-NOTES.md
deferred: SPRINT-006-DEFERRED.md
---

# SPRINT-006: Multi-GPU Execution Context And Layer Skeleton

## Overview

Sprint 005 proved one narrow resident source-dtype contract: BF16 bytes can be
gathered from a `ds4_gpu_arena` on V100 and expanded to F32 exactly for
diagnostics. Sprint 006 turns that proof into a production-shaped multi-GPU
runtime skeleton without pretending decode correctness already exists.

The deliverable is structural and diagnostic: a V100-specific sidecar context
that owns per-GPU streams, cuBLAS handles, scratch budgets, relay buffers,
typed resident tensor descriptors, execution-format policy, and the baseline
8-stage layer map from `docs/architecture/DS4-V100-LAYOUT.md`. The context must
initialize from reconciled pack metadata, print a topology/memory/policy
report, validate an HC relay boundary, and walk the no-math layer skeleton.

V100 has no native BF16, FP8, or FP4 tensor-core execution. Sprint 006 must
encode that as code and reports:

- production dense GEMMs target FP16 HMMA with FP32 accumulation;
- broad FP32 GEMMs are not an acceptable default for main model math;
- FP32 is reserved for control, reductions, norms, router metadata, debug, and
  softmax-adjacent work;
- BF16 is source/probe/explicit-conversion only;
- FP8, MXFP4, and FP4 are packed source/runtime inputs for later registered
  unpack/dequant or low-bit kernels.

This sprint does not enable normal source-model generation, decode, prefill,
KV population, routed-expert execution, output projection, MTP, server
exposure, or throughput benchmarking.

## Outcome Contract

- `SHIP`: the sidecar V100 context exists; topology, descriptor policy, memory
  reserve, relay, and guard checks pass locally and on the V100 pod; a real
  cross-stage HC relay validates device-to-device transfer; the no-math layer
  skeleton walks the planned 8-stage map; source-layout generation remains
  guarded.
- `EXTEND`: the host-side context, policy classifier, descriptors, and no-math
  skeleton land, but cluster CUDA relay or real-pack context validation is
  blocked by infrastructure. The exact missing validation is recorded.
- `STOP`: the 8x V100 topology, peer relay path, memory reserve, descriptor
  policy, or source-layout generation guard cannot be made fail-closed without
  a larger rewrite or without widening into decode.

## Non-Goals

- No source-model decode or first-token correctness.
- No prefill, KV allocation, compressed attention, indexer updates, or slot
  scheduler.
- No routed MoE, shared-expert, FP8, MXFP4, INT8, or INT4 production kernels.
- No output-head math or vocab-parallel output-head split.
- No MTP or speculative decoding.
- No tensor-parallel topology exceptions.
- No server, API, deployment, batching, health check, or throughput benchmark.
- No host-backed, managed-memory, SSD-backed, or persistent dequantized weight
  path.

## Planning Inputs

| File | Role |
|---|---|
| `docs/sprints/VISION.md` | Sprint sequence and North Star |
| `docs/architecture/DS4-V100-LAYOUT.md` | Topology, dtype, tensor family, memory, and relay anchor |
| `docs/sprints/SPRINT-004-REPORT.md` | Pack residency proof and guard status |
| `docs/sprints/SPRINT-005-REPORT.md` | Resident BF16 gather/expand proof and guard status |
| `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv` | Current semantic tensor ownership and shard coordinates |
| `ds4_pack.h`, `ds4_pack.c` | Pack metadata and lookup surface |
| `ds4_gpu.h`, `ds4_cuda.cu`, `ds4_gpu_arena_stub.c` | Existing arena and CUDA implementation surface |
| `tools/ds4-v100-residency-smoke.c` | Existing diagnostic tool pattern and cluster smoke precedent |

## Use Cases

1. **Context topology report**: an operator can build a V100 context from a
   source GGUF plus pack metadata and see stage ownership, visible device IDs,
   PCI bus IDs, peer matrix, arena bytes, scratch bytes, relay bytes, and
   reserve/headroom before any decode work starts.
2. **Typed descriptor binding**: the runtime can resolve semantic tensor ids
   into descriptors keyed by owning GPU, layer id, arena span, source dtype,
   runtime layout, and execution class.
3. **Execution policy enforcement**: BF16, FP8, MXFP4, and FP4 source tensors
   cannot silently bind to unsupported native V100 compute claims.
4. **HC relay validation**: a bounded `[active_slots][4][4096]` boundary
   payload can be transferred device-to-device across at least one real stage
   boundary with byte equality verification.
5. **No-math layer walk**: the layer skeleton can walk all planned stages and
   validate ownership, family presence, descriptor bounds, and boundary
   allocation without launching attention, MoE, KV, or output kernels.

## Architecture

Sprint 006 introduces `ds4_v100_context` as a sidecar module. It does not
rewrite the existing global CUDA path and does not make normal `ds4.c`
generation succeed.

```text
GGUF + pack-index.tsv
    |
    v
pack reconcile + source-layout guard
    |
    v
ds4_v100_context_open(...)
    |
    +--> per-stage CUDA resources
    +--> typed tensor descriptors
    +--> execution-format policy
    +--> topology / peer / memory report
    |
    v
tools/ds4-v100-context-smoke
    |
    +--> policy report
    +--> HC relay smoke
    +--> no-math layer skeleton walk
    +--> guard regression check
```

### Module Boundary

`ds4_v100_context.h` is the public boundary for the diagnostic tool. It should
not be re-exported through `ds4.h` unless implementation proves a narrow
inspect-only need. Normal decode wiring remains deferred.

The new context owns:

- one global context record with topology, pack provenance, options, and report
  counters;
- one stage record per owning GPU with visible ordinal, PCI bus id, UUID if
  available, layer range, borrowed arena pointer, scratch budget, stream,
  cuBLAS handle, relay buffers, and peer facts;
- typed resident descriptors keyed by semantic tensor id;
- a no-math layer descriptor table for SWA-only, ratio-4, and ratio-128 layer
  classes.

### Baseline Layer Map

The initial map remains the architecture-doc layer-sharded baseline:

| Stage | GPU | Layers | Global Ownership |
|---:|---:|---|---|
| 0 | `gpu0` | 0-5 | token embedding |
| 1 | `gpu1` | 6-11 | none |
| 2 | `gpu2` | 12-17 | none |
| 3 | `gpu3` | 18-23 | none |
| 4 | `gpu4` | 24-29 | none |
| 5 | `gpu5` | 30-34 | none |
| 6 | `gpu6` | 35-39 | none |
| 7 | `gpu7` | 40-42 | output-head reserve and MTP reserve placeholder |

Sprint 006 may not silently replan this topology. A mismatch with
`DS4-V100-LAYOUT.md` is a STOP.

### Execution-Format Policy

The policy must be encoded as data, not comments. A minimal enum shape is:

```c
typedef enum {
    DS4_V100_EXEC_F32_CONTROL,
    DS4_V100_EXEC_F16_HMMA,
    DS4_V100_EXEC_LOWBIT_KERNEL,
    DS4_V100_EXEC_DIAGNOSTIC_ONLY,
    DS4_V100_EXEC_UNSUPPORTED,
} ds4_v100_exec_kind;
```

The context report must emit counts by execution class and must include this
policy table:

| Tensor Family | Source Format | Sprint 006 Operation | V100 Execution Target | Forbidden Claim |
|---|---|---|---|---|
| norms, router metadata, HC control, small globals | F32 / I32 / selected BF16 | descriptor binding, report output, small diagnostics | F32 control/reduction | decode completion |
| token embedding and selected BF16 families | BF16 | exact diagnostic gather/expand and descriptor binding | explicit later conversion to FP16 data path or F32 control path | native BF16 tensor-core execution |
| dense attention and shared-expert packs | F8_E4M3_B128 | descriptor binding only | later unpack/dequant feeding FP16 HMMA or registered low-bit kernel | native FP8 tensor-core execution |
| routed expert packs | MXFP4 / FP4 | descriptor binding only | later grouped low-bit kernel or bounded unpack-to-FP16 tile path | native FP4 tensor-core execution |
| activations and HC relay payloads | runtime values | synthetic relay allocation/copy | FP16 normal, FP32 debug | full layer compute in Sprint 006 |
| KV cache | runtime cache | reserve/report fields only | F16 first, later validated alternatives | real prefill/decode KV population |

If a non-FP16 source dtype is classified toward `DS4_V100_EXEC_F16_HMMA`, the
descriptor must carry an explicit future conversion-stub name. The kernel does
not have to exist in this sprint, but the policy slot must be visible.

### Descriptor Scope

Sprint 006 validates metadata that proves the contract; it does not bind every
execution call site. The ship bar is:

- global embedding descriptor;
- representative F32/control descriptors;
- representative FP8 and MXFP4 families as descriptors only;
- HC control tensor family descriptors;
- at least one representative full layer row set;
- all 43 layer ownership records and all three layer classes covered by the
  no-math skeleton.

Binding must fail closed on missing pack entries, wrong source dtype, wrong
owning GPU, duplicate semantic tensor ids, bad layer id, arena span overflow,
byte-length/shape contradiction, or policy/layout mismatch.

### Relay Contract

The HC relay payload shape is `[active_slots][4][4096]`.

- FP16 is the normal relay mode.
- FP32 is a debug relay mode and must be reported as such.
- Relay buffers are double-buffered for future scheduler overlap.
- Same-GPU loopback is valid only for local synthetic tests.
- Real multi-GPU validation must use device-to-device peer copy on at least one
  stage boundary and emit the full peer matrix.
- Host-pinned relay may not make a production V100 topology smoke pass.

### Memory Contract

The context report must account for:

- borrowed weight arena bytes;
- scratch allocation;
- double-buffered FP16 relay bytes;
- optional FP32 debug relay bytes;
- cuBLAS/CUDA overhead estimate or measured delta;
- planned KV reservation field;
- `gpu7` output-head reserve and small MTP reserve placeholder.

The default reserve floor is 2 GiB per GPU. The context must fail closed if any
stage falls below the declared reserve after context allocations and planned
reservation fields are included.

## Implementation

### Phase 0: Baseline Hygiene

**Files:**
- no source edits expected

**Tasks:**
- [ ] Confirm local Sprint 005 build/test targets still pass.
- [ ] Confirm no unrelated worktree changes are required for Sprint 006.
- [ ] Record any existing unrelated failures before editing.

### Phase 1: Context Contract, Topology, And Policy

**Files:**
- `ds4_v100_context.h`
- `ds4_v100_context.c`
- `tests/v100_context_smoke.c`
- `Makefile`

**Tasks:**
- [ ] Define opaque `ds4_v100_context`, stage, descriptor, layer class, source
      dtype, and execution-class types.
- [ ] Implement `ds4_v100_classify_or_die()` for source dtype plus tensor
      family policy.
- [ ] Encode the 8-stage layer map from `DS4-V100-LAYOUT.md`.
- [ ] Emit a topology and policy report.
- [ ] Fail closed unless the production mode sees exactly eight visible V100
      devices with compute capability 7.0 and 32 GB-class VRAM.
- [ ] Emit visible device id, PCI bus id, stage id, and UUID when available.
- [ ] Fail closed if stage-adjacent peer edges cannot be enabled.
- [ ] Add model-less tests for layer map, policy classifier, invalid topology,
      and unsupported source-dtype/family pairs.

### Phase 2: Descriptor Binding

**Files:**
- `ds4_v100_context.c`
- `ds4_pack.h`
- `ds4_pack.c`
- `tests/v100_context_smoke.c`

**Tasks:**
- [ ] Build typed descriptors from validated pack rows.
- [ ] Validate semantic id, owning GPU, layer id, source dtype, runtime layout,
      arena offset, byte length, and shard offset.
- [ ] Reconcile descriptor byte length against declared shape and source format
      where the format is known.
- [ ] Reject duplicate descriptor ownership and arena span overflow.
- [ ] Cover the descriptor scope listed above, including SWA-only, ratio-4, and
      ratio-128 layer classes.
- [ ] Keep pack metadata read-only.

### Phase 3: CUDA Resource Ownership And HC Relay

**Files:**
- `ds4_v100_context_cuda.cu`
- `ds4_v100_context.h`
- `tests/cuda_v100_context_smoke.c`
- `tests/cuda_hc_relay_smoke.c`
- `Makefile`

**Tasks:**
- [ ] Add per-stage CUDA resource ownership for default stream, relay stream,
      cuBLAS handle, scratch allocation, relay buffers, and peer facts.
- [ ] Keep legacy globals such as `g_cublas` and `g_cuda_tmp` out of the new
      context contract.
- [ ] Implement FP16 normal relay buffer allocation for
      `[2][active_slots][4][4096]`.
- [ ] Implement explicit FP32 debug relay allocation/copy mode.
- [ ] Validate `active_slots`, GPU ids, buffer sizes, transfer mode, and
      boundary direction.
- [ ] Run same-device loopback synthetic tests where only one GPU is visible.
- [ ] Run at least one real cross-stage device-to-device relay on the V100 pod.

### Phase 4: No-Math Layer Skeleton And Tooling

**Files:**
- `tools/ds4-v100-context-smoke.c`
- `tests/v100_context_smoke.c`
- `Makefile`

**Tasks:**
- [ ] Add a standalone smoke tool that opens the context, prints reports, runs
      relay smoke, performs the no-math layer walk, and exits nonzero on any
      fail-closed condition.
- [ ] Walk all 43 layers in stage order and validate ownership plus required
      descriptor family classes.
- [ ] Do not link or call attention, MoE, KV, output-head, MTP, or server
      execution paths.
- [ ] Optionally reuse the Sprint 005 BF16 probe as a descriptor-backed spot
      check only.
- [ ] Support `probe-only` and `use-existing-arenas` validation modes.

### Phase 5: Cluster Validation And Close-Out

**Files:**
- `docs/sprints/drafts/SPRINT-006-CONTEXT-SMOKE.log`
- `docs/sprints/drafts/SPRINT-006-RELAY.log`
- `docs/sprints/drafts/SPRINT-006-GUARD.log`
- `docs/sprints/SPRINT-006-REPORT.md`
- `docs/sprints/VISION.md`

**Tasks:**
- [ ] Build with `CUDA_ARCH=sm_70`.
- [ ] Run model-less context tests locally.
- [ ] Run CUDA synthetic context and relay tests on V100.
- [ ] Run real-pack context smoke on the 8x V100 cluster.
- [ ] Archive topology, memory, peer, policy, relay, and layer-walk output.
- [ ] After successful context smoke, run the normal source-layout generation
      path and archive proof that the guard still rejects generation.
- [ ] Run `git diff --check` and record final validation.

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `ds4_v100_context.h` | Create | V100 context, descriptor, policy, and tool-facing API |
| `ds4_v100_context.c` | Create | Host-side topology, descriptor binding, policy, reporting, and skeleton walk |
| `ds4_v100_context_cuda.cu` | Create | Per-stage CUDA resources, peer facts, scratch, and HC relay |
| `ds4_pack.h` / `ds4_pack.c` | Modify if needed | Descriptor-oriented helpers over existing pack rows |
| `tools/ds4-v100-context-smoke.c` | Create | Context report, relay smoke, no-math layer walk, guard check |
| `tests/v100_context_smoke.c` | Create | Model-less topology, policy, descriptor, and invalid-config checks |
| `tests/cuda_v100_context_smoke.c` | Create | CUDA context allocation and resource ownership checks |
| `tests/cuda_hc_relay_smoke.c` | Create | FP16/FP32 HC relay synthetic and V100 peer-copy checks |
| `Makefile` | Modify | Build new tool and tests |
| `docs/sprints/SPRINT-006-REPORT.md` | Create during execute | Final result and validation summary |
| `docs/sprints/drafts/SPRINT-006-*.log` | Create during execute | Cluster and guard validation artifacts |

## Definition Of Done

- [ ] `ds4_v100_context` exists as a sidecar context and does not widen normal
      decode/generation APIs.
- [ ] V100 execution-format policy is encoded in code and emitted in reports.
- [ ] Reports explicitly say BF16, FP8, and FP4 are not native V100 tensor-core
      execution formats.
- [ ] Broad FP32 GEMM fallback is not introduced.
- [ ] Topology preflight validates 8x V100, CC 7.0, 32 GB-class VRAM, visible
      device mapping, PCI bus id mapping, and stage-adjacent peer edges.
- [ ] Descriptor binding fails closed on missing, duplicate, inconsistent, or
      out-of-range pack metadata.
- [ ] Memory reserve fails closed below the declared per-GPU floor.
- [ ] FP16 HC relay and FP32 debug relay are validated with bounded payloads.
- [ ] Real V100 validation includes a device-to-device cross-stage relay.
- [ ] The no-math layer skeleton walks all 43 layers and all three layer classes
      without calling decode kernels.
- [ ] Source-layout generation remains guarded after a successful context smoke.
- [ ] Local and V100 validation artifacts are archived and `git diff --check`
      passes.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| Sprint expands into decode, attention, or MoE work | High | High | Keep only context, relay, descriptors, and no-math skeleton in scope |
| Numeric policy stays ambiguous | High | High | Encode execution class and forbidden claims in code and reports |
| Context tangles with legacy global CUDA state | Medium | High | Use sidecar `ds4_v100_context` and standalone smoke tool |
| Descriptor binding validates presence of wrong bytes | Medium | High | Reconcile dtype, shape, byte length, owner, and arena span |
| Smoke succeeds on degraded topology | Medium | High | Require 8x V100 topology, peer edges, and reserve floor in production mode |
| Relay silently falls back to host | Medium | High | Require device-to-device relay for real multi-GPU validation |
| Scratch/relay reserve overfills HBM | Medium | High | Report memory components and fail below reserve floor |
| Guard regresses after context bring-up | Medium | High | Run positive guard rejection test after context smoke |

## Security

- Treat GGUF, pack index, shard metadata, and context options as untrusted
  inputs.
- Validate all GPU ids, layer ids, offsets, byte lengths, descriptor counts,
  relay sizes, and output buffer lengths before allocation or copy.
- Keep model and pack artifacts read-only.
- Do not expose device pointers or generic memory-inspection APIs.
- Do not expose context smoke through `ds4-server` or any network surface.
- Preserve the source-layout generation guard.

## Dependencies

- Sprint 004 pack loading, reconciliation, shard validation, and device
  residency APIs.
- Sprint 005 BF16 gather/expand probe and arena-view diagnostics.
- `docs/architecture/DS4-V100-LAYOUT.md` for baseline topology and dtype
  policy.
- Current pack index and shard coordinate helpers.
- V100 cluster access with `CUDA_ARCH=sm_70`.

## Open Questions

1. What exact reserve floor should become the long-term default after Sprint
   006 measurements: 2 GiB, 3 GiB, or 4 GiB per GPU?
2. Should UUID matching become mandatory in all cluster modes, or only in
   production mode when persistent resident arenas are reused?
3. Which descriptor family should be the first full vertical layer row set for
   Sprint 006: a ratio-4 layer, because it is the long-context-heavy class, or
   a stage-0 layer, because it includes the embedding boundary?
