---
sprint: 001
title: Baseline DS4 V100 Appliance Planner And Source Inventory
status: completed
date: 2026-05-17
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
cluster_reference: /Users/ravi/repos/deepseek/docs/sprints/SPRINT-026-CLUSTER-TESTING.md
archived_prior_plan: drafts/SPRINT-001-OLD-DRIFTED.md
merge_notes: drafts/SPRINT-001-MERGE-NOTES.md
---

# SPRINT-001: Baseline DS4 V100 Appliance Planner And Source Inventory

## Overview

This sprint resets Sprint 001 around the baseline we now understand: build the
planning and inventory foundation for a DeepSeek V4 Flash appliance on the
homelab 8x V100-SXM2-32GB host, without prematurely committing to a full
multi-GPU decode implementation.

The source of truth for layout, sharding, dtype expectations, dimensions,
memory estimates, and first kernel-family choices is
[`docs/architecture/DS4-V100-LAYOUT.md`](../architecture/DS4-V100-LAYOUT.md).
If execution discovers that the real model inventory contradicts that document,
update the architecture document first and then update this sprint plan.

Sprint 001 should answer a narrower question:

Can we identify the exact high-intelligence quantized DSv4 source model, confirm
its tensor dtypes and dimensions, map it onto the V100 appliance layout without
overfilling VRAM, and produce a concrete pack/runtime contract that is ready for
implementation?

This sprint is successful even if it does not run model tokens yet. The output
should be a verified model inventory, a deterministic 8-GPU memory planner, and
a short implementation report that tells us whether the baseline layer-sharded
appliance is feasible before we start invasive CUDA/runtime work.

## Outcome Contract

- `SHIP`: exact model inventory and planner prove the baseline architecture can
  fit pure device resident across 8x V100 with declared reserve, and the repo
  contains a reproducible planner/diagnostic artifact.
- `EXTEND`: model inventory and architecture alignment are done, but a bounded
  loader/type-table or planner implementation issue needs one more sprint.
- `STOP`: the source model is unavailable, the source tensor layout contradicts
  the DS4 appliance assumptions in a material way, or the memory planner shows
  pure VRAM residency is not credible.

## Non-Goals

- No full decode GO bar in Sprint 001.
- No broad CUDA graph rewrite.
- No MTP/speculative decoding.
- No server batching or multi-slot scheduler implementation.
- No tensor-parallel runtime implementation.
- No LM-head split.
- No broad TurboMind or tc-grid kernel import.
- No persistent dequantized weight buffers.
- No SSD, host-backed, or managed-memory default path.
- No `VISION.md` until feasibility is understood.

## Planning Inputs

| File | Role |
|---|---|
| `docs/architecture/DS4-V100-LAYOUT.md` | Authoritative baseline for sharding, memory layout, dtype expectations, tensor dimensions, and kernel-family choices |
| `/Users/ravi/repos/deepseek/docs/sprints/SPRINT-026-CLUSTER-TESTING.md` | Operational guide for using the 8x V100 cluster test pod |
| `docs/sprints/drafts/SPRINT-001-OLD-DRIFTED.md` | Archived broader plan; useful context, not current sprint scope |
| `docs/sprints/SPRINT-001-DEFERRED.md` | Deferred performance/runtime work to revisit after this sprint |
| `/Users/ravi/repos/deepseek/docs/sprints/SPRINT-025-DS4-EVAL.md` | Prior DS4 evaluation context and kernel learnings |

## Use Cases

1. **Exact Source Model Inventory**: identify the canonical DSv4 Flash model
   file by path, size, hash, tensor count, tensor names, dimensions, and GGML
   type IDs.
2. **Architecture Alignment**: compare the real model inventory to
   `DS4-V100-LAYOUT.md`, including native/source dtype, tensor dimensions, and
   estimated resident bytes.
3. **No-Overfill VRAM Planning**: produce an 8-GPU layer plan with weights, KV,
   scratch, relay, output-head, MTP-off, and reserve/headroom accounting.
4. **Format Contract**: decide which tensor families stay source-faithful
   FP8/MXFP4/BF16, which are F32 control/cache tensors, and where INT8 remains
   a candidate instead of a default.
5. **Implementation Readiness**: leave a concrete planner CLI/report that can
   guide the next sprint's loader, packer, and CUDA ownership changes.

## Architecture

### Source Of Truth

`docs/architecture/DS4-V100-LAYOUT.md` is part of this sprint's contract. The
sprint report must cite the architecture revision used and include an explicit
"architecture deltas" section:

- no deltas found;
- inventory confirmed and estimates updated; or
- a material mismatch was found and the sprint closes `EXTEND` or `STOP`.

Do not duplicate the full layer table inside this sprint plan. The table in the
architecture document is the maintained copy.

### Baseline Topology

Start with the 8-GPU contiguous layer-sharded layout from the architecture doc:

| GPU | Layers | Global Ownership |
|---:|---|---|
| gpu0 | 0-5 | token embedding, layers 0-5 weights/KV |
| gpu1 | 6-11 | layers 6-11 weights/KV |
| gpu2 | 12-17 | layers 12-17 weights/KV |
| gpu3 | 18-23 | layers 18-23 weights/KV |
| gpu4 | 24-29 | layers 24-29 weights/KV |
| gpu5 | 30-34 | layers 30-34 weights/KV |
| gpu6 | 35-39 | layers 35-39 weights/KV |
| gpu7 | 40-42 | layers 40-42 weights/KV, output head |

This layout keeps the cross-GPU payload to hidden context only and makes the
first planner understandable. Tensor-parallel variants remain evaluation
targets after the baseline planner can quantify memory and communication.

### Runtime Format Stance

Use the format stance from `DS4-V100-LAYOUT.md`:

- dense attention and output projections: expected source FP8
  `F8_E4M3_B128`, runtime packed for V100-friendly FP8-dequant plus FP16 HMMA
  or a validated INT8 candidate later;
- routed experts: expected source MXFP4/FP4 expert weights, runtime source
  MXFP4 grouped pack first, with INT4/INT8/FP8 alternatives only after
  calibrated quality and memory checks;
- embedding and output head: BF16 source-faithful first unless a later quality
  gate approves FP8/Q8/INT8 or vocab tensor parallelism;
- router, norms, HC control, compressor/indexer control tensors: F32/BF16/I32
  as inventory dictates;
- activations: FP16 first;
- KV cache: F16 first, F8 later only after correctness and bandwidth work;
- no persistent dequantized weight copies in VRAM.

Sprint 001 should distinguish source dtype from runtime layout in every report
table. The goal is to minimize casting and data movement, not to force one
quantization format everywhere.

### Memory And KV Planning

Use the corrected DeepSeek4 KV schedule:

```text
kv_size = n_swa + (ratio ? n_ctx_seq / ratio : 0)
n_swa = 128
layers 0-1: ratio 0, SWA-only
even layers 2..42: ratio 4 with indexer
odd layers 3..41: ratio 128 without indexer
```

The planner must report admitted slots for at least these tiers:

| Context Tier | Required Report |
|---:|---|
| 128K | max slots, per-GPU KV, reserve |
| 256K | max slots, per-GPU KV, reserve |
| 512K | max slots, per-GPU KV, reserve |
| 1M | max slots, per-GPU KV, reserve |

For each GPU, report:

- packed weight bytes;
- KV bytes by layer class and slot count;
- compression-state estimate or exact allocation when known;
- scratch estimate;
- relay arena;
- embedding/output/MTP-off globals;
- CUDA/cuBLAS/allocator reserve;
- final planned headroom.

### Planner CLI Contract

The first executable artifact should be a local diagnostic, either as
`tools/ds4-v100-plan.c` or a similarly narrow helper:

```bash
ds4-v100-plan --ctx 262144 --slots 4 --gpus 8 --mtp off
```

It should print:

- architecture document path and version/date;
- layer map;
- tensor-family source dtype assumptions;
- selected first-choice kernel family per tensor group;
- per-GPU weight/KV/scratch/relay/global/reserve/headroom bytes;
- admitted max slots for 128K, 256K, 512K, and 1M contexts;
- warnings when INT8-expanded experts would exceed the reserve;
- whether the plan is `SHIP`, `EXTEND`, or `STOP` ready.

The planner can use static DS4 constants at first. It should be structured so a
later revision can consume exact inventory JSON or a pack manifest.

### Pack Manifest Contract

The planner should also define the minimal manifest fields needed by the next
sprint:

```text
semantic_tensor_id
source_name
source_dtype
source_shape
runtime_layout
owning_gpu
layer_id
kernel_family
byte_offset
byte_length
scale_offset
checksum
```

This is a contract only. Sprint 001 does not need to create full per-GPU weight
shards.

## Implementation

### Phase 0: Repository And Cluster Orientation

**Files:**

- `docs/sprints/SPRINT-001-REPORT.md`
- `/Users/ravi/repos/deepseek/docs/sprints/SPRINT-026-CLUSTER-TESTING.md`

**Tasks:**

- [ ] Confirm working repo, branch, remotes, and uncommitted docs state.
- [ ] Record the architecture document path and current file timestamp/hash in
      the sprint report.
- [ ] Confirm the cluster test workflow from `SPRINT-026-CLUSTER-TESTING.md`.
- [ ] Confirm whether the 8x V100 node is available before running any cluster
      job.

**Kill gate:**

- Stop if cluster access or source model access is blocked and cannot be
  resolved locally.

### Phase 1: Source Model Inventory

**Files:**

- `ds4.c` or a focused inventory helper, if existing tooling is insufficient
- `docs/sprints/SPRINT-001-REPORT.md`
- `docs/architecture/DS4-V100-LAYOUT.md`, only if inventory corrects it

**Tasks:**

- [ ] Identify the canonical model path, expected initially as
      `/models/DSv4-Flash-256e-fixed.gguf`.
- [ ] Record model size and SHA-256.
- [ ] Inventory tensor names, dimensions, element counts, GGML type IDs, and
      resident byte estimates.
- [ ] Group tensors by DS4 family: global, HC, attention, compressor, indexer,
      router, routed experts, shared expert, output head, MTP sidecar if
      present.
- [ ] Compare inventory against `DS4-V100-LAYOUT.md`.
- [ ] Update the architecture doc if native/source dtype, dimensions, or
      memory estimates are wrong.

**Kill gate:**

- Stop if the source model is not the expected DS4 Flash graph, or if arbitrary
  GGUF support would be required.

### Phase 2: Static V100 Memory Planner

**Files:**

- `tools/ds4-v100-plan.c` or equivalent narrow diagnostic
- `Makefile`, only if needed for the diagnostic target
- `docs/sprints/SPRINT-001-REPORT.md`

**Tasks:**

- [ ] Implement the baseline layer map from the architecture document.
- [ ] Encode DS4 layer classes: SWA-only, ratio-4 plus indexer, ratio-128.
- [ ] Encode planning bytes for F16/BF16, F32, FP8 `F8_E4M3_B128`, Q8_0,
      MXFP4, and INT8 candidate packs.
- [ ] Print per-layer and per-GPU weight estimates using the architecture
      tensor table.
- [ ] Print KV budgets using the corrected DeepSeek4 formula for 128K, 256K,
      512K, and 1M.
- [ ] Keep configured slots separate from active microbatch assumptions.
- [ ] Reject plans that exceed 32 GiB minus reserve.
- [ ] Include a conservative reserve field that can be tuned after cluster
      measurements.

**Kill gate:**

- Stop if the baseline architecture cannot fit with a credible reserve before
  any CUDA runtime work begins.

### Phase 3: Format And Kernel Policy Report

**Files:**

- `docs/sprints/SPRINT-001-REPORT.md`
- `docs/architecture/DS4-V100-LAYOUT.md`, if policy changes

**Tasks:**

- [ ] For every tensor family, record source dtype, runtime layout, kernel
      family, and fallback path.
- [ ] Mark INT8 as a candidate only where memory, scale policy, and quality
      gates are explicit.
- [ ] Confirm that no planner mode assumes persistent F16 dequantized copies
      of large packed weights.
- [ ] Record which existing TurboMind/tc-grid kernels are relevant to the next
      sprint and which are deliberately deferred.
- [ ] Preserve the MXFP4 nibble-lane mapping requirement for future tests:
      low nibble maps to `k = j`, high nibble maps to `k = j + 16`.

**Kill gate:**

- Stop if the only credible execution format requires a second large resident
  copy of expert or dense weights.

### Phase 4: Next-Sprint Implementation Contract

**Files:**

- `docs/sprints/SPRINT-001-REPORT.md`
- optional `docs/sprints/SPRINT-002-SEED.md`

**Tasks:**

- [ ] Produce the manifest schema and pack/runtime contract.
- [ ] List the minimal loader/type-table changes needed next.
- [ ] List the minimal CUDA ownership changes needed next.
- [ ] List the first kernel integration target by tensor family.
- [ ] Record whether Sprint 002 should start with loader/packer, per-device
      CUDA state, or first-kernel execution.

**Kill gate:**

- Stop if Sprint 002 cannot be bounded to a concrete implementation surface.

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `docs/architecture/DS4-V100-LAYOUT.md` | Read, update only if inventory proves a mismatch | Planning source of truth for topology, dtype, dimensions, memory, kernels |
| `docs/sprints/SPRINT-001.md` | Replace | Current scoped sprint plan |
| `docs/sprints/drafts/SPRINT-001-OLD-DRIFTED.md` | Create | Preserve broader superseded plan |
| `docs/sprints/SPRINT-001-REPORT.md` | Create during execution | Evidence, commands, inventory, planner output, verdict |
| `tools/ds4-v100-plan.c` | Create if C helper is chosen | Static planner and admission-control diagnostic |
| `Makefile` | Modify if needed | Build planner or inventory helper |
| `ds4.c` | Inspect or minimally modify if inventory tooling is missing | GGUF type/name/dimension inventory |
| `docs/sprints/SPRINT-002-SEED.md` | Optional | Seed next implementation sprint from actual findings |

## Definition Of Done

- [ ] `DS4-V100-LAYOUT.md` is explicitly cited in the sprint report as the
      planning baseline.
- [ ] The target source model is identified by path, size, and SHA-256.
- [ ] Tensor inventory includes names, dimensions, GGML type IDs, source dtype,
      grouped tensor family, and byte estimates.
- [ ] Inventory is reconciled against `DS4-V100-LAYOUT.md`; any architecture
      deltas are documented and patched.
- [ ] Planner prints the baseline 8-GPU layer map from the architecture doc.
- [ ] Planner prints per-GPU weight, KV, scratch, relay, global, reserve, and
      headroom estimates.
- [ ] Planner reports admitted slots for 128K, 256K, 512K, and 1M context tiers.
- [ ] Planner distinguishes source dtype from runtime layout and marks INT8 as
      candidate, not assumed default.
- [ ] Planner rejects configurations that overfill 32 GiB V100 VRAM after
      reserve.
- [ ] Report records whether the baseline closes `SHIP`, `EXTEND`, or `STOP`.
- [ ] Next-sprint implementation surface is concrete enough to begin coding
      without reopening the topology/dtype discussion.

## Risks And Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| Architecture doc is stale relative to actual model tensors | Medium | High | Phase 1 inventory must reconcile every tensor family before planner conclusions |
| Planner uses optimistic bytes and later overfills VRAM | Medium | High | Include reserve, compression-state envelope, CUDA overhead, and reject-on-overfill behavior |
| INT8 is treated as a blanket conversion path | Medium | High | Mark INT8 candidate per tensor family only, with future scale and quality gates |
| Full decode work starts before format/topology are settled | Medium | High | Keep Sprint 001 scoped to inventory, planner, and contract |
| Cluster availability delays inventory | Medium | Medium | Prepare local static planner while waiting; close `EXTEND` if source access is the only blocker |
| Existing broader plan hides in docs and causes drift | Medium | Medium | Archive it as `SPRINT-001-OLD-DRIFTED.md` and make this file the current plan |
| Tensor-parallel exceptions look attractive too early | Medium | Medium | Keep TP as a follow-up evaluated only after baseline layer-sharded memory plan |

## Security Considerations

- Keep DS4 model loading narrow. Do not relax fixed graph validation to load
  arbitrary GGUFs.
- Do not expose local model paths or hashes through any server API in this
  sprint.
- Bounds-check all model inventory parsing and planner arithmetic.
- Treat host/SSD fallback as diagnostic only, not a success path.

## Dependencies

- Private repo `rapatel0/ds4`.
- 8x V100-SXM2-32GB node and CUDA sm70-capable build environment.
- Canonical DSv4 Flash quantized source model, expected initially at
  `/models/DSv4-Flash-256e-fixed.gguf`.
- Cluster operating procedure in
  `/Users/ravi/repos/deepseek/docs/sprints/SPRINT-026-CLUSTER-TESTING.md`.
- Prior kernel and measurement context in `/Users/ravi/repos/deepseek`.

## Open Questions

- What exact model file should become canonical after inventory: the fixed GGUF
  path above or a different high-intelligence DSv4 Flash artifact?
- What reserve should the first planner enforce per V100: 3 GiB, 4 GiB, or a
  measured value from the cluster?
- Should Sprint 002 start with source loader/type-table support, offline packer
  manifest generation, or per-device CUDA ownership?
- Which first execution path should be the kernel proving ground after the
  planner: routed expert MXFP4, routed expert INT8 candidate, dense FP8, or
  output head?
