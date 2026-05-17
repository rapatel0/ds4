---
sprint: 004
title: Runtime Pack Loading And V100 Device Residency Smoke
status: completed
date: 2026-05-17
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
cluster_reference: /Users/ravi/repos/deepseek/docs/sprints/SPRINT-026-CLUSTER-TESTING.md
merge_notes: drafts/SPRINT-004-MERGE-NOTES.md
deferred: SPRINT-004-DEFERRED.md
---

# SPRINT-004: Runtime Pack Loading And V100 Device Residency Smoke

## Overview

Sprint 003 proved that the DS4 V100 source-layout manifest can produce a
deterministic `pack-index.tsv` and optional `gpuN.weights` shard files. The
runtime still does not consume that pack contract, and the source model remains
correctly guarded from decode.

Sprint 004 moves one step forward: the runtime learns to validate
`pack-index.tsv` against the loaded source GGUF, resolve every source tensor
to a physical provider, and upload source-faithful packed bytes into
per-GPU device arenas on the 8x V100-SXM2-32GB host. This is a structural
residency sprint, not a decode sprint.

The sprint is successful when the repo can prove:

1. the pack index matches the real model tensor binding exactly;
2. emitted shards are complete and tied to the same source model;
3. each planned tensor is resident on its owning GPU as packed source bytes;
4. the device residency report shows no 32 GB VRAM overfill with declared
   reserve;
5. normal source-model generation remains disabled.

## Outcome Contract

- `SHIP`: pack-index reconciliation passes for the real model; full
  real-model shards are emitted to persistent scratch; both GGUF and shard
  providers upload all planned bytes into 8 V100 device arenas; artifact logs
  prove logical bytes, shard hashes, spot checks, and VRAM headroom.
- `EXTEND`: local parser, reconciliation, arena sidecar, and synthetic smoke
  land, but cluster shard emission or full residency smoke is blocked by
  infrastructure. The blocker is recorded concretely.
- `STOP`: source metadata contradicts the pack manifest in dtype, shape,
  offset, byte length, or tensor count; or planned device residency exceeds
  32 GiB minus reserve on the real V100 node.

## Non-Goals

- No source-model decode enablement.
- No MTP or speculative decoding.
- No tensor-parallel runtime or LM-head split.
- No broad `ds4_gpu_context[8]` execution refactor.
- No HC relay or multi-device layer scheduler.
- No source-format math probe, including BF16 embedding or FP8/MXFP4 GEMM.
- No persistent F16/F32 dequantized weight copies.
- No managed-memory, SSD, or host-backed fallback for a successful residency
  verdict.
- No server/API exposure of pack artifacts.

## Planning Inputs

| File | Role |
|---|---|
| `docs/architecture/DS4-V100-LAYOUT.md` | Baseline topology, tensor families, source dtype expectations, memory layout, and kernel-family intent |
| `docs/sprints/drafts/SPRINT-002-PACK-MANIFEST.tsv` | Source-to-owner mapping produced by source inventory work |
| `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv` | Current pack-index schema and dry-run shard plan |
| `tools/ds4-v100-pack.c` | Existing offline packer and shard emitter |
| `ds4.c`, `ds4.h` | GGUF loader, source-layout binding, generation guard, engine options |
| `ds4_gpu.h`, `ds4_cuda.cu` | CUDA backend surface and existing single-runtime CUDA globals |
| `/Users/ravi/repos/deepseek/docs/sprints/SPRINT-026-CLUSTER-TESTING.md` | V100 cluster operating procedure |

## Use Cases

1. **Pack Validation**: as the operator, I can point the runtime at a source
   GGUF and `pack-index.tsv` and get a fail-closed verdict if the pack is
   stale, malformed, incomplete, or not from the same model.
2. **Tensor Resolution**: as the runtime, every source-layout tensor resolves
   to `(owning_gpu, provider, offset, byte_length, source_dtype,
   runtime_layout, kernel_family)`.
3. **Device Residency Proof**: as the operator, I can run a smoke tool that
   uploads all planned packed bytes into per-GPU arenas and reports logical
   bytes, actual VRAM headroom, residency class, and spot-check results.
4. **Shard Artifact Proof**: as the next sprint, I can reuse shard sizes,
   shard hashes, reconcile logs, and residency logs without redoing pack
   plumbing.
5. **Guarded Bring-Up**: as the maintainer, I can land runtime pack plumbing
   without weakening the source-model generation guard.

## Architecture

### Source Of Truth

`docs/architecture/DS4-V100-LAYOUT.md` remains the architectural baseline.
`SPRINT-002-PACK-MANIFEST.tsv` and `SPRINT-003-PACK-INDEX.tsv` are treated as
derived contracts. If the real source GGUF contradicts the pack plan, Sprint
004 stops and the manifest/layout must be corrected before runtime work
continues.

### Module Boundaries

Add two narrow pieces of runtime infrastructure:

```text
ds4_pack.h / ds4_pack.c
    pack-index parser
    tensor lookup
    source GGUF reconciliation
    provider/shard validation helpers

ds4_gpu_arena_* in ds4_gpu.h / ds4_cuda.cu
    upload-only per-device arena API
    device-memory allocation
    upload/readback spot checks
    per-GPU memory and residency reporting
```

`ds4.c` remains the authority for GGUF metadata and source-layout binding, but
TSV parsing and reconciliation logic live in `ds4_pack.c` so they can be tested
without growing the already-large loader.

The existing CUDA globals in `ds4_cuda.cu` are not converted into a production
multi-device execution context in this sprint. The arena API is a sidecar for
residency smoke only.

### Pack-Index Contract

The reader targets the exact Sprint 003 schema:

```text
semantic_tensor_id
source_name
source_dtype
source_shape
runtime_layout
owning_gpu
layer_id
kernel_family
source_offset
byte_length
shard_file
shard_offset
scale_offset
checksum
```

Parser rules:

- validate the header by name and expected order;
- reject BOM, CRLF, wrong column count, embedded tabs, malformed numeric
  fields, overflow, duplicate `semantic_tensor_id`, and duplicate
  `source_name`;
- allow legitimate control/global rows such as `layer_id = -1`;
- reject `owning_gpu` outside the configured GPU count;
- validate `source_offset + byte_length` against source GGUF file size;
- validate `shard_offset + byte_length` against the shard file size when the
  shard provider is used;
- treat missing, stale, or partially emitted shard files as hard errors for
  the shard provider.

Reconciliation checks every pack row against the loaded source model:

- source tensor exists;
- source dtype matches GGUF type;
- source shape matches;
- byte length matches the source tensor payload;
- source offset is in range;
- tensor count matches the pack-index row count.

The reconcile log must contain one deterministic row per tensor with a status
such as `OK`, `MISSING_BINDING`, `DTYPE_MISMATCH`, `SHAPE_MISMATCH`,
`BYTE_LENGTH_MISMATCH`, `OFFSET_OUT_OF_RANGE`, or `BAD_OWNING_GPU`.

### Source-Byte Providers

The residency smoke supports two providers:

| Provider | Source | Purpose |
|---|---|---|
| `gguf` | `mmap_base + source_offset` from the source GGUF | Fast local and cluster iteration before full shard emission |
| `shard` | `pread` from `gpuN.weights` at `shard_offset` | End-to-end validation of emitted per-GPU shards |

Both providers must produce identical bytes for any tensor. The cluster run
performs one deterministic cross-provider compare, using a non-edge tensor if
available, and records the result.

### Device Residency Model

The arena path allocates one weight arena per owning GPU. Uploads are
bit-for-bit copies of source-faithful packed bytes. There is no persistent
expanded F16/F32 mirror.

The CUDA implementation must:

- call `cudaSetDevice(arena->gpu)` around each allocation, upload, readback,
  and free;
- use `cudaMalloc` for successful residency, not managed memory;
- call `cudaPointerGetAttributes` or equivalent to report device residency;
- reject uploads whose `offset + bytes` exceeds the arena size;
- support chunked uploads for large tensors so host buffers do not require a
  multi-GiB allocation;
- invalidate and free an arena after a partial upload failure;
- report `cudaMemGetInfo` before allocation, after allocation, and after
  upload;
- optionally call `cudaDeviceReset` at smoke startup or record why it was not
  used.

### Topology And P2P Reporting

The real cluster verdict requires exactly 8 visible V100-class devices with
at least 32 GB total memory each. If `CUDA_VISIBLE_DEVICES` remaps device
numbers, the smoke reports both visible device IDs and PCI bus IDs so the pack
plan can be interpreted correctly.

Local synthetic tests may run with zero GPUs through the CPU/stub path, but
they cannot produce a real residency verdict.

The smoke reports the `cudaDeviceCanAccessPeer` matrix as planning signal for
the HC relay sprint. Sprint 004 does not require enabling peer access and does
not implement inter-GPU transfer.

### Memory Validation

The plan separates two measurements:

- **Logical arena bytes**: sum of pack-index `byte_length` values per owning
  GPU. This must match the planner and the shard sizes exactly.
- **Observed CUDA memory facts**: `cudaMemGetInfo` before and after allocation
  and upload. These values are used to prove headroom and catch overfill, but
  they are not expected to match logical bytes exactly because CUDA context
  and allocator overhead are real.

Default reserve is 3 GiB per GPU. A successful real residency run must show
each GPU has enough total memory for logical arena bytes plus reserve. If any
GPU is within 256 MiB of the 32 GiB ceiling after applying reserve, the sprint
returns `STOP` and the architecture must be revisited.

## Implementation

### Phase 0: Orientation And Build Hygiene

**Files:**

- `docs/sprints/SPRINT-004-REPORT.md`
- `Makefile`

**Tasks:**

- [ ] Confirm repo status and ignore unrelated untracked files.
- [ ] Confirm `make cpu` and the existing `tools/ds4-v100-pack` target build.
- [ ] Re-read the cluster procedure in
      `/Users/ravi/repos/deepseek/docs/sprints/SPRINT-026-CLUSTER-TESTING.md`.
- [ ] Confirm persistent scratch path with at least 160 GiB free.
- [ ] Confirm the V100 pod image/toolchain can build CUDA for `sm_70`.
- [ ] Record source model path, pack index path, scratch path, CUDA version,
      driver version, and visible GPU list in the report.

**Kill gate:** EXTEND if persistent scratch or V100 node access is blocked and
cannot be resolved within the sprint.

### Phase 1: Pack-Index Reader And Reconciliation

**Files:**

- `ds4_pack.h` (new)
- `ds4_pack.c` (new)
- `ds4.h`
- `ds4.c`
- `Makefile`
- `tests/pack_index_smoke.c` (new)

**Tasks:**

- [ ] Add `ds4_pack_open`, `ds4_pack_close`, lookup/iteration helpers, and
      per-GPU aggregate byte/count helpers.
- [ ] Add reconciliation against source GGUF tensor binding and emit a
      deterministic row-per-tensor report.
- [ ] Add `pack_index_path` to `ds4_engine_options` or an equivalently narrow
      internal option used by `--inspect`.
- [ ] Keep the source-model decode guard active for normal generation.
- [ ] Unit-test happy path, duplicate rows, malformed fields, dtype mismatch,
      shape mismatch, byte-length mismatch, bad offsets, bad GPU ownership,
      `layer_id = -1`, and missing rows.
- [ ] Add `git diff --check` and parser tests to the local verification path.

**Kill gate:** STOP if real-model reconciliation exposes a material manifest
or source-model mismatch that is not a parser bug.

### Phase 2: Upload-Only Per-GPU Arena Sidecar

**Files:**

- `ds4_gpu.h`
- `ds4_cuda.cu`
- `Makefile`
- `tests/gpu_arena_smoke.c` (new)

**Tasks:**

- [ ] Add `ds4_gpu_arena_open`, `ds4_gpu_arena_upload`,
      `ds4_gpu_arena_read`, `ds4_gpu_arena_close`, and memory-report helpers.
- [ ] Keep the API explicitly documented as residency-only.
- [ ] Add CPU/stub behavior so non-CUDA builds can test orchestration without
      producing a real residency verdict.
- [ ] Use plain device memory for CUDA success paths.
- [ ] Validate upload offsets and chunking.
- [ ] Report residency class and fail if a successful CUDA verdict is not
      device memory.
- [ ] Handle partial upload failure by making the arena unusable and freeing
      it before exit.

**Kill gate:** STOP if the V100 node cannot allocate planned arenas with the
declared reserve.

### Phase 3: Local Synthetic Residency Smoke

**Files:**

- `tools/ds4-v100-residency-smoke.c` (new)
- `tests/residency_smoke_synthetic.sh` (new)
- `Makefile`

**Tasks:**

- [ ] Add a standalone smoke tool rather than hiding this path inside normal
      generation.
- [ ] Support `--model`, `--index`, `--shard-dir`, `--provider gguf|shard`,
      `--reserve-mib`, and `--report`.
- [ ] Exercise parser, reconciliation, provider reads, arena uploads, readback
      spot checks, memory reporting, and failure paths on a small synthetic
      fixture.
- [ ] Ensure the CPU/stub path can run locally without CUDA.

**Kill gate:** EXTEND or STOP before cluster work if the synthetic smoke cannot
pass. The cluster must not be the first end-to-end test of orchestration.

### Phase 4: Cluster Shard Emission On Persistent Scratch

**Files:**

- `docs/sprints/drafts/SPRINT-004-SHARD-SIZES.tsv`
- `docs/sprints/drafts/SPRINT-004-SHARD-SHA256.tsv`
- `docs/sprints/SPRINT-004-REPORT.md`

**Tasks:**

- [ ] Build `tools/ds4-v100-pack` on the cluster.
- [ ] Emit full real-model shards from
      `/models/DSv4-Flash-256e-fixed.gguf` to persistent scratch.
- [ ] Write a fresh `pack-index.tsv` in the same pack directory.
- [ ] Reject or clean stale `gpuN.weights` files before emission.
- [ ] Record per-shard file sizes.
- [ ] Record per-shard SHA-256.
- [ ] Compare measured shard sizes to logical per-GPU payload bytes from the
      index.
- [ ] Record any interruption or partial emission as a STOP/EXTEND artifact,
      not as a successful shard directory.

**Kill gate:** EXTEND if persistent scratch cannot hold the full pack plus
margin. Do not fall back to disposable pod storage for SHIP.

### Phase 5: Cluster Per-GPU Residency Smoke

**Files:**

- `docs/sprints/drafts/SPRINT-004-RECONCILE.log`
- `docs/sprints/drafts/SPRINT-004-RESIDENCY-GGUF.log`
- `docs/sprints/drafts/SPRINT-004-RESIDENCY-SHARD.log`
- `docs/sprints/drafts/SPRINT-004-CROSSCHECK.log`
- `docs/sprints/SPRINT-004-REPORT.md`

**Tasks:**

- [ ] Build CPU and CUDA targets on the V100 pod.
- [ ] Run reconciliation against the real model and archive the log.
- [ ] Run residency smoke with `--provider gguf`.
- [ ] Run residency smoke with `--provider shard`.
- [ ] Archive logical bytes, observed CUDA memory facts, residency class,
      visible device IDs, PCI bus IDs, and P2P matrix.
- [ ] Verify first and last 4 KiB spot checks for every tensor.
- [ ] Verify one deterministic provider cross-check.
- [ ] Confirm no normal source-model generation path was enabled.
- [ ] Delete the temporary cluster pod after copying artifacts out.

**Kill gate:** STOP if device residency cannot be proven as pure device memory,
if any GPU overfills the reserve, or if shard-provider bytes disagree with
GGUF-provider bytes.

### Phase 6: Report And Follow-Ups

**Files:**

- `docs/sprints/SPRINT-004-REPORT.md`
- `docs/sprints/SPRINT-004-FOLLOWUPS.md`

**Tasks:**

- [ ] Record final verdict: SHIP, EXTEND, or STOP.
- [ ] Link all validation artifacts.
- [ ] Record architecture deltas, or explicitly state that none were needed.
- [ ] Identify the next sprint's first source-format math probe candidate.
- [ ] Preserve source-model generation guard status in the report.

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `ds4_pack.h` | Create | Pack-index parser, lookup, aggregate, and reconciliation API |
| `ds4_pack.c` | Create | TSV parsing, validation, and source-model reconciliation |
| `ds4.h` | Modify | Add narrow pack-index option or equivalent inspect-only path |
| `ds4.c` | Modify | Wire pack reconciliation into engine open/inspect path while preserving decode guard |
| `ds4_gpu.h` | Modify | Add residency-only arena API |
| `ds4_cuda.cu` | Modify | Implement device arenas, upload/readback, residency reporting |
| `tools/ds4-v100-residency-smoke.c` | Create | Standalone diagnostic for local and cluster residency proof |
| `tools/ds4-v100-pack.c` | Modify only if needed | Fix full-emission issues found during cluster shard run |
| `Makefile` | Modify | Build new tool and tests |
| `tests/pack_index_smoke.c` | Create | Parser/reconciliation unit coverage |
| `tests/gpu_arena_smoke.c` | Create | Arena offset/chunking and CUDA device smoke coverage |
| `tests/residency_smoke_synthetic.sh` | Create | End-to-end synthetic smoke before cluster |
| `docs/sprints/SPRINT-004-REPORT.md` | Create during execution | Verdict and evidence |
| `docs/sprints/SPRINT-004-FOLLOWUPS.md` | Create during execution | Next-sprint implementation surface |
| `docs/sprints/drafts/SPRINT-004-SHARD-SIZES.tsv` | Create on cluster | Per-shard size facts |
| `docs/sprints/drafts/SPRINT-004-SHARD-SHA256.tsv` | Create on cluster | Per-shard hash facts |
| `docs/sprints/drafts/SPRINT-004-RECONCILE.log` | Create on cluster | Row-per-tensor reconciliation |
| `docs/sprints/drafts/SPRINT-004-RESIDENCY-GGUF.log` | Create on cluster | GGUF-provider residency report |
| `docs/sprints/drafts/SPRINT-004-RESIDENCY-SHARD.log` | Create on cluster | Shard-provider residency report |
| `docs/sprints/drafts/SPRINT-004-CROSSCHECK.log` | Create on cluster | Provider byte-identity check |
| `docs/architecture/DS4-V100-LAYOUT.md` | Read; modify only on STOP | Architecture baseline and reserve assumptions |

## Definition Of Done

- [ ] `ds4_pack` parses the exact Sprint 003 schema and rejects malformed,
      duplicate, stale, or out-of-range rows.
- [ ] Reconciliation emits one deterministic row per source tensor and fails
      closed on any model/pack mismatch.
- [ ] Tensor count, logical payload bytes, and per-GPU aggregate bytes match
      the pack index and planner.
- [ ] `ds4_engine_open` or the inspect path can run reconciliation without
      enabling source-model generation.
- [ ] `ds4_gpu_arena_*` is implemented as a residency-only sidecar without
      refactoring existing production kernel state.
- [ ] CUDA success paths use device memory and report residency class.
- [ ] Local parser tests, arena tests, and synthetic smoke pass before cluster
      validation.
- [ ] Full real-model shards are emitted on persistent scratch, or an EXTEND
      artifact records the concrete storage blocker.
- [ ] Cluster smoke runs both `gguf` and `shard` providers on 8 V100s.
- [ ] Per-tensor first/last 4 KiB spot checks pass.
- [ ] One deterministic cross-provider byte comparison passes.
- [ ] Per-shard sizes and SHA-256 values are recorded.
- [ ] Per-GPU logical bytes, observed CUDA memory facts, reserve, visible
      devices, PCI bus IDs, and P2P matrix are recorded.
- [ ] No GPU exceeds the 32 GiB budget after the declared reserve.
- [ ] The source-model generation guard remains active.
- [ ] Verification includes `make cpu`, the packer target, the new smoke tool,
      parser/arena/synthetic tests, and `git diff --check`.

## Risks And Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| Persistent scratch is unavailable or too small | Medium | High | Check in Phase 0; EXTEND rather than using disposable pod storage |
| Source GGUF, pack index, and shards come from different runs | Medium | High | Validate shard sizes/hashes in same directory and fail stale artifacts |
| Parser accepts drifted TSV schema | Medium | High | Validate header by name/order and exact column count |
| `CUDA_VISIBLE_DEVICES` hides or remaps GPUs | Medium | High | Report visible IDs and PCI bus IDs; require exactly 8 V100s for real verdict |
| Logical bytes and CUDA memory deltas are conflated | Medium | Medium | Report them separately; use logical bytes for planner match and CUDA facts for headroom |
| Host-mapped or managed memory creates fake residency | Medium | High | Require `cudaMalloc` device arenas and residency-class reporting |
| Partial shard emission or upload is mistaken for success | Medium | High | Clean/reject stale directories; invalidate arenas on partial upload failure |
| Arena sidecar becomes production multi-GPU API | Medium | Medium | Document residency-only boundary and defer execution context design |
| Scope creeps into math or decode | Medium | High | Keep decode guard and defer all source-format math to Sprint 005 |
| Upload I/O is slower than expected | Medium | Low | Record timing and throughput; optimize upload scheduling later |
| P2P topology surprises next sprint | Medium | Medium | Record P2P matrix now without implementing relay |

## Security

- Open model, index, and shard inputs read-only.
- Reject path traversal and unexpected shard names.
- Bounds-check all offsets and lengths before file reads or device copies.
- Use overflow-safe arithmetic for `offset + byte_length`.
- Do not expose pack artifacts through `ds4_server.c` or any network API.
- Do not mutate source GGUF, pack index, or shard files in the smoke tool.
- Keep source-model generation disabled until a later correctness harness
  explicitly enables it.

## Dependencies

- Sprint 002 source-layout manifest and source binding work.
- Sprint 003 packer schema and dry-run pack index.
- Source model at `/models/DSv4-Flash-256e-fixed.gguf`.
- Persistent cluster scratch with at least 160 GiB free.
- 8x V100-SXM2-32GB node and CUDA `sm_70` build environment.
- Existing cluster workflow in
  `/Users/ravi/repos/deepseek/docs/sprints/SPRINT-026-CLUSTER-TESTING.md`.

## Open Decisions Resolved For Execution

| Question | Sprint 004 Decision |
|---|---|
| Require full real-model shard emission? | Yes for SHIP; EXTEND if persistent scratch is blocked |
| Runtime provider? | Support both GGUF and shard providers |
| Include math probe? | No; stop at raw packed device residency |
| Multi-device refactor scope? | Minimal residency-only arenas; no production execution context |
| Validation artifact level? | Shard sizes, shard SHA-256, reconcile log, spot checks, cross-provider compare, VRAM/P2P reports |
| Reserve default? | 3 GiB per GPU, recorded and revisited if cluster data requires it |
