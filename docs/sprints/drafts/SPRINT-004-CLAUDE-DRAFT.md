# SPRINT-004 — Runtime Pack Loading And V100 Device-Residency Smoke (CLAUDE draft)

**Status:** DRAFT 2026-05-17
**Predecessor:** SPRINT-003 (manifest packer baseline)
**Successor:** First source-format math path (BF16 embedding or F8 attention probe), gated on this sprint shipping structural residency cleanly.

---

## Overview

Sprint 003 left the repo with a deterministic offline shard contract: the
manifest in `docs/sprints/drafts/SPRINT-002-PACK-MANIFEST.tsv` describes every
source tensor, and `tools/ds4-v100-pack` produces a matching
`pack-index.tsv` plus optional `gpuN.weights` shard files. Runtime decode of
the source model is still explicitly guarded in `ds4.c`, and nothing in the
runtime knows about the pack index.

Sprint 004 is the smallest credible step from "offline pack exists on disk"
to "the V100 appliance has source-faithful packed bytes resident in 8 device
arenas." It does not attempt decode. It does not attempt any source-format
math. It does prove, end-to-end, that we can:

1. parse `pack-index.tsv` in the runtime and reconcile it with the loaded
   source GGUF tensor binding, dtype, layout, and byte length;
2. allocate per-GPU device arenas on a real 8x V100 node without overfilling
   any 32 GiB GPU, with a documented reserve;
3. upload source-faithful packed bytes for every tensor to its owning GPU
   from either the source GGUF or an emitted shard, and report per-GPU VRAM
   usage that matches the planner;
4. emit shard files to persistent cluster scratch (Sprint 003 deferred this);
5. produce validation artifacts — per-GPU shard size, per-shard SHA-256,
   per-tensor spot checks, per-GPU VRAM report — that future numerical work
   can rebase on without re-doing the structural plumbing.

The decode guard stays in place. The CUDA backend's existing single-runtime
kernels are not touched. The new per-GPU residency path is a sidecar API used
only by a new diagnostic tool (`tools/ds4-v100-residency-smoke`) and the
runtime's startup-time pack reconciliation. The next sprint can pick one
tensor family (BF16 embedding is the cheapest candidate) and add the first
source-format math probe on top of this.

**Outcome contract:**

- **SHIP** if all of the following land: pack-index reader matches source
  binding for the full real model; per-GPU arenas allocate and accept uploads
  for every tensor on the V100 node; per-GPU VRAM usage at end-of-smoke
  matches the Sprint 003 planner within a declared tolerance (≤ 64 MiB per
  GPU); shard emission to persistent scratch completes with checksums; the
  decode guard remains in force.
- **EXTEND** if the pack reader + arena scaffolding + local synthetic smoke
  land but the cluster residency smoke or shard emission is blocked by
  infrastructure (scratch, node availability) — record the blocker and roll
  the cluster step into Sprint 005.
- **STOP** if the source manifest contradicts the source tensor binding in a
  material way (dtype, shape, or byte length disagrees), or if the planned
  per-GPU residency exceeds 32 GiB minus reserve on the real model.

---

## Use Cases

Each phase has a useful artifact even if a later phase slips:

| Phase | Useful output if sprint stops here |
|---|---|
| P0 | Local CPU/CUDA build remains green; `make tools/ds4-v100-pack` still works; cluster orientation documented. |
| P1 | `ds4_pack.h/c` parses `pack-index.tsv`, exposes lookup-by-semantic-id, and reconciles against the loaded source GGUF tensor table at engine open. `inspect_only` flag prints the reconciliation table and exits. |
| P2 | `ds4_gpu_arena_*` per-device sidecar API lives in `ds4_cuda.cu` with a CPU stub for non-CUDA builds. Allocations call `cudaSetDevice(gpu)`, allocations and frees are tracked per device, and a memory report prints per-GPU bytes used. |
| P3 | `tools/ds4-v100-residency-smoke` runs locally against a small synthetic pack and exits clean, validating end-to-end wiring of (pack reader → arena allocator → upload). |
| P4 | Cluster: full `--emit-shards` run on persistent scratch completes; per-shard SHA-256 and sizes are recorded. |
| P5 | Cluster: residency smoke uploads all 145.42 GiB of source-faithful bytes into 8 per-GPU arenas, prints per-GPU VRAM, exits without decode. |
| P6 | `SPRINT-004-REPORT.md` + `SPRINT-004-FOLLOWUPS.md` are written, listing the next-sprint surface (first math probe and CUDA multi-device refactor scope). |

---

## Architecture

### Source of truth

`docs/architecture/DS4-V100-LAYOUT.md` remains the planning baseline.
`docs/sprints/drafts/SPRINT-002-PACK-MANIFEST.tsv` is the authoritative
source-to-owner mapping, and `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv`
is the derived shard plan. This sprint does not modify those files. If the
runtime reconciliation finds a manifest/source mismatch, that is a STOP
signal, not a license to silently rewrite the manifest.

### Module boundaries

The new code lives in two narrow modules. The public engine boundary in
`ds4.h` does not grow; pack loading is wired in behind the existing
`ds4_engine_open` and exposed only as an additional `ds4_engine_options`
field for the pack-index path.

```
ds4_pack.h/c        # pack-index parser, tensor lookup, source reconciliation
ds4_gpu.h           # +ds4_gpu_arena_* sidecar API (no changes to existing
                    #  single-runtime kernel signatures)
ds4_cuda.cu         # arena implementation + per-device state
                    #  (additive; existing globals untouched)
tools/ds4-v100-residency-smoke.c   # new diagnostic
```

The existing `g_cublas`, `g_model_host_base`, `g_model_device_base`,
`g_model_ranges` (single-device caches) are **not** refactored in this sprint.
The arena API is a parallel, residency-only surface. A proper
`ds4_gpu_context[gpu]` split is explicitly deferred to a follow-up sprint
once at least one source-format math path forces the question.

### Pack-index reader contract

`ds4_pack` parses `pack-index.tsv` produced by `ds4-v100-pack
--write-index`. The schema from Sprint 003 is:

```
semantic_tensor_id source_name source_dtype source_shape
runtime_layout owning_gpu layer_id kernel_family
source_offset byte_length shard_file shard_offset
scale_offset checksum
```

The reader exposes:

```c
typedef struct ds4_pack ds4_pack;

typedef struct {
    const char *semantic_id;
    const char *source_name;
    const char *source_dtype;     // bf16 | f32 | f8_e4m3_b128 | mxfp4 | i32 | ...
    const char *runtime_layout;
    const char *kernel_family;
    int         owning_gpu;
    int         layer_id;
    uint64_t    source_offset;    // absolute offset into the source GGUF
    uint64_t    byte_length;
    const char *shard_file;       // gpuN.weights, may be NULL when shards
                                  //  are not emitted
    uint64_t    shard_offset;
} ds4_pack_entry;

int  ds4_pack_open(ds4_pack **out, const char *index_path);
void ds4_pack_close(ds4_pack *p);
int  ds4_pack_lookup(const ds4_pack *p, const char *semantic_id,
                     ds4_pack_entry *out);
int  ds4_pack_for_each(const ds4_pack *p,
                       int (*cb)(const ds4_pack_entry *e, void *ud),
                       void *ud);
uint64_t ds4_pack_payload_bytes(const ds4_pack *p, int gpu);
int      ds4_pack_tensor_count(const ds4_pack *p, int gpu);
```

A second function reconciles the pack against the loaded source model:

```c
int ds4_pack_reconcile(const ds4_pack *p,
                       const ds4_engine *e,
                       FILE *report_fp);   // prints per-tensor row,
                                           //  returns nonzero on mismatch
```

Reconciliation checks per tensor:

- `source_name` exists in the loaded tensor binding;
- `source_dtype` matches the GGML type of the bound tensor;
- `source_shape` parses and matches the bound tensor dimensions;
- `byte_length` matches the bound tensor's resident byte estimate;
- `source_offset` falls inside the mmap range and `source_offset +
  byte_length` does not exceed model size;
- `owning_gpu` is in `[0, n_gpus)`.

Any mismatch is a hard error at engine open when the pack path is requested.
There is no "warn and continue" mode — this is the structural contract.

### Per-GPU arena API

A minimal sidecar in `ds4_gpu.h`:

```c
typedef struct ds4_gpu_arena ds4_gpu_arena;

int   ds4_gpu_device_count(void);              // wraps cudaGetDeviceCount
                                                //  (CPU build returns 0)

int   ds4_gpu_arena_open(ds4_gpu_arena **out,
                         int gpu, uint64_t bytes);
void  ds4_gpu_arena_close(ds4_gpu_arena *a);

// Append-style upload at a known offset.  Internally cudaSetDevice(gpu)
// then cudaMemcpy from host pointer (which may be an mmap'd source range
// or a pread buffer from a gpuN.weights file).
int   ds4_gpu_arena_upload(ds4_gpu_arena *a,
                           uint64_t offset,
                           const void *host_src,
                           uint64_t bytes);

// Spot-check for validation: read back a small range.
int   ds4_gpu_arena_read(const ds4_gpu_arena *a,
                         uint64_t offset, void *dst, uint64_t bytes);

uint64_t ds4_gpu_arena_bytes(const ds4_gpu_arena *a);
uint64_t ds4_gpu_arena_used(const ds4_gpu_arena *a);
int      ds4_gpu_arena_gpu(const ds4_gpu_arena *a);
void     ds4_gpu_arena_print_memory_report(ds4_gpu_arena * const *arenas,
                                           int n);
```

Important constraints:

- Arenas are **device-local**: every `cuda*` call inside the implementation
  is bracketed by `cudaSetDevice(arena->gpu)` even when the global
  `cudaSetDevice` left a different device current.
- No persistent F16/F32 dequantized copies. Uploads are bit-for-bit copies of
  the source-faithful packed bytes named in the pack-index.
- The arena keeps a small per-tensor offset map for spot-checks; this is for
  validation only and is not the long-term per-tensor descriptor.
- The implementation file (`ds4_cuda.cu`) gains a new translation unit
  region; existing globals are untouched.

### Source-byte providers

Two providers feed `ds4_gpu_arena_upload`, chosen at smoke time:

| Provider | Use | Why |
|---|---|---|
| `gguf` | mmap the source GGUF; for each pack entry, hand the arena `mmap_base + source_offset` | Fastest iteration; no scratch required; works before shard emission |
| `shard` | `pread` from `gpuN.weights` into an aligned host buffer; hand to arena | Validates the emitted shards end-to-end; uses persistent scratch |

Both providers produce the same arena contents byte-for-byte. The smoke
records which provider was used per run and includes a small same-tensor
cross-check (any single tensor, picked deterministically) between the two
providers on cluster runs where both are available.

### Residency smoke contract

`tools/ds4-v100-residency-smoke`:

```bash
ds4-v100-residency-smoke \
  --model  /models/DSv4-Flash-256e-fixed.gguf \
  --index  /srv/scratch/ds4-pack/pack-index.tsv \
  --shard-dir /srv/scratch/ds4-pack \
  --provider gguf|shard \
  --reserve-mib 3072 \
  --report /srv/scratch/ds4-pack/SPRINT-004-RESIDENCY.log
```

Behaviour:

1. Open the engine with `inspect_only=true` and the new pack-index path.
2. Run `ds4_pack_reconcile` against the loaded binding; abort on any
   mismatch.
3. For each GPU `g`, allocate an arena of size `pack_payload_bytes(g)`
   rounded up to the smoke's alignment.
4. For each pack entry in `owning_gpu` order, upload the bytes via the
   chosen provider.
5. After every upload, perform a 4 KiB spot read from the arena's first and
   last bytes of that tensor and verify they match the source bytes (this
   catches `cudaMemcpy` length/offset errors cheaply).
6. Print `ds4_gpu_arena_print_memory_report` for all arenas.
7. Compare per-GPU used bytes against the Sprint 003 planner. Fail if any
   GPU exceeds 32 GiB − `reserve-mib`.
8. Exit. No decode, no kernels, no math.

### Validation artifacts

Sprint 004 lands these artifacts in `docs/sprints/drafts/`:

| Artifact | Source | Sized for |
|---|---|---|
| `SPRINT-004-SHARD-SHA256.tsv` | `sha256sum gpuN.weights` for `N in 0..7` | 8 lines |
| `SPRINT-004-SHARD-SIZES.tsv` | `stat` on each shard | 8 lines |
| `SPRINT-004-RECONCILE.log` | `ds4_pack_reconcile` output | 1328 tensor rows |
| `SPRINT-004-RESIDENCY.log` | smoke run with `--provider gguf` | per-GPU memory report + spot-check pass count |
| `SPRINT-004-RESIDENCY-SHARD.log` | smoke run with `--provider shard` (cluster only) | same shape |
| `SPRINT-004-CROSSCHECK.log` | one-tensor compare between providers | one row |

Per-tensor full SHA-256 is deliberately not in scope. Per-tensor spot checks
at the first and last 4 KiB of every tensor combined with whole-shard
SHA-256 are sufficient for the structural contract; full per-tensor hashing
is reserved for the math sprint where it is cheap to fold into the kernel
correctness harness.

---

## Implementation

### Phase 0: Orientation And Build Hygiene

**Files:**

- `docs/sprints/SPRINT-004-REPORT.md` (create as work starts)
- `Makefile`

**Tasks:**

- [ ] Confirm `make cpu` and `make tools/ds4-v100-pack` build clean on
      laptop.
- [ ] Confirm the 8x V100 cluster pod recipe in
      `/Users/ravi/repos/deepseek/docs/sprints/SPRINT-026-CLUSTER-TESTING.md`
      works for this sprint's workload (CPU build, CUDA build, persistent
      scratch path).
- [ ] Record the persistent scratch path that will host emitted shards.
- [ ] Add a `make tools/ds4-v100-residency-smoke` target (CPU-stubbed when
      built without CUDA so that the parser/IO portion can be unit-tested
      locally).

**Kill gate:** stop if cluster persistent scratch is not provisioned and
cannot be provisioned within this sprint. In that case Sprint 004 ships
P1–P3 (local) and rolls P4–P5 to Sprint 005 as EXTEND.

### Phase 1: Pack-Index Reader And Source Reconciliation

**Files:**

- `ds4_pack.h` (new)
- `ds4_pack.c` (new)
- `ds4.h` (extend `ds4_engine_options` with `const char *pack_index_path`)
- `ds4.c` (call `ds4_pack_open` and `ds4_pack_reconcile` in
  `ds4_engine_open` when the option is set; produce
  `SPRINT-004-RECONCILE.log` under `--inspect-only`)
- `Makefile`
- `tests/pack_index_smoke.c` (new; parses a 3-row synthetic TSV)

**Tasks:**

- [ ] Parser handles header row, tab-delimited fields, comments,
      trailing newlines, and the exact column order produced by
      `ds4-v100-pack --write-index`.
- [ ] Lookup is O(log n) or O(1) — a sorted array with bsearch is fine
      given ~1328 entries.
- [ ] Reconciliation produces a single row per tensor:
      `semantic_id source_name source_dtype source_shape owning_gpu
      layer_id byte_length status` where `status` is `OK` or a specific
      mismatch tag.
- [ ] All mismatch tags are mechanical: `MISSING_BINDING`,
      `DTYPE_MISMATCH`, `SHAPE_MISMATCH`, `BYTE_LENGTH_MISMATCH`,
      `OFFSET_OUT_OF_RANGE`, `BAD_OWNING_GPU`.
- [ ] `tests/pack_index_smoke.c` covers happy-path, dtype mismatch,
      byte-length mismatch, and missing binding.
- [ ] `ds4_engine_open` with `pack_index_path != NULL` and any
      reconciliation failure returns nonzero and logs the offending row.
- [ ] `--inspect-only` writes the full reconcile log and exits 0 only when
      every row is `OK`.

**Kill gate:** stop if reconciliation fails on the real model in a way that
indicates the Sprint 002 manifest is wrong rather than the reader. A real
mismatch is a STOP that escalates to fixing the manifest.

### Phase 2: Per-GPU Arena Sidecar

**Files:**

- `ds4_gpu.h` (extend with the arena API only — no kernel-signature
  changes)
- `ds4_cuda.cu` (add arena implementation; existing globals untouched)
- `Makefile`
- `tests/gpu_arena_smoke.c` (new; CPU build uses a stub that exercises the
  offset/length validation logic; CUDA build runs against a real device
  when available)

**Tasks:**

- [ ] `ds4_gpu_arena_open`, `_upload`, `_read`, `_close` implemented.
- [ ] All `cuda*` calls bracketed by `cudaSetDevice(arena->gpu)`.
- [ ] Allocation uses plain `cudaMalloc`; no managed memory in this
      sprint. Managed memory remains a diagnostic-only flag elsewhere.
- [ ] Per-GPU memory report includes `bytes_total`, `bytes_used`,
      `peak_used`, `gpu_free_after_alloc`, and `gpu_total`.
- [ ] CPU stub returns `0` from `ds4_gpu_device_count`, returns errors
      from arena calls, and lets the parser/upload-orchestration logic in
      the smoke tool be exercised in unit tests without a GPU.

**Kill gate:** stop if `cudaSetDevice` + per-device `cudaMalloc` cannot
allocate the planned arena sizes on the V100 node with the declared
reserve. That outcome is a STOP for the residency claim and feeds back into
architecture review.

### Phase 3: Local Synthetic Residency Smoke

**Files:**

- `tools/ds4-v100-residency-smoke.c` (new)
- `Makefile`
- `tests/residency_smoke_synthetic.sh` (new; builds a 4-tensor synthetic
  GGUF + manifest + pack-index and runs the smoke in CPU/stub mode)

**Tasks:**

- [ ] Tool parses CLI options, opens engine with pack-index, runs
      reconciliation, allocates arenas, walks the pack-index per-GPU,
      uploads, spot-checks, prints memory report.
- [ ] On CPU/stub builds, "upload" becomes an in-memory copy and the
      memory report reads from the host arena. This is sufficient to
      exercise the orchestration code in CI.
- [ ] `tests/residency_smoke_synthetic.sh` runs end-to-end on the laptop
      without CUDA.

**Kill gate:** stop if local synthetic smoke cannot pass — the cluster
runs should never be the first place the orchestration is exercised.

### Phase 4: Cluster Shard Emission On Persistent Scratch

**Files:**

- `docs/sprints/drafts/SPRINT-004-SHARD-SIZES.tsv` (artifact, written by
  the cluster run)
- `docs/sprints/drafts/SPRINT-004-SHARD-SHA256.tsv` (artifact)
- `docs/sprints/SPRINT-004-REPORT.md`

**Tasks:**

- [ ] On the V100 pod, build `tools/ds4-v100-pack`.
- [ ] Run `ds4-v100-pack --emit-shards --source
      /models/DSv4-Flash-256e-fixed.gguf --out-dir /srv/scratch/ds4-pack
      --write-index` against persistent scratch.
- [ ] Record per-shard `stat -c '%n\t%s'` into `SPRINT-004-SHARD-SIZES.tsv`.
- [ ] Record per-shard `sha256sum` into `SPRINT-004-SHARD-SHA256.tsv`.
- [ ] Cross-check measured per-shard size against the Sprint 003 plan; any
      delta is a sprint-blocker.

**Kill gate:** stop if persistent scratch cannot hold ~146 GiB plus a safety
margin. Falling back to ephemeral scratch is not allowed for this step.

### Phase 5: Cluster Per-GPU Residency Smoke

**Files:**

- `docs/sprints/drafts/SPRINT-004-RECONCILE.log` (artifact)
- `docs/sprints/drafts/SPRINT-004-RESIDENCY.log` (artifact)
- `docs/sprints/drafts/SPRINT-004-RESIDENCY-SHARD.log` (artifact)
- `docs/sprints/drafts/SPRINT-004-CROSSCHECK.log` (artifact)
- `docs/sprints/SPRINT-004-REPORT.md`

**Tasks:**

- [ ] Build CPU + CUDA targets on the pod, including the residency smoke
      with CUDA.
- [ ] Run reconciliation against the real model; archive the log.
- [ ] Run residency smoke with `--provider gguf`; archive the report and
      per-GPU memory rows.
- [ ] Run residency smoke with `--provider shard`; archive the report.
- [ ] Run a single deterministic cross-check between providers for one
      tensor (pick `blk.21.attn_q_b.weight` as a non-edge example) and
      confirm byte equality.
- [ ] Compare per-GPU used bytes to the Sprint 003 planner; fail if any
      GPU exceeds 32 GiB − 3 GiB reserve (tunable, but report the value
      used).
- [ ] After validation, delete the cluster pod per the cluster-testing
      guide.

**Kill gate:** stop if the residency smoke reports any GPU within 256 MiB
of the 32 GiB ceiling after reserve; that is a planning problem, not an
implementation problem, and must be escalated to the architecture document.

### Phase 6: Sprint Report And Follow-Ups

**Files:**

- `docs/sprints/SPRINT-004-REPORT.md` (create)
- `docs/sprints/SPRINT-004-FOLLOWUPS.md` (create)
- `docs/sprints/drafts/SPRINT-005-SEED.md` (optional but recommended)

**Tasks:**

- [ ] Record verdict (SHIP / EXTEND / STOP) with evidence pointers.
- [ ] Enumerate the next-sprint surface: which tensor family should be
      the first source-format math probe, what the minimal CUDA
      multi-device refactor needs to look like to support real kernel
      execution, and which validation artifacts the math sprint should
      consume.
- [ ] Move follow-ups from `SPRINT-003-FOLLOWUPS.md` that are now
      completed into the report.
- [ ] Leave the source-model decode guard in place.

**Kill gate:** none — this phase always runs, even on STOP/EXTEND.

---

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `docs/sprints/SPRINT-004.md` | Create (after merge of drafts) | Final sprint plan |
| `docs/sprints/SPRINT-004-REPORT.md` | Create during execution | Evidence and verdict |
| `docs/sprints/SPRINT-004-FOLLOWUPS.md` | Create at end | Next-sprint surface |
| `docs/sprints/drafts/SPRINT-004-SHARD-SIZES.tsv` | Create on cluster | Per-shard file sizes |
| `docs/sprints/drafts/SPRINT-004-SHARD-SHA256.tsv` | Create on cluster | Per-shard checksums |
| `docs/sprints/drafts/SPRINT-004-RECONCILE.log` | Create on cluster | Pack reconciliation row table |
| `docs/sprints/drafts/SPRINT-004-RESIDENCY.log` | Create on cluster | Per-GPU memory + spot-check report (provider=gguf) |
| `docs/sprints/drafts/SPRINT-004-RESIDENCY-SHARD.log` | Create on cluster | Same, provider=shard |
| `docs/sprints/drafts/SPRINT-004-CROSSCHECK.log` | Create on cluster | One-tensor provider cross-check |
| `ds4_pack.h` | Create | Pack-index parser/lookup/reconcile API |
| `ds4_pack.c` | Create | Implementation; depends only on stdio/string |
| `ds4.h` | Modify | Add `pack_index_path` to `ds4_engine_options` |
| `ds4.c` | Modify | Wire `ds4_pack_open` and `ds4_pack_reconcile` into engine open; keep source-model decode guard in place |
| `ds4_gpu.h` | Modify | Add `ds4_gpu_arena_*` sidecar API only |
| `ds4_cuda.cu` | Modify | Implement arenas; do not touch existing single-runtime kernels or globals |
| `tools/ds4-v100-residency-smoke.c` | Create | New diagnostic tool |
| `Makefile` | Modify | Add `tools/ds4-v100-residency-smoke` and test targets |
| `tests/pack_index_smoke.c` | Create | Parser/reconciliation unit tests |
| `tests/gpu_arena_smoke.c` | Create | Arena offset/length validation and (when CUDA available) live device test |
| `tests/residency_smoke_synthetic.sh` | Create | End-to-end synthetic smoke |
| `docs/architecture/DS4-V100-LAYOUT.md` | Read; modify only on STOP | Source of truth for layout/reserve |
| `docs/sprints/drafts/SPRINT-002-PACK-MANIFEST.tsv` | Read-only | Authoritative source-to-owner map |
| `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv` | Read-only | Derived shard plan |
| `tools/ds4-v100-pack.c` | Read; modify only to add a tiny CLI hardening if Phase 4 surfaces a real issue | Shard emitter |

---

## Definition Of Done

- [ ] `ds4_pack` parses `pack-index.tsv` and exposes the documented
      lookup and per-GPU aggregate functions.
- [ ] `ds4_pack_reconcile` produces a deterministic per-tensor table and
      returns nonzero on any mismatch.
- [ ] `ds4_engine_open` accepts a `pack_index_path` and runs
      reconciliation before any GPU work; failures abort engine open.
- [ ] `ds4_gpu_arena_*` is implemented in `ds4_cuda.cu` without modifying
      existing single-runtime globals or kernel signatures.
- [ ] `tools/ds4-v100-residency-smoke` runs locally against the
      synthetic fixture and on the cluster against the real model with
      both providers (`gguf` and `shard`).
- [ ] Cluster shard emission completes on persistent scratch with shard
      sizes matching the Sprint 003 plan and `sha256sum` recorded.
- [ ] Per-GPU VRAM after upload matches the planner within 64 MiB on
      every GPU, with a 3 GiB reserve documented in the report.
- [ ] All per-tensor spot checks (first 4 KiB and last 4 KiB) pass.
- [ ] The single deterministic cross-tensor compare between providers
      passes byte-for-byte.
- [ ] No persistent F16/F32 dequantized copies of source tensors are
      created at any point; bytes uploaded to arenas are source-faithful.
- [ ] The source-model generation guard in `ds4.c` remains in place; no
      kernel dispatch path is enabled for the source model.
- [ ] `SPRINT-004-REPORT.md` records the verdict, evidence pointers, and
      a concrete Sprint 005 surface.
- [ ] No upstream PR; all work lives on the private fork. CLAUDE.md and
      AGENT.md constraints are preserved.

---

## Risks And Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| Persistent cluster scratch is not provisioned in time | Medium | High | Phase 0 confirms scratch first; if unavailable, sprint ships P1–P3 + report and rolls cluster steps to Sprint 005 as EXTEND |
| Manifest/source mismatch is real, not a reader bug | Low | High | Reconciliation is mechanical; mismatch flags are specific; fix the manifest (Sprint 002) rather than silently rewriting in Sprint 004 |
| Single-device CUDA globals collide with new per-device arenas | Medium | Medium | Arena API is a parallel surface; do not call legacy `ds4_gpu_set_model_*` from the smoke tool; reset `cudaSetDevice` at every entry and exit of arena functions |
| Per-GPU residency exceeds 32 GiB minus reserve on real model | Low | High | Sprint 003 plan shows max 20.98 GiB per GPU; recheck with a 3 GiB reserve; STOP and revisit architecture if the real run disagrees |
| Spot checks pass but mid-tensor bytes are wrong | Low | Medium | Add the one deterministic cross-provider compare in Phase 5; whole-shard SHA-256 catches drift between source GGUF and emitted shards |
| Scope creep into source-format math (BF16 embedding, FP8 dequant) | Medium | Medium | Open Question 3 is explicitly resolved as "no math this sprint"; that decision lives in this plan and the smoke tool refuses any non-residency flag |
| CUDA pod gets recycled or has no CUDA 12 driver | Medium | Medium | Match the homelab build image used in Sprint 003; pin CUDA version in the report |
| Test fixtures rot because the manifest and pack-index TSV headers drift | Low | Medium | Parser asserts header columns by name and order at open time; synthetic test fixture lives in `tests/` and is updated alongside any header change |
| The arena API quietly becomes the new "production" multi-GPU API and accumulates kernel code | Medium | Medium | API stays in `ds4_gpu.h` under a documented "residency-only" comment; the next sprint's first math probe is the trigger to design the real `ds4_gpu_context[gpu]` |

---

## Security Considerations

- The pack-index reader treats `pack-index.tsv` as trusted local input but
  still bounds-checks every numeric field (`source_offset`, `byte_length`,
  `shard_offset`, `owning_gpu`). Overflow or out-of-range values are hard
  errors at parse time.
- All `mmap`/`pread` source-byte reads validate `[offset, offset+length)`
  against the file size before issuing the syscall.
- The arena API rejects uploads with `offset + bytes >
  ds4_gpu_arena_bytes(a)`.
- The residency smoke writes only to its own report file and the per-GPU
  arenas. It does not write to the source GGUF, the pack-index, or the
  shard files. It opens all model inputs `O_RDONLY`.
- No paths, hashes, or model bytes are exposed over any server API; the
  HTTP server in `ds4_server.c` is not modified in this sprint.
- The arena API does not enable `cudaHostRegister` or managed memory
  defaults; pinned/managed paths remain diagnostic flags elsewhere.
- The source-model generation guard in `ds4.c` is preserved verbatim.
  Enabling decode for the source model still requires a bounded correctness
  harness in a later sprint.

---

## Dependencies

- Private repo `rapatel0/ds4` on `main`.
- 8x V100-SXM2-32GB cluster node with the homelab `homelab-k8s-dev` build
  image (CUDA 12.x, sm_70).
- Persistent cluster scratch with ~160 GiB free for emitted shards plus
  margin.
- Source model at `/models/DSv4-Flash-256e-fixed.gguf` (confirmed in
  Sprints 002–003).
- `docs/sprints/drafts/SPRINT-002-PACK-MANIFEST.tsv` and
  `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv` as inputs.
- Cluster operating procedure
  `/Users/ravi/repos/deepseek/docs/sprints/SPRINT-026-CLUSTER-TESTING.md`.
- Architecture document `docs/architecture/DS4-V100-LAYOUT.md` for
  reserve/headroom expectations.

---

## Open Questions

The Sprint 004 intent left five open questions. Resolutions proposed in this
draft (subject to review at merge time):

1. **Full real-model shard emission scope.** Required for SHIP if persistent
   scratch is available. If scratch is not provisioned, sprint ships
   EXTEND with shard emission moved to Sprint 005. Sprint does not
   downgrade to disposable storage.
2. **Read from `gpuN.weights` vs direct GGUF offsets.** Support **both**
   via the `--provider` flag. Default to `gguf` for iteration speed; run
   `shard` once on the cluster to validate the emitted shards. The single
   cross-provider compare in Phase 5 is the binding contract.
3. **Stop at raw packed bytes or attempt one tiny math path.** **Stop at
   raw packed bytes.** No math, no kernel dispatch, no decode. The
   smallest credible math probe (BF16 embedding) becomes the seed for
   Sprint 005; bundling it here pulls in `ds4_gpu_context[gpu]` scope
   that is better designed against a concrete kernel.
4. **CUDA multi-device refactor scope.** **Minimal per-device arenas
   only.** No `ds4_gpu_context[gpu]` split. Existing globals are not
   touched. The next sprint pays for the proper refactor when it has a
   real kernel to drive the API shape.
5. **Minimum validation artifact.** Per-GPU shard size + per-shard
   SHA-256 + per-tensor first/last-4 KiB spot checks + one deterministic
   cross-provider compare. Full per-tensor SHA-256 is deferred to the
   math sprint where it can fold into kernel correctness checks.

Remaining questions for the merge step:

- Should the reserve in Phase 5 be 3 GiB (CUDA-context envelope) or a
  measured value from the cluster pod? Defaulting to 3 GiB; revisit when
  cluster numbers land.
- Should the report cite per-GPU `cudaMemGetInfo` before and after upload,
  or only after? Both, if cheap.
- Should `ds4_pack_reconcile` also emit a JSON summary, or is TSV
  sufficient? TSV-only for this sprint; JSON is a Sprint 005 ask if the
  math harness wants it.
