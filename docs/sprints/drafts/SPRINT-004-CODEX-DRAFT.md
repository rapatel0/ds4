# Sprint 004: Runtime Pack Loading And V100 Residency Smoke

## Overview

Sprint 003 proved that the source-layout manifest can be turned into a
deterministic `pack-index.tsv` plus optional `gpuN.weights` shard files. The
next blocking gap is runtime consumption: `ds4.c` still binds tensors to GGUF
offsets, `ds4_cuda.cu` still assumes one global model mapping, and the source
layout remains correctly guarded off from decode. Sprint 004 should close that
gap without pretending decode correctness exists yet.

The sprint should add a narrow runtime pack contract that validates the pack
index against the real GGUF tensor directory, resolves every source-layout
tensor to `(owning_gpu, shard_file, shard_offset, byte_length, dtype, layout,
kernel_family)`, and uploads those source-faithful bytes into per-GPU device
arenas on the V100 host. The output of the sprint is a structural proof:
runtime pack metadata is exact, shard files are loadable, and eight V100s can
hold the planned weight bytes in device memory with explicit reporting.

The sprint should stay intentionally bounded. It should not attempt broad
multi-device execution, HC relay, or source-format math dispatch for normal
generation. The existing source-model generation guard should remain in place.
If persistent scratch is unavailable for full shard emission, the sprint may
end with an explicit `STOP` backed by the concrete storage blocker and the
validation already completed.

## Use Cases

1. **Pack validation before runtime work**: As the operator, I can point a
   diagnostic runtime path at a GGUF plus a pack directory and get a fail-closed
   verdict if `pack-index.tsv` does not exactly match the source model metadata.
2. **Deterministic tensor resolution**: As the runtime, I can resolve
   source-layout tensors from `gpuN.weights` and shard offsets instead of
   directly from GGUF byte offsets.
3. **Device residency proof**: As the operator, I can run a V100 smoke that
   uploads all planned shard bytes into the owning GPUs, prints per-GPU loaded
   bytes and free memory, and exits before decode.
4. **Packer artifact verification**: As the decision maker, I can see concrete
   shard file sizes and residency logs that show whether the pack contract is
   usable on the real 8x V100 target.
5. **Guarded bring-up**: As the maintainer, I can land runtime pack loading
   without enabling normal source-model generation until a later correctness
   harness exists.

## Architecture

Sprint 004 should introduce a bounded runtime-pack path that sits beside the
current GGUF-backed execution path.

```text
GGUF model_path
    |
    v
ds4.c model loader + source-layout binder
    |
    +--> validate tensor names/dtypes/shapes against pack-index rows
    |        |
    |        v
    |   runtime pack plan
    |   (semantic tensor -> gpuN.weights + shard offset + bytes)
    |
    +--> existing source decode guard remains in place
             |
             v
tools/ds4-v100-residency-smoke
    |
    v
ds4_gpu.h / ds4_cuda.cu upload-only residency layer
    |
    v
per-GPU device arenas + memory report + validation artifacts
```

Key design points:

- `ds4.c` remains the authority for GGUF metadata, tensor identity, and the
  recognized DeepSeek V4 Flash source layout. The pack index is an additional
  runtime contract, not a replacement for GGUF validation.
- Add a small runtime-pack parser/plan builder that treats `pack-index.tsv` as
  immutable input. It should reject duplicate tensor rows, missing tensors,
  mismatched dtypes/shapes, invalid `gpuN.weights` names, out-of-range shard
  offsets, and rows that do not match the bound source-layout tensors.
- Keep the public API narrow. The cleanest entry point is a bounded diagnostic
  function exported from `ds4.h`, for example
  `ds4_v100_residency_smoke(model_path, pack_dir, ...)`, rather than widening
  normal generation APIs or weakening the existing source-model guard.
- `ds4_cuda.cu` should not be fully refactored into general multi-device decode
  state in this sprint. Instead, add a contained per-device residency layer that
  owns:
  visible-device enumeration,
  per-GPU arena allocation,
  shard upload,
  loaded-byte accounting,
  and per-GPU memory reporting.
- Source-faithful bytes are the target artifact. No persistent F16-expanded
  copies of BF16, FP8, or MXFP4 tensors should remain resident after upload.

## Implementation

### Phase 1: Runtime Pack Contract And Parser (~30% of effort)

**Files:**
- `ds4.c` — Add internal runtime-pack row validation and plan construction.
- `ds4.h` — Export one narrow diagnostic entry point for residency smoke.
- `tests/ds4_test.c` — Add synthetic parser/validator coverage.

**Tasks:**
- [ ] Parse the Sprint 003 `pack-index.tsv` schema exactly as emitted today.
- [ ] Validate that every source-layout tensor bound by `ds4.c` has exactly one
      corresponding pack row.
- [ ] Cross-check `source_name`, `source_dtype`, shape, `runtime_layout`,
      `owning_gpu`, `kernel_family`, `byte_length`, and shard coordinates.
- [ ] Reject malformed shard file names and any row whose shard range would run
      past the recorded file size.
- [ ] Add synthetic tests for duplicate rows, missing rows, dtype drift, shape
      drift, bad GPU ownership, and bad shard offsets.

### Phase 2: Bind Pack Rows To A Runtime Residency Plan (~20% of effort)

**Files:**
- `ds4.c` — Build a `ds4_runtime_pack_plan` after source-layout binding.
- `docs/architecture/DS4-V100-LAYOUT.md` — Update only if the implemented
  residency plan diverges from the documented ownership assumptions.

**Tasks:**
- [ ] Build a compact runtime plan keyed by semantic tensor identity, not by raw
      GGUF byte offset alone.
- [ ] Preserve the existing source-layout generation guard for normal engine
      open and session creation.
- [ ] Make the new diagnostic path reuse the same loader/binder checks as normal
      runtime startup so pack validation is not a separate one-off parser.
- [ ] Emit a clear failure message when the pack directory is absent, stale, or
      inconsistent with the GGUF in hand.

### Phase 3: Add Upload-Only Per-GPU Residency Smoke (~35% of effort)

**Files:**
- `ds4_gpu.h` — Add the minimal residency-upload/report API.
- `ds4_cuda.cu` — Implement per-device upload state and memory reporting.
- `tools/ds4-v100-residency-smoke.c` — Standalone diagnostic tool for cluster
  validation.
- `Makefile` — Build the new smoke tool.

**Tasks:**
- [ ] Enumerate visible CUDA devices and fail closed unless the intended V100
      topology is available for the real cluster smoke.
- [ ] Allocate one bounded weight arena per owning GPU sized from the runtime
      pack plan.
- [ ] Upload each `gpuN.weights` shard into its owning device arena without
      leaving persistent dequantized mirrors behind.
- [ ] Record per-GPU loaded bytes, free/total VRAM before and after upload, and
      the residency class used for each upload.
- [ ] Keep the existing single-device math and tensor APIs working; the new
      per-device state in this sprint is for upload and reporting only.
- [ ] Exit before decode, with the source-model generation guard still active in
      the main runtime path.

### Phase 4: Cluster Validation And Artifacts (~15% of effort)

**Files:**
- `docs/sprints/drafts/SPRINT-004-RESIDENCY-SMOKE.log` — Create validation log.
- `docs/sprints/drafts/SPRINT-004-SHARD-FILES.tsv` — Create shard size facts
  and optional checksums.
- `docs/sprints/drafts/SPRINT-004-STOP.md` — Create only if persistent scratch
  or residency fit blocks completion.

**Tasks:**
- [ ] Run `tools/ds4-v100-pack --emit-shards` against the real
      `/models/DSv4-Flash-256e-fixed.gguf` on persistent scratch.
- [ ] Record `gpuN.weights` file sizes and, if practical, full-shard checksums
      or a documented subset of byte-spot checks.
- [ ] Run the new residency smoke on the 8x V100 pod and capture the memory
      report plus loaded-byte totals.
- [ ] Delete the temporary test pod after artifacts are copied out.
- [ ] If persistent scratch is unavailable, stop quickly and record the exact
      blocker instead of widening scope into decode or alternative offload paths.

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `ds4.c` | Modify | Validate pack-index rows against bound source-layout tensors and build the runtime residency plan. |
| `ds4.h` | Modify | Export one narrow diagnostic entry point for source-layout residency smoke without weakening normal generation APIs. |
| `ds4_gpu.h` | Modify | Add minimal upload/report hooks for pack-backed multi-device residency. |
| `ds4_cuda.cu` | Modify | Add contained per-device shard-upload state, per-GPU arena allocation, and memory reporting. |
| `tools/ds4-v100-residency-smoke.c` | Create | Run the bounded GGUF + pack-dir validation and device residency smoke on the cluster. |
| `Makefile` | Modify | Build the new residency smoke tool. |
| `tests/ds4_test.c` | Modify | Cover parser and contract validation on synthetic pack-index inputs. |
| `docs/sprints/drafts/SPRINT-004-RESIDENCY-SMOKE.log` | Create | Capture cluster residency output and per-GPU memory facts. |
| `docs/sprints/drafts/SPRINT-004-SHARD-FILES.tsv` | Create | Record shard file sizes and optional checksum facts from the real emission run. |
| `docs/sprints/drafts/SPRINT-004-STOP.md` | Create if needed | Document the exact storage or residency blocker if the sprint cannot complete. |

## Definition of Done

- [ ] A runtime pack validator exists and rejects malformed or stale
      `pack-index.tsv` inputs against the real GGUF tensor directory.
- [ ] Every source-layout tensor needed for Sprint 004 resolves to one validated
      `(owning_gpu, shard_file, shard_offset, byte_length, dtype, layout,
      kernel_family)` record.
- [ ] `tools/ds4-v100-residency-smoke` builds and can run against synthetic or
      local pack fixtures without decode.
- [ ] Full real-model shard emission is completed on persistent scratch, or the
      repo contains an explicit `STOP` artifact naming the exact storage blocker.
- [ ] The V100 cluster smoke uploads all planned shard bytes into the owning GPU
      arenas without reported overfill of any 32 GB V100.
- [ ] Validation artifacts capture shard file-size facts and per-GPU VRAM usage
      after upload.
- [ ] The existing source-model generation guard remains in place for normal
      runtime decode paths.
- [ ] Local build/test verification passes for the touched code paths:
      `make cpu`, `make tools/ds4-v100-pack`, the new smoke-tool target, and
      `git diff --check`.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| The upload-only per-device layer quietly grows into a broad CUDA runtime refactor. | Medium | High | Keep Sprint 004 limited to shard upload, accounting, and reporting; defer HC relay and decode scheduling. |
| Persistent scratch is not available for the 145.42 GiB real-model shard emission. | Medium | High | Treat persistent scratch as a hard dependency; record an immediate `STOP` with the concrete blocker instead of inventing temporary offload workarounds. |
| Host-mapped or managed fallbacks make the smoke look successful even when pure VRAM residency is not real. | Medium | High | Report residency class explicitly and require device-resident weight arenas for a successful verdict. |
| Pack index drift or stale shard directories produce false confidence. | Medium | High | Validate pack rows against GGUF metadata and shard file sizes before any upload begins. |
| The current single-device CUDA globals interfere with the new per-device upload path. | Medium | Medium | Isolate upload bookkeeping from the existing math state; do not route normal kernels through the new path yet. |
| Full-shard checksum capture is too slow for the available cluster window. | Low | Medium | Accept shard file-size facts plus documented spot checks for Sprint 004 if full checksums are operationally expensive. |

## Security

- Treat shard directories, file sizes, and residency logs as local internal
  artifacts; do not expose them through `ds4_server.c` or any network API in
  this sprint.
- Validate pack-directory inputs strictly. Reject unexpected shard file names,
  path traversal, duplicate rows, and file-size mismatches before allocating
  device memory.
- Open shard files read-only in the diagnostic runtime path. The smoke tool
  should not mutate source GGUFs or shard payloads.
- Keep the normal source-model decode guard in place so a partial upload path
  cannot be mistaken for a correct inference path.

## Dependencies

- Sprint 002 source-layout manifest and tensor binding work.
- Sprint 003 packer output schema and the existing `pack-index.tsv`.
- `/models/DSv4-Flash-256e-fixed.gguf` on the cluster.
- Persistent scratch large enough for about 145.42 GiB of emitted shard files.
- 8x V100-SXM2-32GB availability plus a CUDA `sm_70` build environment.
- Existing V100 layout assumptions in `docs/architecture/DS4-V100-LAYOUT.md`.

## Open Questions

1. Should the bounded diagnostic entry live in `ds4.h`, or should Sprint 004
   accept a slightly larger internal refactor to avoid any public API growth?
2. Is shard validation for this sprint satisfied by file sizes plus a small set
   of byte spot checks, or is full-shard checksum capture required?
3. Should shard upload use explicit read buffers, `mmap` plus `cudaMemcpy`, or
   another bounded path that best fits the V100 pod environment?
4. Does the residency smoke need to require exactly eight visible V100s, or is
   a smaller synthetic bring-up mode acceptable as long as the real verdict is
   still gated on the full cluster run?
5. If persistent scratch is blocked, does the sprint stop immediately, or
   should it also land the runtime validator and synthetic upload smoke before
   closing with `STOP`?
