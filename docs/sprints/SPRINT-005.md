---
sprint: 005
title: First Resident BF16 Gather/Expand Probe
status: shipped
date: 2026-05-17
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-005-INTENT.md
merge_notes: drafts/SPRINT-005-MERGE-NOTES.md
deferred: SPRINT-005-DEFERRED.md
---

# SPRINT-005: First Resident BF16 Gather/Expand Probe

## Overview

Sprint 004 proved that source-faithful DS4 V100 packed weights can be
reconciled, sharded, uploaded, and held resident on the owning V100 devices.
The runtime still has no proven compute contract over those resident bytes.

Sprint 005 adds the smallest useful resident tensor probe: gather BF16 rows
from a `ds4_gpu_arena`, expand the source BF16 bit patterns to F32, and compare
against a CPU reference. V100 does not have native BF16 tensor-core execution;
this is a diagnostic gather/expand path, not the production performance path.
The first semantic target is
`token_embd.weight`, which is source BF16, owned by `gpu0`, and large enough to
exercise real arena offsets without requiring decode, KV, MTP, or layer
scheduling.

This is a diagnostic sprint, not a decode sprint. The probe proves pointer,
descriptor, dtype, bounds-checking, and CUDA launch semantics for resident
packed bytes. Sprint 006 can then design the production multi-GPU execution
context against a measured contract instead of a guess.

## Outcome Contract

- `SHIP`: local synthetic tests pass; the BF16 probe API has matching
  host-stub and CUDA implementations; a direct CUDA synthetic test passes on
  V100 with `CUDA_ARCH=sm_70`; a focused `token_embd.weight` residency-smoke
  probe runs on the cluster and records resident-span facts and expected versus
  actual F32 samples; source-model generation remains guarded.
- `EXTEND`: local/stub and CUDA build work land, but V100 execution or the
  real resident probe is blocked by cluster availability or infrastructure.
  The blocker and exact missing validation are recorded.
- `STOP`: arena-resident bytes cannot be used safely as CUDA kernel inputs
  without a larger incompatible refactor, or the measured source BF16 bytes do
  not match the expected source dtype contract.

## Non-Goals

- No source-model decode enablement.
- No prefill, KV allocation, compressed attention, or indexer integration.
- No MTP or speculative decoding.
- No FP8, MXFP4, INT8, or routed expert kernels.
- No production `ds4_gpu_context[8]`, HC relay, layer scheduler, or slot
  scheduler.
- No persistent F16/F32 dequantized weight copies.
- No server/API exposure for the probe.
- No device-output or FP16-output production path.
- No broad `ds4.c` source-layout dtype refactor unless needed to build the
  isolated probe.

## Planning Inputs

| File | Role |
|---|---|
| `docs/sprints/VISION.md` | Sprint sequence and North Star |
| `docs/architecture/DS4-V100-LAYOUT.md` | Source dtype, memory layout, and tensor-family anchor |
| `docs/sprints/SPRINT-004-REPORT.md` | Residency proof and current guard status |
| `docs/sprints/SPRINT-004-DEFERRED.md` | First source-format math probe and future scheduler work |
| `docs/sprints/SPRINT-004-FOLLOWUPS.md` | Direct CUDA arena test follow-up |
| `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv` | Current `token_embd.weight` semantic id, owner, and shard offset |
| `ds4_gpu.h`, `ds4_cuda.cu`, `ds4_gpu_arena_stub.c` | Arena API and GPU/stub implementation sites |
| `tools/ds4-v100-residency-smoke.c` | Existing residency upload and spot-check tool |

## Use Cases

1. **Model-less local correctness**: a developer can validate BF16 conversion
   and row addressing against synthetic data without the 145 GiB model or CUDA.
2. **Direct CUDA probe**: a developer can run a small CUDA-linked test that
   opens an arena, uploads a synthetic BF16 table, launches the probe, and
   verifies F32 output.
3. **Resident embedding proof**: the cluster smoke can probe
   `token_embd.weight` from resident arena bytes and log row ids, arena span,
   output samples, and reference comparison.
4. **Reusable contract for Sprint 006**: the next sprint gets a narrow
   descriptor-shaped contract for resident tensor views and fail-closed checks.
5. **Guarded bring-up**: a successful compute probe cannot be mistaken for full
   decode support.

## Architecture

### BF16 Matrix View

Add one descriptor-shaped view for BF16 matrix rows inside an arena:

```c
typedef struct {
    uint64_t arena_offset;
    uint64_t byte_length;
    uint32_t rows;
    uint32_t cols;
    uint32_t row_stride_elements;
} ds4_gpu_bf16_matrix_view;
```

Expected public probe shape:

```c
int ds4_gpu_arena_bf16_row_gather_f32(
        const ds4_gpu_arena             *arena,
        const ds4_gpu_bf16_matrix_view  *view,
        const uint32_t                  *row_ids,
        uint32_t                         n_rows,
        float                           *out_f32,
        uint64_t                         out_bytes);
```

The API follows the arena convention: return `0` on success and nonzero on
failure. The output contract for Sprint 005 is host F32 only. Device-resident
outputs and stream ownership are deferred to the production execution context.

### BF16 Conversion

BF16 is not IEEE F16. Conversion to F32 is the BF16 bit pattern shifted into
the high 16 bits of an IEEE F32 value:

```c
static inline float bf16_to_f32(uint16_t x) {
    uint32_t bits = (uint32_t)x << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}
```

Tests must include at least:

- `0x0000` -> `0.0f`
- `0x8000` -> `-0.0f`
- `0x3f80` -> `1.0f`
- `0xbf80` -> `-1.0f`
- `0x4000` -> `2.0f`
- `0x7f80` -> `+inf`
- `0xff80` -> `-inf`
- `0x7fc0` -> NaN
- one value where BF16 interpretation differs from IEEE F16 interpretation

### Validation Rules

The probe must fail closed for:

- null arena, view, row id, or output pointers;
- invalid arena state;
- `n_rows == 0` or `cols == 0`;
- `row_stride_elements < cols`;
- odd `arena_offset` or odd row stride in bytes;
- `byte_length` not divisible by 2;
- row id `>= view->rows`;
- `arena_offset + byte_length` overflow or out of arena range;
- row offset multiplication overflow;
- output byte calculation overflow;
- insufficient `out_bytes`.

Invalid row ids fail. They are not clamped.

### CUDA Semantics

The CUDA implementation must:

- call `cudaSetDevice(ds4_gpu_arena_gpu(arena))` before launching;
- derive input from the internal arena allocation plus `view->arena_offset`;
- not use `model_map`, `cuda_model_range_ptr`, GGUF offsets, shard file
  buffers, or `ds4_gpu_arena_read` as the compute source;
- use the default stream for Sprint 005 and document that stream-aware variants
  are deferred;
- allocate only bounded temporary buffers needed for row ids and F32 output;
- copy the small F32 output back to the caller-provided host buffer;
- synchronize or otherwise ensure the host output is valid before returning.

## Implementation

### Phase 0: Baseline Hygiene

**Files:**
- no source edits expected

**Tasks:**
- [ ] Confirm current local build targets used by Sprint 004 still build:
      `make cpu`, `make tools/ds4-v100-pack`,
      `make tools/ds4-v100-residency-smoke`,
      `make tests/pack_index_smoke`, and `make tests/gpu_arena_smoke`.
- [ ] Confirm current local focused tests still pass:
      `./tests/pack_index_smoke`, `./tests/gpu_arena_smoke`, and
      `tests/residency_smoke_synthetic.sh`.
- [ ] Record any existing unrelated failures before making code changes.

### Phase 1: Probe API And Host Reference

**Files:**
- `ds4_gpu.h`
- `ds4_gpu_arena_stub.c`
- `tests/bf16_probe_smoke.c`
- `Makefile`

**Tasks:**
- [ ] Add `ds4_gpu_bf16_matrix_view` and
      `ds4_gpu_arena_bf16_row_gather_f32`.
- [ ] Implement host-stub row gather from arena memory with exact BF16-to-F32
      conversion.
- [ ] Add synthetic unit coverage for canonical BF16 bit patterns,
      BF16-vs-F16 divergence, row 0, last row, repeated row ids, invalid row
      ids, invalid spans, alignment failures, and output-size failures.
- [ ] Add a local model-less test target for the BF16 probe.

### Phase 2: CUDA Synthetic Probe

**Files:**
- `ds4_cuda.cu`
- `tests/cuda_bf16_probe.c`
- `Makefile`

**Tasks:**
- [ ] Implement the CUDA version of
      `ds4_gpu_arena_bf16_row_gather_f32`.
- [ ] Add a direct CUDA-linked synthetic test that opens an arena, uploads a
      tiny BF16 matrix, gathers rows, and compares exact F32 outputs.
- [ ] Build the CUDA target with `CUDA_ARCH=sm_70`.
- [ ] Run the direct CUDA target on V100 and capture its output.

### Phase 3: Residency-Smoke Probe Mode

**Files:**
- `tools/ds4-v100-residency-smoke.c`
- `tests/residency_smoke_synthetic.sh`
- `Makefile`

**Tasks:**
- [ ] Add a focused BF16 probe mode to the residency smoke tool.
- [ ] Support a synthetic probe path that does not require the real model.
- [ ] Support a real `token_embd.weight` probe using the pack index semantic
      id `token_embd.weight`.
- [ ] For probe-only runs, upload only the required tensor span instead of all
      145 GiB when full residency is not under test.
- [ ] Report provider, semantic tensor id, owning GPU, arena offset, byte
      length, row ids, sample count, expected samples, actual samples, and
      result.
- [ ] Extend the synthetic residency smoke to exercise the probe mode.

### Phase 4: Cluster Validation And Close-Out

**Files:**
- `docs/sprints/drafts/SPRINT-005-CUDA-SYNTHETIC.log`
- `docs/sprints/drafts/SPRINT-005-BF16-PROBE-GGUF.log`
- `docs/sprints/drafts/SPRINT-005-BF16-PROBE-SHARD.log`
- `docs/sprints/drafts/SPRINT-005-GUARD.log`
- `docs/sprints/SPRINT-005-REPORT.md`
- `docs/sprints/SPRINT-005-FOLLOWUPS.md` if needed
- `docs/sprints/VISION.md`

**Tasks:**
- [ ] Build and run the local and CUDA test targets in the V100 cluster
      environment.
- [ ] Run the synthetic residency-smoke probe on the cluster.
- [ ] Run the real `token_embd.weight` probe using the existing source model
      and pack artifacts.
- [ ] Confirm the source-model generation guard still fails closed for normal
      generation startup.
- [ ] Write a sprint report with verdict, validation commands, artifacts, and
      deviations.
- [ ] Update `docs/sprints/VISION.md` with the Sprint 005 outcome.

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `ds4_gpu.h` | Modify | Add BF16 matrix view and arena row-gather API |
| `ds4_gpu_arena_stub.c` | Modify | Implement host-stub BF16 row gather |
| `ds4_cuda.cu` | Modify | Implement CUDA BF16 row gather from arena device memory |
| `tests/bf16_probe_smoke.c` | Create | Local model-less BF16 conversion and stub probe tests |
| `tests/cuda_bf16_probe.c` | Create | Direct CUDA synthetic arena probe test |
| `tools/ds4-v100-residency-smoke.c` | Modify | Add focused BF16 probe mode |
| `tests/residency_smoke_synthetic.sh` | Modify | Exercise synthetic probe mode |
| `Makefile` | Modify | Add new test/tool targets and clean rules |
| `docs/sprints/drafts/SPRINT-005-CUDA-SYNTHETIC.log` | Create | Cluster synthetic CUDA validation artifact |
| `docs/sprints/drafts/SPRINT-005-BF16-PROBE-GGUF.log` | Create | GGUF-provider cluster probe artifact |
| `docs/sprints/drafts/SPRINT-005-BF16-PROBE-SHARD.log` | Create | Shard-provider cluster probe artifact |
| `docs/sprints/drafts/SPRINT-005-GUARD.log` | Create | Source-model generation guard artifact |
| `docs/sprints/SPRINT-005-REPORT.md` | Create | Execution report |
| `docs/sprints/VISION.md` | Modify | Record planned sprint refinement now and outcome after execution |

## Definition of Done

- [ ] The BF16 matrix view and `ds4_gpu_arena_bf16_row_gather_f32` API exist.
- [ ] Host-stub and CUDA implementations have matching semantics.
- [ ] BF16 conversion is tested by exact bit pattern and is not interpreted as
      IEEE F16.
- [ ] Alignment, overflow, null pointer, invalid row, invalid span,
      zero-dimension, and output-size failures are tested and fail closed.
- [ ] CUDA implementation reads from arena device memory, not GGUF mmap,
      shard host buffers, `cuda_model_range_ptr`, or `ds4_gpu_arena_read`.
- [ ] Local model-less tests pass.
- [ ] Direct CUDA synthetic test builds with `CUDA_ARCH=sm_70` and passes on
      the V100 cluster.
- [ ] Residency-smoke BF16 probe mode passes on synthetic data.
- [ ] Real `token_embd.weight` probe passes on V100 and records resident-span
      facts plus expected and actual F32 samples.
- [ ] No persistent dequantized weight copy is introduced.
- [ ] Normal source-model generation remains guarded.
- [ ] `git diff --check` passes.
- [ ] Sprint report and vision update are written.

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| BF16 bytes are accidentally treated as IEEE F16 | High | High | Exact BF16 bit-pattern tests and BF16-vs-F16 divergence test |
| Probe accidentally reads host/GGUF bytes instead of arena device memory | Medium | High | CUDA API accepts arena/view, logs resident span, and forbids model-map helpers |
| API grows into execution context design | Medium | Medium | Host F32 output only, default stream only, no scheduler state |
| Probe-only mode becomes a partial-upload subsystem | Medium | Medium | Limit to one semantic tensor span and diagnostic use |
| Cluster is unavailable | Medium | Medium | Verdict becomes `EXTEND`; record exact missing validation |
| Real tensor sizes expose overflow not seen in tiny tests | Medium | High | Add explicit overflow checks and run real `token_embd.weight` probe |

## Security Considerations

- Treat probe arguments as untrusted diagnostic inputs.
- Validate semantic ids, row ids, offsets, byte lengths, alignment, and output
  sizes before launching CUDA.
- Keep model files and shard files read-only.
- Do not expose probe output through `ds4-server`.
- Do not export raw device pointers or arbitrary device memory reads.
- Preserve the source-model generation guard.

## Dependencies

- Sprint 004 runtime pack loading and per-GPU arena residency sidecar.
- Existing `pack-index.tsv` semantic id `token_embd.weight`.
- V100 cluster access for CUDA validation.
- Existing cluster pack artifacts under persistent scratch, or the ability to
  regenerate them from the source model.

## Open Questions

No blocking planning questions remain. The final plan resolves the draft
questions as follows:

- Output is host F32 only in Sprint 005.
- The API uses a descriptor/view.
- Invalid row ids fail; they are not clamped.
- Stream-aware and device-output variants are deferred.
- HC expansion and `ds4.c` source-layout embedding fixes are deferred unless
  they become strictly necessary to validate the row-gather probe.
