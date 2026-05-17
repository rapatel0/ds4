---
sprint: 005
title: First Resident BF16 Compute Probe
status: draft
date: 2026-05-17
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-005-INTENT.md
---

# SPRINT-005: First Resident BF16 Compute Probe

## Overview

Sprint 004 established the memory foundation: the DS4 source model is packed into per-GPU shards and uploaded to 8x V100 device arenas. The runtime now has bit-faithful residency, but it lacks the compute contract to use those bytes.

Sprint 005 implements the first narrow compute path directly from V100-resident packed bytes. The target is the BF16 token embedding table on `gpu0`. Instead of enabling full decode, this sprint produces a diagnostic "probe" that gathers rows from the resident arena and converts them to F32/F16. This proves the pointer, descriptor, and kernel contract on a bounded tensor family before the multi-GPU layer scheduler is introduced in Sprint 006.

The sprint is successful when the repo can:
1. bind a source-layout tensor inside a `ds4_gpu_arena`;
2. launch a BF16 row-gather probe that reads directly from resident device memory;
3. verify BF16-to-F32/F16 numerical correctness on both host-stub and CUDA backends;
4. handle arena range errors and invalid token IDs without crashing;
5. run a synthetic compute smoke test locally and on the V100 cluster.

## Use Cases

1. **Resident Row Gather**: as a developer, I can specify a list of token IDs and get the corresponding BF16 embedding rows gathered from the GPU arena and converted to F32 for inspection.
2. **Local Logic Validation**: as a maintainer, I can verify the BF16 conversion and gather logic on my laptop using the host-stub arena without needing CUDA hardware.
3. **Cluster Compute Smoke**: as the operator, I can run a compute-probe mode in the residency smoke tool to verify that `gpu0` is correctly executing BF16 math from resident bytes.
4. **Contract Baseline**: as the architect of the next sprint, I have a working pattern for descriptor-based tensor access that I can extend to FP8 and MXFP4 GEMMs.

## Architecture

### Probe API

Add a narrow, diagnostic probe API to `ds4_gpu.h`. This is not the final production scheduler, but a tool to validate the residency-to-compute path.

```c
typedef struct ds4_gpu_probe_descriptor {
    uint32_t tensor_id;
    uint32_t rows;
    uint32_t cols;
    uint32_t stride_bytes;
    size_t   arena_offset;
    // ... source dtype and layout info
} ds4_gpu_probe_descriptor;

// Launch a BF16 row-gather probe
ds4_status ds4_gpu_probe_bf16_gather(
    ds4_gpu_arena *arena,
    const ds4_gpu_probe_descriptor *desc,
    const uint32_t *token_ids,
    uint32_t num_tokens,
    float *out_f32,
    ds4_gpu_stream stream
);
```

### BF16 Conversion

DeepSeek V4 Flash uses BF16 for embeddings. V100 does not have a native BF16 gather-and-convert-to-F32 instruction in the way later architectures do. The implementation must:
- Extract the high 16 bits for the exponent and mantissa.
- Shift/pad to form an IEEE F32.
- Handle BF16 subnormals and NaNs if necessary, though typical weights are well-behaved.

### Backends

| Backend | Implementation | Purpose |
|---|---|---|
| **Host-Stub** | `ds4_gpu_arena_stub.c` | CPU-based BF16-to-F32 gather loop for local tests. |
| **CUDA** | `ds4_cuda.cu` | `sm_70` kernel reading from `arena->device_ptr + offset`. |

The CUDA kernel should use a simple 1D or 2D grid where threads handle individual values or rows, optimized for the V100's HBM2 bandwidth.

## Implementation

### Phase 1: BF16 Logic and Unit Tests

**Files:**
- `ds4_gpu.h`
- `tests/bf16_smoke.c` (new)

**Tasks:**
- [ ] Implement a standalone `bf16_to_f32` conversion helper for the host.
- [ ] Unit test with exact bit patterns: 0.0, 1.0, -2.0, max BF16, subnormals.
- [ ] Verify that BF16 is treated as the top 16 bits of an F32, not IEEE F16 (half-precision).

### Phase 2: Host-Stub Probe Implementation

**Files:**
- `ds4_gpu_arena_stub.c`
- `tests/gpu_probe_smoke.c` (new)

**Tasks:**
- [ ] Implement `ds4_gpu_probe_bf16_gather` in the stub.
- [ ] Add bounds checking: `arena_offset + rows * stride_bytes` must be within arena size.
- [ ] Handle invalid `token_ids` (e.g., `>= rows`) by clamping or failing.
- [ ] Create a synthetic BF16 arena in the test and verify gathered F32 values match expectation.

### Phase 3: CUDA Probe Implementation

**Files:**
- `ds4_cuda.cu`
- `Makefile`

**Tasks:**
- [ ] Implement the `ds4_gpu_probe_bf16_gather` CUDA kernel for `sm_70`.
- [ ] Ensure the kernel reads directly from the `ds4_gpu_arena` device pointer.
- [ ] Use `cudaMemcpyAsync` to transfer token IDs to the device if needed, or use a small constant-memory buffer.
- [ ] Use `cudaMemcpyAsync` to read back the F32 results to the host `out_f32` buffer.
- [ ] Synchronize on the provided `stream`.

### Phase 4: Integration into Residency Smoke

**Files:**
- `tools/ds4-v100-residency-smoke.c`
- `tests/residency_compute_smoke_synthetic.sh` (new)

**Tasks:**
- [ ] Add `--compute-probe` flag to the residency smoke tool.
- [ ] When enabled, after successful upload of `gpu0` shards, identify the `token_embedding` tensor.
- [ ] Launch a probe for a few deterministic token IDs (e.g., 0, 1, 12345).
- [ ] For synthetic runs, verify results match the synthetic pattern.
- [ ] For real-model runs, report the first 8 F32 values of the gathered rows.

### Phase 5: Cluster Validation

**Tasks:**
- [ ] Build for `CUDA_ARCH=sm_70` on the cluster.
- [ ] Run residency smoke with `--compute-probe` on the 8x V100 node.
- [ ] Archive the probe output for both synthetic and real shard providers.
- [ ] Verify that the probe executes on `gpu0` while other GPUs remain in their resident idle state.

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `ds4_gpu.h` | Modify | Add probe API and BF16 descriptors |
| `ds4_gpu_arena_stub.c` | Modify | Implement CPU-side probe logic |
| `ds4_cuda.cu` | Modify | Implement sm_70 BF16 gather kernel and probe API |
| `tools/ds4-v100-residency-smoke.c` | Modify | Add compute-probe diagnostic mode |
| `tests/bf16_smoke.c` | Create | BF16 bit-pattern unit tests |
| `tests/gpu_probe_smoke.c` | Create | Unit tests for the probe API and stub backend |
| `tests/residency_compute_smoke_synthetic.sh` | Create | E2E synthetic compute smoke |
| `Makefile` | Modify | Add new tests and ensure sm_70 build |

## Definition Of Done

- [ ] `ds4_gpu_probe_bf16_gather` is implemented for both host-stub and CUDA backends.
- [ ] BF16-to-F32 conversion is verified against exact bit patterns (not F16).
- [ ] Probe correctly identifies and gathers rows from a `ds4_gpu_arena`.
- [ ] Range checks prevent out-of-bounds arena access.
- [ ] Synthetic tests pass locally with `make cpu`.
- [ ] CUDA implementation builds for `sm_70`.
- [ ] Cluster smoke test successfully gathers and reports embedding values from a resident V100 arena.
- [ ] `docs/sprints/VISION.md` is updated with the Sprint 005 results.

## Risks

- **BF16 vs IEEE F16**: Mistaking the source format for standard FP16 would cause immediate numerical failure. Mitigation: explicitly test BF16 bit patterns.
- **Arena Offsets**: The pack index uses absolute arena offsets; the probe must respect these exactly. Mitigation: share descriptor logic with the reconciliation tool.
- **Kernel Performance**: While not a throughput sprint, extremely slow gather could mask other issues. Mitigation: use standard 2D grid patterns for V100.
- **Source Model Generation Guard**: Must remain active. The probe is a diagnostic side-channel, not an enablement of the full generator.

## Security

- No changes to the source-model generation guard.
- Bounds check `token_ids` against the descriptor `rows`.
- Bounds check `arena_offset + size` against the allocated arena.
- Do not log or expose raw weight bytes in production logs; only report diagnostic probe results during smoke tests.

## Dependencies

- Sprint 004 successful residency (arenas and upload logic).
- `pack-index.tsv` with correct BF16 embedding metadata.
- V100 cluster for CUDA validation.

## Open Questions

1. Should the probe support F16 output for later hidden-context relay compatibility, or is F32 sufficient for this diagnostic step?
2. Should we include a small F32 control tensor probe (e.g., `output_norm`) to isolate BF16 conversion issues from general arena access issues?
3. How should the probe handle tokens that are split across multiple GPUs (if any)? (Currently, the layout suggests embeddings are purely on `gpu0`).
