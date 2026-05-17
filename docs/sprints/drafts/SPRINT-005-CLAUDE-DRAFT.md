# SPRINT-005 — First Resident BF16 Compute Probe (CLAUDE draft)

**Status:** DRAFT 2026-05-17
**Predecessor:** SPRINT-004 (runtime pack loading and V100 device residency smoke)
**Successor:** Multi-GPU execution context and layer skeleton, gated on this sprint shipping a validated arena-resident compute contract.

---

## Overview

Sprint 004 proved structural residency: all 1328 source-faithful packed tensors
live on their owning V100 device arenas, reconciled against the source GGUF,
with spot checks and cross-provider byte identity passing. The runtime still
cannot perform any computation from those resident bytes. The existing CUDA
embedding kernels read from the legacy `model_map` range and interpret packed
16-bit values as IEEE FP16 (`__half`), while the source token embedding tensor
is BF16.

Sprint 005 closes the gap between "bytes are on the GPU" and "we can compute
from them." It introduces a narrow, bounded BF16 row-gather probe that reads
directly from a `ds4_gpu_arena` and produces F32 output. This proves:

1. the arena pointer contract works for compute, not just upload and readback;
2. BF16 bit patterns (not IEEE FP16) are correctly converted to F32;
3. row selection (token-id gather) from arena-resident bytes is validated
   end-to-end;
4. the host-stub path enables full local synthetic testing without CUDA;
5. the CUDA path produces correct results from device arena memory on V100.

The sprint intentionally stops at one tensor family (BF16 token embedding) and
one operation (row gather with BF16-to-F32 conversion). It does not introduce
the production multi-GPU execution context, layer scheduling, FP8 dequant,
MXFP4 unpack, hidden-context relay, KV allocation, or decode. The probe API
is diagnostic: it exists so the next sprint can design the real
`ds4_gpu_context[gpu]` against proven pointer semantics rather than guesses.

**Outcome contract:**

- **SHIP** if: a public BF16 arena probe API exists with both stub and CUDA
  implementations; local synthetic tests pass covering BF16 conversion
  accuracy, row gather, HC expansion, edge cases, and invalid ranges; CUDA
  synthetic test builds for `sm_70` and runs on V100; the probe reads from
  device arena memory, not from model_map or host buffers.
- **EXTEND** if: all local/stub tests pass but cluster V100 access is not
  available within the sprint window. Record the blocker and ship the API +
  local validation.
- **STOP** if: the arena pointer contract cannot support compute dispatch
  without an incompatible refactor, or BF16 source bytes do not match the
  expected bit pattern in the source GGUF.

---

## Use Cases

| Phase | Useful output if sprint stops here |
|---|---|
| P0 | Existing builds stay green; Sprint 004 artifacts remain valid. |
| P1 | A reference C BF16-to-F32 conversion function is available for test oracles. |
| P2 | `ds4_gpu_arena_bf16_row_gather` exists with a host-stub implementation; local tests validate BF16 conversion and row gather from synthetic arenas. |
| P3 | CUDA kernel implements the same probe reading directly from device arena pointers; CUDA synthetic test passes on V100. |
| P4 | Optional integration probe using a real resident embedding shard confirms bit-exact results against the reference on actual model bytes. |
| P5 | Sprint report documents the arena-compute contract and seeds the multi-GPU execution context design for Sprint 006. |

---

## Architecture

### Source Of Truth

`docs/architecture/DS4-V100-LAYOUT.md` states the token embedding is
`[4096 x 129280]` BF16 on gpu0. The source model at
`/models/DSv4-Flash-256e-fixed.gguf` contains BF16 bit patterns for this
tensor. The existing embedding kernels in `ds4_cuda.cu` cast these bytes as
`__half` (IEEE FP16), which is numerically wrong for BF16 source weights.

### BF16 Format

BF16 (Brain Float 16) uses 1 sign bit, 8 exponent bits, and 7 mantissa bits.
IEEE FP16 uses 1 sign bit, 5 exponent bits, and 10 mantissa bits. The
conversion from BF16 to F32 is a simple left-shift: pad the 16-bit BF16 value
with 16 zero bits in the mantissa to form the 32-bit IEEE F32 representation.

```c
static inline float bf16_to_f32(uint16_t x) {
    uint32_t bits = (uint32_t)x << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}
```

This is cheaper than FP16-to-F32 conversion and has no precision loss for
normal values.

### Probe API

The new probe API in `ds4_gpu.h`:

```c
int ds4_gpu_arena_bf16_row_gather(
        float              *out_f32,
        uint64_t            out_bytes,
        const ds4_gpu_arena *arena,
        uint64_t            weight_offset,
        uint64_t            weight_bytes,
        uint32_t            row_stride_elements,
        const uint32_t     *row_ids,
        uint32_t            n_rows,
        uint32_t            n_cols);
```

Semantics:

- Reads `n_rows` rows of `n_cols` BF16 elements each from the arena at
  `weight_offset`.
- Each row `i` is gathered from arena byte offset
  `weight_offset + row_ids[i] * row_stride_elements * 2`.
- Output is `n_rows * n_cols` F32 values written to `out_f32`.
- Returns 0 on success, nonzero on error.
- Validates: arena pointer validity, `weight_offset + weight_bytes <= arena_bytes`,
  each `row_ids[i] * row_stride_elements + n_cols <= total_elements`,
  `out_bytes >= n_rows * n_cols * 4`.

A second convenience wrapper adds HC expansion:

```c
int ds4_gpu_arena_bf16_embed_hc(
        float              *out_f32,
        uint64_t            out_bytes,
        const ds4_gpu_arena *arena,
        uint64_t            weight_offset,
        uint64_t            weight_bytes,
        uint32_t            n_vocab,
        uint32_t            n_embd,
        uint32_t            n_hc,
        uint32_t            token_id);
```

This gathers one row (`token_id`) and replicates it `n_hc` times into the
output buffer (the HC expansion pattern used by the DS4 hyper-connection
architecture). Output size is `n_hc * n_embd` F32 values.

### Module Boundaries

```text
ds4_gpu.h                  +ds4_gpu_arena_bf16_row_gather
                           +ds4_gpu_arena_bf16_embed_hc

ds4_gpu_arena_stub.c       stub implementation (host memory, reference C)
ds4_cuda.cu                CUDA implementation (device arena, BF16 kernel)

tests/bf16_probe_smoke.c   synthetic correctness tests
tests/bf16_probe_cuda.c    CUDA-linked synthetic test (small dimensions)
Makefile                   new test targets
```

The existing embedding kernels (`ds4_gpu_embed_token_hc_tensor` etc.) are not
modified. They continue to serve the legacy `model_map` path. The new probe
is additive and diagnostic. When Sprint 006 introduces the production
execution context, it will either replace the legacy kernels or redirect them
through the arena path — that design decision is explicitly deferred.

### Relationship To Existing Arena API

The probe reads from the arena using `arena->ptr + offset` (stub) or a
device pointer derived from the arena's CUDA allocation (CUDA). It does not
call `ds4_gpu_arena_read` (which copies to host); it computes directly on
resident bytes. This is the key contract Sprint 005 proves: arena memory is
usable as kernel input, not just as a storage target.

The CUDA implementation must:

- call `cudaSetDevice(arena->gpu)` before kernel launch;
- derive the device pointer from the arena's internal allocation;
- not copy arena bytes to a host buffer before computing;
- produce F32 output in a caller-provided device or host buffer (the probe
  is diagnostic, so host output for validation is acceptable).

### Output Target

The probe writes F32 output to a caller-provided host buffer (`float *out_f32`).
This simplifies validation: the caller can immediately compare against reference
values without a device-to-host copy in the test harness. The CUDA kernel uses
a small device buffer internally and copies results back, or (for the CUDA
synthetic test) the caller may provide a device pointer with a flag.

For the diagnostic purpose of this sprint, host-output is the default. The
production path in Sprint 006 will produce device-resident F32/F16 output.

---

## Implementation

### Phase 0: Build Hygiene And Sprint 004 Follow-Up

**Files:**

- `Makefile`
- `tests/ds4_test.c` (optional)

**Tasks:**

- [ ] Confirm `make cpu`, `make tools/ds4-v100-pack`,
      `make tools/ds4-v100-residency-smoke`, `make tests/pack_index_smoke`,
      `make tests/gpu_arena_smoke` all build and pass on the laptop.
- [ ] Confirm `tests/residency_smoke_synthetic.sh` passes.
- [ ] Confirm `git diff --check` is clean.
- [ ] Optionally add a model-less default `make test` target that runs only
      parser, arena, and synthetic tests (Sprint 004 follow-up: model-less
      default test target). This is included only if it helps validate Sprint
      005 without expanding scope; otherwise note it as still deferred.

**Kill gate:** none — this is housekeeping.

### Phase 1: Reference BF16-to-F32 Conversion And Test Oracle

**Files:**

- `tests/bf16_probe_smoke.c` (new)

**Tasks:**

- [ ] Implement a standalone C reference `bf16_to_f32` function.
- [ ] Build a table of known BF16 bit patterns and expected F32 results:
  - positive normal: `0x3F80` → 1.0f
  - negative normal: `0xBF80` → -1.0f
  - zero: `0x0000` → 0.0f
  - negative zero: `0x8000` → -0.0f
  - subnormal BF16: smallest positive subnormal `0x0001`
  - large value: `0x7F00` → max normal (~3.39e38)
  - NaN: `0x7FC0` → NaN (preserved)
  - Inf: `0x7F80` → +Inf
  - typical model weight patterns (small values near 0.01, 0.1)
- [ ] Validate that BF16 conversion is NOT equivalent to FP16 conversion —
      include a test value where BF16 and FP16 interpretations diverge and
      assert the BF16 path is used.

**Kill gate:** stop if BF16 source bytes in the actual source GGUF do not
match expected BF16 bit patterns (would indicate a manifest or loader
misidentification of the dtype).

### Phase 2: Host-Stub Arena Probe Implementation

**Files:**

- `ds4_gpu.h` (extend with new probe declarations)
- `ds4_gpu_arena_stub.c` (implement probe functions)
- `tests/bf16_probe_smoke.c` (extend with arena-based tests)
- `Makefile`

**Tasks:**

- [ ] Add `ds4_gpu_arena_bf16_row_gather` and `ds4_gpu_arena_bf16_embed_hc`
      declarations to `ds4_gpu.h`.
- [ ] Implement both in `ds4_gpu_arena_stub.c` using the reference
      `bf16_to_f32` and direct arena pointer access.
- [ ] Validate all bounds checks:
  - `weight_offset + weight_bytes > arena_bytes` → error
  - `row_ids[i] * stride + n_cols > total_rows * stride` → error
  - `out_bytes < n_rows * n_cols * 4` → error
  - NULL arena, NULL out, NULL row_ids → error
  - `n_rows == 0` → success with no work
  - invalid arena (previously failed upload) → error
- [ ] Test row gather with synthetic BF16 weights:
  - known pattern: row `i`, col `j` has value `bf16_encode(i * 100 + j * 0.1)`
  - gather specific rows in arbitrary order
  - gather the same row multiple times
  - gather row 0 and last row
- [ ] Test HC expansion: single token → replicated `n_hc` times.
- [ ] Test with `n_hc = 1` (degenerate case, no expansion).
- [ ] Confirm the probe never calls `ds4_gpu_arena_read` — it accesses
      `arena->ptr` directly (for the stub; CUDA uses the device pointer).
- [ ] Add `make tests/bf16_probe_smoke` target.

**Kill gate:** stop if the arena struct layout cannot be accessed for direct
pointer use without breaking the existing API contract.

### Phase 3: CUDA BF16 Arena Probe Implementation

**Files:**

- `ds4_cuda.cu` (add CUDA kernel and C wrapper)
- `tests/bf16_probe_cuda.c` (new; CUDA-linked test)
- `Makefile`

**Tasks:**

- [ ] Implement a CUDA kernel `bf16_row_gather_kernel` that:
  - reads BF16 values from `arena_device_ptr + weight_offset + row * stride * 2 + col * 2`
  - converts BF16 to F32 using `__uint_as_float((uint32_t)x << 16)`
  - writes F32 to the output buffer
- [ ] Implement the HC expansion variant kernel.
- [ ] Wire `ds4_gpu_arena_bf16_row_gather` and `ds4_gpu_arena_bf16_embed_hc`
      C entry points in `ds4_cuda.cu`:
  - `cudaSetDevice(arena->gpu)` before launch
  - derive device pointer from arena internal state
  - allocate a temporary device output buffer if the caller provides a host
    pointer, then `cudaMemcpy` results back
  - or directly write to a device buffer if a future flag indicates device output
- [ ] Error paths: return nonzero and do NOT corrupt arena state on invalid
      parameters.
- [ ] `tests/bf16_probe_cuda.c`:
  - create a small arena on GPU 0
  - upload known BF16 patterns
  - call the probe
  - compare output against the C reference oracle
  - test edge cases (out-of-range row, zero-length, invalid arena)
- [ ] Add `make tests/bf16_probe_cuda` target (CUDA-linked, builds with
      `CUDA_ARCH=sm_70` for V100).
- [ ] Confirm the kernel reads from device memory and does NOT stage through
      a host-side copy of the arena contents.

**Kill gate:** stop if the arena's internal device pointer cannot be safely
passed to a kernel without modifying the arena struct in a way that breaks
`ds4_gpu_arena_upload` or `ds4_gpu_arena_read`.

### Phase 4: Optional Real-Model Integration Probe (Cluster)

**Files:**

- `tools/ds4-v100-residency-smoke.c` (extend with `--bf16-probe` mode)
  OR a new `tests/bf16_probe_resident.c`
- `docs/sprints/drafts/SPRINT-005-BF16-PROBE.log` (artifact)

**Tasks:**

- [ ] On the V100 cluster, load the token embedding tensor into gpu0's arena
      using the existing residency smoke infrastructure.
- [ ] Call `ds4_gpu_arena_bf16_embed_hc` for a small set of known token IDs.
- [ ] Compare output against the reference C conversion applied to the same
      source bytes read directly from the GGUF mmap.
- [ ] Record results in `SPRINT-005-BF16-PROBE.log`:
  - token IDs tested
  - max absolute error (should be exactly 0.0 for BF16→F32)
  - any NaN or unexpected values
  - device, CUDA version, arena kind confirmation
- [ ] This phase is optional for SHIP if cluster access is not available.
      If skipped, the sprint is still SHIP with synthetic-only validation.

**Kill gate:** EXTEND if cluster access is unavailable. STOP if real model
bytes produce different results from the synthetic oracle in a way that
indicates the source dtype classification is wrong.

### Phase 5: Sprint Report And Follow-Ups

**Files:**

- `docs/sprints/SPRINT-005-REPORT.md` (create)
- `docs/sprints/SPRINT-005-FOLLOWUPS.md` (create)
- `docs/sprints/VISION.md` (update Sprint 005 entry with outcome)

**Tasks:**

- [ ] Record verdict (SHIP / EXTEND / STOP) with evidence pointers.
- [ ] Document the arena-compute pointer contract that Sprint 006 can rely on.
- [ ] Enumerate Sprint 006 surface: which tensors/operations to add next,
      what the `ds4_gpu_context[gpu]` API shape should look like based on the
      probe experience, and whether the legacy `model_map` embed path should
      be redirected.
- [ ] Acknowledge Sprint 004 follow-ups consumed or still deferred.
- [ ] Preserve source-model generation guard status.
- [ ] Update `docs/sprints/VISION.md` Sprint 005 entry with outcome.

**Kill gate:** none — always runs.

---

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `ds4_gpu.h` | Modify | Add `ds4_gpu_arena_bf16_row_gather` and `ds4_gpu_arena_bf16_embed_hc` declarations |
| `ds4_gpu_arena_stub.c` | Modify | Implement BF16 probe using host-stub arena pointer |
| `ds4_cuda.cu` | Modify | Implement CUDA BF16 row-gather and embed-HC kernels reading from device arena |
| `Makefile` | Modify | Add `tests/bf16_probe_smoke` and `tests/bf16_probe_cuda` targets |
| `tests/bf16_probe_smoke.c` | Create | Reference BF16 oracle and synthetic stub-arena probe tests |
| `tests/bf16_probe_cuda.c` | Create | CUDA-linked synthetic probe test |
| `tools/ds4-v100-residency-smoke.c` | Modify (optional) | Add `--bf16-probe` mode for cluster integration test |
| `docs/sprints/SPRINT-005-REPORT.md` | Create | Verdict and evidence |
| `docs/sprints/SPRINT-005-FOLLOWUPS.md` | Create | Sprint 006 surface |
| `docs/sprints/VISION.md` | Modify | Update Sprint 005 outcome |
| `docs/sprints/drafts/SPRINT-005-BF16-PROBE.log` | Create (cluster) | Integration probe results |
| `docs/architecture/DS4-V100-LAYOUT.md` | Read-only | Source dtype and topology reference |
| `ds4_pack.h` | Read-only | Pack entry schema for arena offset resolution |

---

## Definition Of Done

- [ ] `ds4_gpu_arena_bf16_row_gather` and `ds4_gpu_arena_bf16_embed_hc` are
      declared in `ds4_gpu.h` and implemented in both the host-stub and CUDA
      backends.
- [ ] The stub implementation reads directly from `arena->ptr`, not through
      `ds4_gpu_arena_read`.
- [ ] The CUDA implementation reads from device arena memory, not from
      `model_map`, host buffers, or `cudaMemcpy`-to-host-then-compute.
- [ ] BF16-to-F32 conversion produces bit-exact results matching the
      reference oracle for all tested patterns.
- [ ] A test value demonstrates that BF16 and FP16 interpretations diverge,
      and the probe uses BF16.
- [ ] Row gather validates: specific rows, arbitrary order, repeated rows,
      row 0, last row.
- [ ] HC expansion correctly replicates the gathered row `n_hc` times.
- [ ] Invalid inputs fail closed: out-of-range row IDs, insufficient output
      buffer, NULL pointers, invalid arena, `weight_offset + weight_bytes`
      exceeding arena size.
- [ ] `tests/bf16_probe_smoke` builds and passes on the laptop (CPU/stub,
      no CUDA required).
- [ ] `tests/bf16_probe_cuda` builds for `CUDA_ARCH=sm_70` and passes on V100
      (synthetic weights uploaded to arena, then probed).
- [ ] Existing tests (`make cpu`, `tests/pack_index_smoke`,
      `tests/gpu_arena_smoke`, `tests/residency_smoke_synthetic.sh`) remain
      green.
- [ ] `git diff --check` passes.
- [ ] The source-model generation guard in `ds4.c` is not touched.
- [ ] No persistent dequantized weight copies are created.
- [ ] No legacy `model_map`-based embedding kernels are modified.
- [ ] `SPRINT-005-REPORT.md` documents the verdict, pointer contract, and
      Sprint 006 surface.

---

## Risks And Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| Arena struct internal device pointer is not accessible without breaking encapsulation | Low | High | The CUDA implementation already owns the struct definition in `ds4_cuda.cu`; the probe kernel lives in the same translation unit and can access `arena->ptr` directly. Stub does the same. |
| BF16 source bytes are misidentified — the source GGUF actually stores FP16 for token embedding | Low | High | Sprint 002 manifest identifies the dtype as BF16; Phase 1 includes a divergence test that would catch FP16 masquerading as BF16. If hit, STOP and fix the manifest. |
| Kernel launch on arena device pointer causes illegal memory access | Low | High | Arena uses `cudaMalloc` (proven in Sprint 004); the pointer is valid device memory on the arena's GPU. Bracket with `cudaSetDevice`. Run CUDA memcheck in the synthetic test. |
| Scope creeps into production embed path or FP8/MXFP4 probes | Medium | Medium | Sprint 005 scope is exactly one probe (BF16 row gather). The convenience HC wrapper is included because it tests replication, not because it becomes the production path. Defer FP8/MXFP4 to Sprint 006+. |
| V100 cluster access is unavailable during sprint window | Medium | Low | Sprint ships SHIP on local + CUDA synthetic; cluster integration (Phase 4) is optional. Real-model byte validation strengthens confidence but is not required for the structural API contract. |
| Probe API shape locks in a bad signature for the execution context | Medium | Medium | The API is explicitly diagnostic. Sprint 006 designs the production context from scratch, using probe experience as input, not the probe signature as a constraint. |
| Model-less default test target work expands scope | Low | Low | Phase 0 includes this only if trivial (e.g., a `make test-unit` alias). Otherwise it stays deferred. |
| BF16 subnormals behave differently on V100 CUDA vs host | Low | Low | V100 supports F32 subnormals. BF16→F32 is a bitwise operation (shift), not an arithmetic one, so hardware FP behavior does not apply. Include a subnormal test case to confirm. |

---

## Security

- The probe reads from arena memory that was already validated during Sprint 004
  residency upload. No new file I/O or external input is introduced beyond
  what the existing arena API handles.
- All offset and length parameters are bounds-checked against the arena size
  before pointer arithmetic.
- Row IDs are validated against the declared vocabulary/row count before use
  as array indices.
- Output buffer size is validated before writing.
- No network API or server path is modified or exposed.
- The source-model generation guard remains active.
- The probe is read-only with respect to arena contents — it cannot modify
  resident weights.
- Integer overflow in `row_id * stride * 2` and `weight_offset + weight_bytes`
  is checked using overflow-safe arithmetic.

---

## Dependencies

- Sprint 004 SHIP: `ds4_gpu_arena_*` API, `ds4_pack.h`, host-stub and CUDA
  implementations, proven residency of BF16 token embedding on gpu0.
- Source model at `/models/DSv4-Flash-256e-fixed.gguf` (cluster only, Phase 4).
- Persistent scratch at `/srv/dev/ds4-sprint004` with emitted shards (cluster
  only, Phase 4).
- 8x V100-SXM2-32GB node with CUDA 12.x and `sm_70` (Phase 3 CUDA synthetic
  can run on any single V100; Phase 4 uses the full node).
- `docs/architecture/DS4-V100-LAYOUT.md` for token embedding shape and dtype.
- `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv` for token embedding arena
  offset (used only in Phase 4 integration test).

---

## Open Questions

1. **Generic row-gather vs. token-embedding-specific probe?** This draft
   proposes a generic `bf16_row_gather` that takes explicit dimensions, plus
   a convenience `bf16_embed_hc` wrapper for the HC expansion case. The
   generic version is more reusable in Sprint 006 when other BF16 tensors
   (output head, compressor weights) need the same operation. The
   token-embedding-specific wrapper proves the HC expansion contract without
   growing the generic API.

2. **Should the CUDA test live as a standalone binary or a mode of the
   residency smoke?** This draft proposes a standalone `tests/bf16_probe_cuda`
   for fast iteration and isolation from the full residency orchestration.
   The optional Phase 4 integration adds a `--bf16-probe` flag to the
   existing smoke tool for convenience, but this is additive.

3. **F32-only output or also FP16?** This sprint produces F32 output only.
   The production embedding path will likely produce F32 for HC expansion
   (matching the existing kernel's output), and the next sprint can add an
   optional FP16 output path if the HC relay or norm inputs need it. Adding
   FP16 output here would require testing an additional conversion and is
   unnecessary for the correctness proof.

4. **Should model-less default testing land in Sprint 005?** Only if it is
   trivial (a Makefile alias). The core sprint value is the BF16 compute
   probe, not test infrastructure. If it takes more than a few lines in
   Phase 0, defer it.

5. **Should Sprint 005 also add a direct CUDA arena unit target (Sprint 004
   follow-up)?** No. The CUDA probe test (`tests/bf16_probe_cuda`) is a
   stronger validation than a bare arena unit test because it exercises both
   the arena pointer contract and a real kernel. The bare arena unit test
   remains a nice-to-have follow-up.
