# Sprint 005: First Resident BF16 Compute Probe

## Overview

Sprint 004 proved that the DS4 V100 pack contract is structurally sound:
`pack-index.tsv` reconciles against the real source GGUF, emitted shards are
loadable, and all planned packed bytes can reside on the owning V100s. The
main blocker is no longer weight fit. It is whether runtime metadata, arena
offsets, source dtypes, and kernels agree on one real compute path.

Sprint 005 should close that gap with the smallest useful resident compute
contract: a BF16 row-gather probe that reads directly from `ds4_gpu_arena`,
converts source BF16 values to F32, and validates the result against a local
CPU reference. The probe must use resident arena bytes, not `model_map`, GGUF
offset helpers, or persistent dequantized mirrors.

The first tensor family should be `token_embd.weight`. It is global, clearly
documented as BF16 in `docs/architecture/DS4-V100-LAYOUT.md`, already assigned
to `gpu0`, large enough to prove real arena addressing, and simple enough to
validate with synthetic rows and one resident-cluster probe before any decode,
KV, scheduler, or MTP work is attempted.

This sprint stays diagnostic by design. It should not enable source-model
generation, widen into a general multi-GPU execution context, or repurpose the
existing mapped-weight embedding helpers as if they were already valid for
resident BF16 source packs.

## Use Cases

1. **Model-less local correctness**: as a developer, I can run a stub-backed
   BF16 probe on synthetic data without the 145 GiB source model or CUDA
   hardware and confirm exact row addressing plus BF16-to-F32 conversion.
2. **Direct CUDA unit validation**: as a CUDA developer, I can run a small real
   GPU target that opens an arena, uploads a tiny synthetic BF16 table, probes
   selected rows, and catches arena or conversion regressions without rerunning
   the full residency smoke.
3. **Resident token embedding proof**: as the operator, I can point a focused
   tool mode at `token_embd.weight`, selected token rows, and an existing pack
   directory and prove that the returned F32 values came from resident arena
   bytes on the owning V100.
4. **Reusable tensor contract**: as the next sprint, I inherit a narrow
   descriptor for "BF16 matrix view inside an arena" that later layer or
   scheduler work can wrap without re-deriving row offsets from raw shard files.
5. **Fail-closed diagnostics**: as the maintainer, I get explicit failures for
   wrong dtype, bad row id, bad arena span, undersized output buffer, or stale
   pack metadata instead of silent reuse of host-mapped bytes.

## Architecture

### Chosen Contract

Sprint 005 should introduce one reusable primitive plus one token-embedding
wrapper:

- A generic BF16 probe over a validated arena-backed matrix view.
- A thin `token_embd.weight` resolver that uses pack metadata to build that
  view and select token rows.

That split keeps the kernel contract reusable for later BF16 tensors while
keeping the first semantic target concrete and easy to validate.

### Data Flow

```text
pack-index.tsv + ds4_pack_lookup("token_embd.weight")
    |
    v
BF16 probe descriptor
(owning_gpu, arena_offset, row_count, row_elems, row_stride_bytes)
    |
    v
ds4_gpu_arena BF16 probe API
    |             |
    |             +--> host-stub implementation for local synthetic tests
    |
    +--> CUDA implementation reading resident device bytes
            |
            v
small F32 debug buffer
    |
    v
CPU reference compare + optional HC-repeat wrapper
```

### Design Rules

- The probe source of truth is `ds4_gpu_arena` memory, not `model_map`.
- BF16 values must be decoded by BF16 bit pattern, not by treating 16-bit lanes
  as IEEE F16 or `__half`.
- The output for this sprint should stay small and diagnostic: caller-provided
  F32 host buffers are sufficient. Device-resident output tensors can wait for
  the real execution context.
- HC repetition, if needed for a token-embedding smoke, should be a thin
  wrapper around the core row gather instead of being fused into the first
  resident kernel.
- The existing source-model generation guard stays in place. Sprint 005 proves
  one compute contract; it does not claim decode correctness.
- Any CPU diagnostic path that still reads source-layout token embeddings from
  `ds4.c` should become dtype-aware so BF16 is not silently routed through the
  current F16 conversion helper.

### Scope Boundary

Do not turn this sprint into a general `ds4_gpu_context[gpu]` refactor. The
probe API can accept a narrow descriptor and an arena handle without owning
streams, cuBLAS handles, relay buffers, or scheduler state. The goal is to
prove resident addressing plus BF16 conversion, not to design the whole runtime.

## Implementation

### Phase 1: BF16 Probe Contract And Reference Semantics

**Files:**
- `ds4_gpu.h`
- `ds4.c`

**Tasks:**
- [ ] Add a small BF16 probe descriptor and one narrow `ds4_gpu_arena` probe
      API for row gather into caller-provided F32 output.
- [ ] Define fail-closed validation for zero rows, out-of-range row indices,
      invalid arena byte spans, and undersized destination buffers.
- [ ] Add a CPU BF16 bit-conversion/reference path for exact synthetic expected
      values.
- [ ] If source-layout token embedding diagnostics in `ds4.c` remain reachable,
      make their conversion helper dispatch on tensor dtype instead of assuming
      F16.

### Phase 2: Stub And CUDA Arena Implementations

**Files:**
- `ds4_gpu_arena_stub.c`
- `ds4_cuda.cu`

**Tasks:**
- [ ] Implement identical stub and CUDA semantics for the BF16 probe API.
- [ ] Make the CUDA path read from `arena->ptr + arena_offset + row*stride`,
      not from `cuda_model_range_ptr` or GGUF mappings.
- [ ] Add an explicit CUDA BF16-to-F32 helper; do not reinterpret source BF16
      rows as `__half`.
- [ ] Keep the output path bounded: return only the requested F32 rows to the
      host and do not allocate persistent expanded weight buffers.
- [ ] Preserve the Sprint 004 residency APIs and keep normal source decode
      guarded off.

### Phase 3: Focused Tooling And Test Coverage

**Files:**
- `tools/ds4-v100-residency-smoke.c`
- `tests/gpu_arena_smoke.c`
- `tests/cuda_bf16_probe.c`
- `tests/residency_smoke_synthetic.sh`
- `Makefile`

**Tasks:**
- [ ] Add a focused BF16 probe mode to `tools/ds4-v100-residency-smoke`, with
      options such as selected semantic tensor id, row ids, and a narrow
      "probe-only" path that uploads only the needed BF16 span when full-model
      re-upload is unnecessary.
- [ ] Extend `tests/gpu_arena_smoke.c` to cover synthetic BF16 rows, exact
      expected floats, repeated row selection, invalid row ids, invalid range
      failures, and zero-length edge cases through the stub arena.
- [ ] Add `tests/cuda_bf16_probe.c` as a direct CUDA-linked arena unit target
      for the Sprint 004 follow-up: tiny synthetic BF16 table in, exact F32 row
      output back, no full shard dependency.
- [ ] Update `tests/residency_smoke_synthetic.sh` so the tool-level probe mode
      is exercised on synthetic pack fixtures.
- [ ] Keep `make test` scope unchanged unless the model-less default-target
      follow-up falls out naturally from the new targets without widening the
      sprint.

### Phase 4: V100 Validation And Artifacts

**Files:**
- `docs/sprints/drafts/SPRINT-005-RESIDENT-BF16-PROBE.log`

**Tasks:**
- [ ] Build the focused CUDA probe target with `CUDA_ARCH=sm_70`.
- [ ] Run the direct CUDA unit target on the V100 cluster with synthetic BF16
      weights and record the verdict.
- [ ] Run the residency smoke BF16 probe mode against `token_embd.weight` using
      real pack metadata and resident arena bytes, ideally on one or more known
      token ids with deterministic expected output.
- [ ] Record first-value comparisons and the row/gpu/span facts needed to show
      that the probe used the intended resident tensor bytes.
- [ ] Update `docs/sprints/VISION.md` with the actual Sprint 005 outcome after
      execution.

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `ds4_gpu.h` | Modify | Define the narrow BF16 probe descriptor and `ds4_gpu_arena` probe API. |
| `ds4.c` | Modify | Add the CPU BF16 reference helper and, if still needed, make source-layout token-embedding diagnostics dtype-aware. |
| `ds4_gpu_arena_stub.c` | Modify | Mirror the BF16 probe semantics for model-less local correctness tests. |
| `ds4_cuda.cu` | Modify | Implement resident BF16 row gather from device arena bytes with explicit BF16 conversion. |
| `tools/ds4-v100-residency-smoke.c` | Modify | Add a focused BF16 compute-probe mode that proves resident bytes can feed the new API. |
| `tests/gpu_arena_smoke.c` | Modify | Extend the existing stub smoke with BF16 probe coverage and fail-closed edge cases. |
| `tests/cuda_bf16_probe.c` | Create | Direct CUDA unit target for the arena probe on real GPUs. |
| `tests/residency_smoke_synthetic.sh` | Modify | Cover the tool-level BF16 probe mode without requiring the real model. |
| `Makefile` | Modify | Build the new CUDA probe target and any updated model-less smoke targets. |
| `docs/sprints/drafts/SPRINT-005-RESIDENT-BF16-PROBE.log` | Create | Capture cluster probe output, expected-vs-actual values, and resident-span facts. |

## Definition of Done

- [ ] A narrow BF16 resident-probe API exists on top of `ds4_gpu_arena` and is
      implemented with matching stub and CUDA semantics.
- [ ] The CUDA probe reads from device arena memory, not `model_map`, GGUF view
      helpers, or host-side shard buffers.
- [ ] BF16 source values are converted correctly to F32; they are not
      interpreted as IEEE F16 anywhere in the probe path.
- [ ] Model-less local tests pass for synthetic BF16 tables, repeated row
      selection, bad row ids, bad spans, and output-size validation.
- [ ] A direct CUDA unit target builds and passes on `sm_70` hardware.
- [ ] At least one resident `token_embd.weight` probe on the V100 cluster
      matches the CPU reference and records the arena span, row ids, and output
      sample values.
- [ ] No persistent dequantized weight copies are left resident after the
      probe.
- [ ] The source-model generation guard remains active for normal runtime
      startup and decode paths.
- [ ] Verification passes for the touched paths, including `make cpu`, the
      updated smoke/unit targets, and `git diff --check`.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| The sprint grows into a general execution-context refactor. | Medium | High | Keep the API descriptor narrow, host-output diagnostic, and explicitly sidecar to the production runtime. |
| BF16 rows are accidentally treated as F16 because existing code paths already use `__half`/F16 helpers. | High | High | Add explicit BF16 helpers in both CPU and CUDA code and test exact bit-pattern conversions. |
| The probe appears to work but still reads GGUF-mapped bytes instead of resident arena bytes. | Medium | High | Ensure the new CUDA entry point accepts arena offsets, not GGUF offsets, and log resident span facts in the cluster artifact. |
| Range/stride bugs only show up on real tensor sizes, not tiny synthetic fixtures. | Medium | Medium | Test both tiny synthetic rows and one real resident `token_embd.weight` row on cluster hardware. |
| Validation becomes too expensive if every probe requires re-uploading all 145 GiB of shards. | Medium | Medium | Add a focused probe-only mode that can upload just the required BF16 span when full-model residency is not the point under test. |
| Fixing shared CPU diagnostic helpers creates unintended behavior drift for existing q2/q4 paths. | Low | Medium | Make any `ds4.c` conversion change conditional on source tensor dtype and cover it with narrow tests only. |

## Security

- Treat probe inputs as untrusted diagnostic parameters. Validate semantic ids,
  row ids, byte spans, and output sizes before any device read or host copy.
- Keep shard and model inputs read-only. The probe should never mutate GGUFs,
  shard payloads, or pack metadata.
- Do not expose the BF16 probe through `ds4-server` or any network API in this
  sprint.
- Keep the result surface small: debug F32 outputs and log artifacts are
  sufficient; no arbitrary device pointer export should be introduced.
- Preserve the source-model generation guard so a successful probe cannot be
  mistaken for a supported decode path.

## Dependencies

- Sprint 004 pack reconciliation, shard validation, and `ds4_gpu_arena_*`
  residency sidecar.
- `ds4_pack_lookup` and the existing `pack-index.tsv` semantic tensor ids.
- `docs/architecture/DS4-V100-LAYOUT.md` as the BF16 token-embedding ownership
  and dtype anchor.
- `Makefile` support for standalone CPU/stub and CUDA smoke targets.
- V100 cluster access with `CUDA_ARCH=sm_70` for real CUDA validation.
- Real source-model artifacts only for the final resident probe, not for local
  synthetic correctness checks.

## Open Questions

1. Should the BF16 probe API return host F32 output only in Sprint 005, or is a
   device-output variant already necessary for Sprint 006?
2. Should the focused resident probe live inside
   `tools/ds4-v100-residency-smoke`, or should helper extraction justify a
   separate `tools/ds4-v100-bf16-probe` binary?
3. Is `token_embd.weight` sufficient as the only Sprint 005 tensor family, or
   should the sprint also probe one smaller BF16 tensor such as a compressor or
   indexer projection row to prove the contract is not embedding-specific?
4. Should the existing CPU token-embedding helper in `ds4.c` be corrected in
   the same sprint, or should Sprint 005 keep all source-format fixes confined
   to the new resident probe path?
