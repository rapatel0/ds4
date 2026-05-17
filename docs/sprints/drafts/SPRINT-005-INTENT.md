# Sprint 005 Intent: First Resident BF16 Compute Probe

## Seed

Continue from `docs/sprints/VISION.md` and plan Sprint 005:
"First Resident Source-Format Compute Probe." The sprint should execute a
narrow compute path directly from V100-resident packed bytes without enabling
full source-model decode.

## Context

- Sprints 001-004 proved source inventory, pack manifest generation,
  deterministic per-GPU shard layout, runtime pack reconciliation, and full
  device residency on all 8 V100s.
- Source-model generation remains intentionally guarded. There is still no
  source-format decode, prefill, KV allocation, MTP, or throughput benchmark.
- Weight VRAM fit is no longer the primary blocker. The tightest residency
  smoke left more than the planned 3 GiB reserve on a 32 GB V100.
- The next risk is the runtime contract between pack entries, device arena
  pointers, source dtypes, and kernels.
- The first compute probe should be useful but bounded enough to validate
  locally with the host-stub arena and on-cluster with a small CUDA synthetic
  test before touching full decode.

## Recent Sprint Context

- Sprint 002 recognized the native source layout and recorded source dtypes:
  BF16, F32, I32, F8_E4M3_B128, and MXFP4.
- Sprint 003 created `tools/ds4-v100-pack` and a deterministic
  `pack-index.tsv`.
- Sprint 004 added `ds4_pack`, upload-only `ds4_gpu_arena_*` APIs,
  `tools/ds4-v100-residency-smoke`, full shard emission on persistent cluster
  scratch, and 8-GPU residency validation.
- Existing CUDA embedding helpers still read from `model_map` ranges, not from
  V100 pack arenas. The existing embed kernel treats packed 16-bit values as
  IEEE F16, while the source token embedding is BF16.

## Vision Context

North Star: build a DeepSeek V4 Flash appliance for the 8x V100-SXM2-32GB
cluster that runs the high-intelligence source quantized model from pure
device-resident packs by default, preserves model quality, and reaches a
verified deployed serving path before broad throughput tuning.

Sprint 005 sits immediately after pack residency and before the production
multi-GPU execution context. It should prove a narrow source-format compute
contract from resident bytes, not full layer scheduling.

Parking-lot candidates now actionable:

- `docs/sprints/SPRINT-004-DEFERRED.md`: First Source-Format Math Probe.
- `docs/sprints/SPRINT-004-FOLLOWUPS.md`: Direct CUDA arena unit target.
- `docs/sprints/SPRINT-004-FOLLOWUPS.md`: Model-less default test target may
  be included only if it helps validate this sprint without expanding scope.

## Relevant Codebase Areas

- `ds4_gpu.h`: public CUDA/Metal/stub GPU API and residency-only arena API.
- `ds4_cuda.cu`: CUDA arena implementation and existing embedding kernels.
- `ds4_gpu_arena_stub.c`: CPU/stub arena implementation used for local tests.
- `tools/ds4-v100-residency-smoke.c`: current pack upload and spot-check tool.
- `tests/gpu_arena_smoke.c`: synthetic arena behavior test.
- `tests/residency_smoke_synthetic.sh`: synthetic pack-index smoke.
- `Makefile`: local and CUDA targets for new smoke tests.
- `docs/architecture/DS4-V100-LAYOUT.md`: source dtype and topology anchor.

## Constraints

- Preserve the source-model generation guard.
- Do not enable full decode, prefill, KV allocation, MTP, or slot scheduling.
- Do not create persistent dequantized weight copies.
- Keep the new API narrow and diagnostic until the production multi-GPU
  execution context exists.
- Do not introduce C++.
- Local validation must not require the 145 GiB model file or CUDA hardware.
- CUDA validation should be synthetic first, and may optionally run against a
  real resident shard if cluster access is available.
- Do not touch unrelated untracked `logs/`.

## Success Criteria

This sprint succeeds if it implements and verifies a resident BF16 row-gather
or token-embedding probe:

- A caller can bind or reference bytes inside a `ds4_gpu_arena` and launch a
  small BF16-to-F32 compute operation from those resident bytes.
- The same public probe has a host-stub implementation for local synthetic
  correctness tests.
- The CUDA implementation reads from device arena memory, not from the GGUF
  mmap or host-side shard buffer.
- Synthetic tests cover BF16 conversion, row selection, repeated HC expansion
  if included, invalid token/range handling, and arena range errors.
- A focused CUDA target can be built for `CUDA_ARCH=sm_70` and run on the V100
  cluster when available.
- `docs/sprints/VISION.md` is updated with the Sprint 005 outcome after
  execution.

## Verification Strategy

- Reference implementation: local C BF16-to-F32 conversion for synthetic
  weights with exact expected values.
- Spec/documentation: `docs/architecture/DS4-V100-LAYOUT.md` states token
  embedding is `[4096 x 129280]` BF16 and should stay source-faithful first.
- Edge cases:
  - zero-length and out-of-range arena spans fail closed;
  - invalid token id fails or clamps only if explicitly specified;
  - output tensor/host buffer sizing is validated;
  - BF16 bit patterns, not IEEE F16 bit patterns, are converted.
- Testing approach:
  - local CPU/stub unit test for the API and BF16 conversion;
  - CUDA synthetic test on V100 with small dimensions;
  - existing `make cpu`, pack-index, arena, and residency synthetic tests;
  - `git diff --check`.

## Uncertainty Assessment

- Correctness uncertainty: Medium. BF16 conversion is simple, but the new
  contract must prove it is using arena-resident bytes and failing closed on
  range mistakes.
- Scope uncertainty: Medium. The sprint should stop at one probe and resist
  becoming the full execution context.
- Architecture uncertainty: Medium. The API should not hard-code a shape that
  blocks later descriptor-based execution, but it should not overdesign the
  future scheduler either.

## Open Questions

1. Should Sprint 005 expose a generic arena BF16 row-gather primitive, or a
   token-embedding-specific BF16-to-HC probe?
2. Should the CUDA synthetic test live as a standalone test binary, or should
   `tools/ds4-v100-residency-smoke` gain an optional compute-probe mode?
3. Should the first probe produce F32 debug output only, or also FP16 output for
   later hidden-context relay compatibility?
4. Should model-less default testing be in Sprint 005, or left as a follow-up
   so this sprint stays focused on resident compute?
