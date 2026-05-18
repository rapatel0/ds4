---
sprint: 008
title: Source Oracle Harness And V100 KV Anchors
date: 2026-05-18
seed: run sprint-plan and sprint-execute toward the DS4 V100 appliance vision
---

# Sprint 008 Intent

## Seed Prompt

Continue the sprint sequence until the DS4 V100 appliance vision is fully
realized. Sprint 007 shipped the guarded source-layout oracle; the next sprint
must move toward prompt prefill, compressed KV, indexer state, and the first
production-relevant device-side anchors without weakening the source-layout
guard.

## Orientation Summary

- Sprints 001-004 proved source model inventory, pack planning, deterministic
  per-GPU shard layout, and full 8x V100 device residency.
- Sprint 005 proved resident BF16 gather/expand only; V100 has no native BF16,
  FP8, or FP4 tensor-core path, so production math must target FP16 HMMA,
  validated low-bit/integer kernels, and FP32 only for control/reduction
  paths.
- Sprint 006 shipped the V100 execution context, layer ownership skeleton, CUDA
  device facts, reserve enforcement, and HC relay smoke while keeping source
  generation guarded.
- Sprint 007 shipped a guarded CPU source-layout oracle that matched the
  official `short_reasoning_plain` first token exactly. It corrected MXFP4
  low-half/high-half row semantics and restored F16 KV as the source
  correctness baseline.
- No `SPRINT-008.md` exists yet. `docs/sprints/VISION.md` says Sprint 008 is
  the next correctness milestone for prompt prefill, compressed KV, indexer
  state, and first production-relevant device-side source-format anchors.

## Relevant Codebase Areas

- `ds4.c`, `ds4.h`, `ds4_cli.c`: source-layout oracle, CPU/session prefill,
  compressed KV/indexer reference paths, CLI diagnostics.
- `ds4_source_formats.c`, `ds4_source_formats.h`,
  `tests/source_dtypes_smoke.c`: source dtype helpers and MXFP4/F8/BF16 tests.
- `ds4_v100_context.c`, `ds4_v100_context.h`,
  `ds4_v100_context_cuda.cu`: V100 topology, policy, stage ownership, CUDA
  context and relay surfaces.
- `ds4_cuda.cu`, `ds4_gpu.h`: CUDA kernels and low-level device tensor
  operations.
- `tests/test-vectors/official.vec` and
  `tests/test-vectors/official/*.official.json`: official selected-token
  fixtures.
- `tests/cuda_v100_context_smoke.c`, `tests/cuda_long_context_smoke.c`,
  `tests/v100_context_smoke.c`: existing CUDA/context smoke patterns.
- `docs/architecture/DS4-V100-LAYOUT.md`: architecture anchor for memory,
  dtype, topology, kernel selection, and tensor-parallel deferral.

## Constraints

- Preserve normal source-layout generation fail-closed until production V100
  correctness is proven beyond diagnostic oracle mode.
- Do not introduce a broad FP32 fallback for BF16/FP8/MXFP4 model math.
- Do not materialize persistent dequantized copies of large source tensors.
- Keep F16 KV as the baseline correctness mode; F8 KV is a later optimization
  gate.
- Keep layer-owned topology as the default. Tensor parallelism and MTP remain
  later exceptions.
- The cluster target is 8x V100-SXM2-32GB in namespace `llm`, pod
  `llamacpp-build-8gpu`; use archived cluster evidence where possible.
- Follow repo git rules: explicit `git add` paths only; ignore unrelated
  untracked `logs/`.

## Success Criteria

- A `SPRINT-008.md` sprint plan exists and references the vision, Sprint 007
  report/follow-ups, and the architecture layout document.
- Official-vector oracle comparison is automated enough that the
  `short_reasoning_plain` selected-token check is not a manual JSON inspection.
- Source-layout guard behavior has focused tests or a durable validation
  script covering normal guard failure and diagnostic-only oracle behavior.
- The V100 context exposes a concrete F16 KV budget/admission surface for the
  layer-owned schedule, including SWA-only, ratio-4/indexer, and ratio-128
  layer classes.
- At least one device-side anchor for Sprint 008 exists and validates locally
  or on the V100 cluster without claiming production decode throughput.
- Validation logs and a Sprint 008 report/follow-up artifact record what is
  proven and what remains.

## Verification Strategy

- Local: `make cpu`, `make ds4`, `make tests/source_dtypes_smoke`,
  `make tests/v100_context_smoke`, `./tests/source_dtypes_smoke`,
  `./tests/v100_context_smoke`, and `git diff --check`.
- Cluster: build CPU/CUDA relevant targets on `llamacpp-build-8gpu`; run the
  official source-oracle vector check against
  `/models/DSv4-Flash-256e-fixed.gguf`; run any CUDA V100 context/KV anchor
  smoke added by this sprint.
- Artifacts: place local and cluster logs under
  `docs/sprints/drafts/SPRINT-008-*`.

## Actionable Deferred Items

- `docs/sprints/SPRINT-007-DEFERRED.md`: Prefill and KV are explicitly targeted
  at Sprint 008 after the Sprint 007 oracle verdict.
- `docs/sprints/SPRINT-007-DEFERRED.md`: Device-side oracle reads are targeted
  at Sprint 008 or the first production-kernel sprint.
- `docs/sprints/SPRINT-007-DEFERRED.md`: Production FP8/MXFP4 kernels are
  Sprint 008+, but should enter only as bounded anchors, not broad runtime
  claims.

## Actionable Follow-Ups

- `docs/sprints/SPRINT-007-FOLLOWUPS.md`: add direct MXFP4 parity hardening so
  nibble ordering cannot regress.
- `docs/sprints/SPRINT-007-FOLLOWUPS.md`: add a small official-vector runner
  around the `--dump-logprobs` diagnostic command.
- `docs/sprints/SPRINT-007-FOLLOWUPS.md`: add targeted source-oracle guard
  tests for normal generation and diagnostic-session behavior.

## Vision Context

The North Star is a pure device-resident DS4 V100 appliance that runs the
high-intelligence source quantized model on 8x V100-SXM2-32GB, preserves model
quality, reaches verified deployment, and only then broadens throughput tuning.
Sprint 008 sits between the source-layout correctness oracle and deployment:
it must convert the oracle into repeatable validation and establish the KV and
device-side surfaces that production decode/prefill will use.

## Uncertainty Assessment

| Area | Level | Notes |
|---|---|---|
| Correctness | Medium | Sprint 007 gives a strong first-token oracle, but prefill/KV and device-side anchors still need stronger validation. |
| Scope | High | Full prefill plus production kernels is too large; this sprint should define a bounded correctness slice. |
| Architecture | Medium | Layer-owned topology and F16 KV are agreed; exact device-side anchor choice should stay conservative. |

## Open Questions

1. Should Sprint 008's device-side anchor start with KV budget/admission and
   guard tests, or immediately include a CUDA kernel probe for one source
   tensor family?
2. Is `--dump-logprobs` an acceptable durable diagnostic unlock, or should
   Sprint 008 add the explicit oracle unlock token originally considered in
   Sprint 007?
3. Which official vectors beyond `short_reasoning_plain` are cheap enough to
   run on the cluster during this sprint?
