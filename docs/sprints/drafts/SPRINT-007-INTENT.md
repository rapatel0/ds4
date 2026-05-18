# Sprint 007 Intent: Source-Layout Single-Slot Decode Correctness

## Seed

Plan the next sprint after Sprint 006. The goal from `docs/sprints/VISION.md`
is to run a guarded one-slot, small-context decode path and compare logits or
generated tokens against a trusted reference.

This sprint must preserve the V100 precision policy established in Sprint 006:
V100 does not have native BF16, FP8, or FP4 tensor-core execution. Source BF16
large weights are conversion inputs, not a BF16 execution path. Production
dense GEMMs target FP16 tensor cores with FP32 accumulation, low-bit source
packs feed explicitly registered unpack/dequant/integer kernels, and FP32 is
reserved for control, reduction, diagnostics, and the correctness oracle.

## Orientation Summary

- Sprints 001-006 proved source inventory, pack manifesting, deterministic
  pack indexes, full 8x V100 residency, a resident BF16 gather/expand probe,
  and the sidecar V100 execution context with descriptor policy, topology
  checks, HC relay smoke, and no-math layer skeleton.
- Normal source-model generation is still guarded by design in `ds4_engine_open`
  because source FP8/MXFP4 execution kernels are not wired into the runtime.
- Existing CPU decode helpers are mostly legacy-layout oriented: embedding and
  output assume F16/Q8_0, attention/shared expert projections assume Q8_0, HC
  control matvecs assume F16, and source BF16/F32/F8_E4M3_B128/MXFP4 dispatch is
  incomplete.
- Official DeepSeek V4 Flash API logprob fixtures already exist under
  `tests/test-vectors/`, but they provide token/top-logprob slices rather than
  full logits.
- `docs/architecture/DS4-V100-LAYOUT.md` remains the topology and dtype anchor:
  baseline scheduling is layer-owned, HC-only boundary transfer, FP16
  activations, F16 KV first, source FP8/MXFP4 packs, no persistent dequantized
  weight copies, and tensor-parallel exceptions deferred.

## Vision Context

The North Star is a narrow DeepSeek V4 Flash appliance for the 8x
V100-SXM2-32GB cluster that runs the high-intelligence source quantized model
from pure device-resident packs by default.

Sprint 007 is the correctness gate between the shipped V100 execution skeleton
and later prefill/KV/deployment work. It should establish a trustworthy
source-layout reference path and a guarded way to compare local output against
official vectors. It should not optimize throughput or present the appliance as
deployed.

Parking-lot items now relevant:

- Source-layout decode and first-token generation.
- Source-layout embedding dtype cleanup.
- F32 control tensor execution for HC/router/output control.
- BF16 output-head support for source layout.
- FP8 and MXFP4 source compute primitives, at least as exact reference/oracle
  kernels.
- Guarded logprob vector comparison using `tests/test-vectors/official.vec`.

Parking-lot items still deferred:

- Production V100 FP8/MXFP4 CUDA performance kernels.
- Prompt prefill, compressed KV growth, and long-context admission.
- Multi-slot scheduling and wavefront throughput.
- MTP/speculative decoding.
- Tensor-parallel exceptions such as vocab-parallel output head.
- Server deployment and operational health checks.

## Relevant Codebase Areas

- `ds4.c`
  - source-layout detection and guard: `model_uses_v100_source_layout`,
    `weights_validate_v100_source_layout`, `ds4_engine_open`;
  - current CPU reference/decode helpers: `embed_token_f16`, `matvec_f16`,
    `matvec_q8_0`, `matvec_any`, `forward_first_token_cpu`,
    `output_logits_one`;
  - attention/FFN paths that currently assume Q8_0 for legacy model tensors;
  - official vector runner hooks through `ds4_test` / logprob-vector support.
- `ds4_v100_context.h`, `ds4_v100_context.c`, `ds4_v100_context_cuda.cu`
  - Sprint 006 descriptor policy, stage map, and relay contract that must stay
    fail-closed.
- `gguf-tools/quants.h`, `gguf-tools/quants.c`,
  `gguf-tools/deepseek4-quantize.c`
  - existing GGUF type constants and float/BF16 conversion helpers; useful
    references for source dtype behavior.
- `tests/test-vectors/`
  - official API prompts, compact fixture, and JSON logprob/top-token slices.
- `docs/architecture/DS4-V100-LAYOUT.md`
  - source dtype and runtime dtype mapping, layer schedule, memory estimates,
    and the explicit tensor-parallel alternatives.

## Constraints

- Do not remove the normal source-layout generation guard. Any decode/logprob
  path added in this sprint must be opt-in, diagnostic, and visibly guarded.
- Do not claim broad FP32 runtime support. The source-layout CPU/reference path
  may use FP32 as the oracle, but production V100 policy remains FP16 HMMA plus
  registered low-bit kernels.
- Do not add persistent dequantized FP16/F32 copies of large source weights to
  the runtime pack format.
- Do not introduce host-backed, SSD-backed, managed-memory, or offloaded
  successful runtime paths.
- Do not implement prefill, compressed KV long-context behavior, MTP,
  multi-slot batching, server deployment, or throughput benchmarks.
- Avoid huge CPU inference runs on macOS. Full source-model checks should run
  on the V100 cluster or be kept to metadata/synthetic fixtures locally.
- Treat MXFP4/F8_E4M3_B128 format uncertainty as a hard correctness risk. If
  exact source dequant semantics cannot be established, stop at a narrower
  dtype-primitive sprint and document the blocker.

## Success Criteria

- Source-format scalar/row helpers exist for BF16, F32, F8_E4M3_B128, and
  MXFP4 sufficient to build exact CPU/reference matvec tests or to fail closed
  with a precise unsupported-format error.
- Existing source BF16 handling is corrected where it is currently mislabeled
  as F16 in reference paths, especially embedding, HC control matvecs, router
  control, compressor/indexer BF16 tensors, and output head.
- A guarded diagnostic mode exists for source-layout correctness work. It must
  bypass the normal source guard only for a named test/oracle path, not for
  normal generation.
- Local synthetic tests cover dtype conversion/dequant dispatch and prove that
  unsupported source-layout execution still fails closed outside the diagnostic
  mode.
- If exact FP8/MXFP4 reference primitives are available, the source-layout
  single-slot path can run at least one official short vector far enough to
  compare generated tokens or top-logprob membership against
  `tests/test-vectors/official.vec`.
- If exact FP8/MXFP4 reference primitives are not available, the sprint produces
  a shipped blocker report, commits the completed source dtype primitives, and
  leaves full single-token decode explicitly unshipped.
- Cluster validation archives real logs under `docs/sprints/drafts/`, including
  guard proof, dtype primitive tests, and any official-vector comparison that
  was attempted.

## Verification Strategy

- Local model-less tests:
  - BF16 conversion and dispatch tests for representative row-gather/matvec
    behavior;
  - F8_E4M3_B128 and MXFP4 reference decode tests with small synthetic byte
    fixtures once semantics are known;
  - guard regression showing normal source-model generation still rejects.
- Local build:
  - build changed tools/tests with `make`;
  - run focused unit tests and `git diff --check`.
- Cluster checks:
  - build with `CUDA_ARCH=sm_70`;
  - run source-layout metadata and diagnostic tests against
    `/models/DSv4-Flash-256e-fixed.gguf`;
  - run any official-vector comparison on the V100 pod, not as a macOS huge CPU
    inference run.
- Evidence:
  - archive stdout/stderr logs under `docs/sprints/drafts/SPRINT-007-*.log`;
  - write `SPRINT-007-REPORT.md` with a `SHIP`, `EXTEND`, or `STOP` verdict.

## Uncertainty Assessment

- Correctness uncertainty: High. The sprint depends on exact F8_E4M3_B128 and
  MXFP4 source semantics, plus matching official serving behavior from
  top-logprob slices rather than full logits.
- Scope uncertainty: High. A full source-layout one-token path touches many
  tensor families. The plan must define a hard stop if format primitives or
  trusted reference comparison are not ready.
- Architecture uncertainty: Medium. The diagnostic path can extend existing CPU
  reference code, but it must not accidentally become a second production
  runtime or conflict with the V100 pack/context sidecar.

## Open Questions

1. What exact F8_E4M3_B128 block scale semantics are used by the source GGUF:
   scale as first/last byte, E8M0 interpretation, and E4M3 variant?
2. What exact MXFP4 codebook/scale semantics are used by the source GGUF and
   are they documented in the repo or only implied by upstream tooling?
3. Is token/top-logprob membership from official API vectors sufficient for
   this sprint, or do we need a separately generated full-logit oracle?
4. Should the first diagnostic decode path run through the existing CPU
   reference graph, or should it be a narrower per-layer/per-family oracle
   until all source tensor families are validated?
5. How much cluster time is acceptable for a one-token CPU/reference source
   run if it is too slow for routine local validation?
