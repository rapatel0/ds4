# Sprint 210 Intent - TP8 Real Layer Body Prototype

Date: 2026-05-23

## Seed Prompt

Continue the high-throughput practical-serving vision after Sprint 209. The TP
path must be completely separate from the PP/layer scheduler; do not add a
generic scheduler abstraction and do not retrofit TP into
`ds4_v100_scheduler.*`.

## Orientation Summary

- Sprint 208 proved `PP1/TP8` is not disqualified by topology or memory for
  the 32-slot / 128K-256K target when KV is sharded.
- Sprint 209 proved a standalone TP8 one-layer boundary with sharded KV
  allocation and two hidden reductions is correct and fast enough to continue.
- The current TP8 body is synthetic compute, so it does not prove DS4 layer
  execution, tensor-core utilization, routed/shared FFN behavior, or attention
  ownership.
- Existing Sprint 207 runtime/kernel edits are dirty in the worktree and must
  not be staged or cleaned up as part of this sprint.
- The next sprint must make the TP path more real while keeping it separate:
  new TP-only source files, new TP-only smoke/benchmark targets, no PP scheduler
  edits.

## Relevant Code Areas

- `tools/ds4-v100-tp8-layer-smoke.cu`: Sprint 209 TP8 boundary and sharded-KV
  allocation contract.
- `tools/ds4-v100-tp8-kv-shard-smoke.c`: KV shard descriptor/admission smoke.
- `tools/ds4-v100-tp8-layer-proxy.cu`: Sprint 208 boundary timing reference.
- `kernels/turbomind/ggml-turbomind/`: existing MXFP4 expert kernels and
  probes; useful as reference for future low-bit body integration.
- `Makefile`: add TP-only CUDA targets.
- `logs/from-cluster/`: cluster evidence location.

## Constraints

- No generic scheduler.
- No PP scheduler changes.
- No launcher default changes.
- No model weight copies in logs.
- Use 8 V100s for the TP8 gates.
- Keep context and slot targets aligned with practical serving:
  32 slots and 128K-256K context, with F8/Q8 KV accounting.

## Success Criteria

- A new TP-only executable runs a more realistic DS4 layer body than Sprint 209:
  tensor-parallel GEMM work resident on all eight V100s, followed by hidden
  reduction.
- The executable explicitly separates:
  - KV shard allocation/accounting;
  - attention/KV-touch placeholder timing;
  - column-parallel gate/up GEMM;
  - activation;
  - row-parallel down GEMM;
  - TP8 hidden reduction.
- Correctness passes against a deterministic reference or cross-device reduced
  output invariant at 32, 64, and 128 token shapes.
- Timing reports compute and reduction separately so we can compare topology
  cost to useful tensor-core work.
- The result chooses the next implementation direction:
  real low-bit TurboMind TP8 FFN body, TP8 attention/KV body, or pause TP8.

## Verification Strategy

- Local hygiene: `git diff --check`; macOS CUDA targets fail with an explicit
  CUDA-required message.
- V100 build: `make -j80 <new-target> CUDA_ARCH=sm_70`.
- V100 run: 32, 64, 128 tokens; 32 slots; 128K and/or 256K context; ratio-4 F8
  KV by default.
- Evidence logs under `logs/from-cluster/sprint210-tp8-real-layer/`.
- Update `docs/sprints/SPRINT-210.md`, `docs/sprints/STATUS.md`, and
  `docs/sprints/VISION.md`.

## Uncertainty

- Correctness: Medium. Synthetic weights are deterministic, but cublas/Tensor
  Core reductions need tolerant checks.
- Scope: Medium. A full low-bit TurboMind TP8 routed body may be too large, so
  the sprint should first land a real resident tensor-core body and record the
  low-bit gap explicitly.
- Architecture: Low. Separate TP-only files are the agreed architecture.

## Open Questions

- Does an FP16 tensor-core TP8 FFN body plus reduction have enough useful work
  at 32 active tokens to make TP8 plausible?
- Does the 64/128 token scaling show that TP8 enters the denser-kernel regime?
- Should Sprint 211 prioritize low-bit TurboMind TP8 experts or sharded
  attention/KV after this gate?

## Vision Context

The north star is a practical high-throughput DS4 V100 appliance. Sprint 210 is
not serving integration. It is the next TP-only proof step: replace the
synthetic Sprint 209 body with resident tensor-core layer work, keep KV sharded,
and decide whether TP8 deserves deeper implementation.
