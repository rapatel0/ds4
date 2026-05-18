# SPRINT-006 Merge Notes

## Inputs

- `docs/sprints/drafts/SPRINT-006-INTENT.md`
- `docs/sprints/drafts/SPRINT-006-CLAUDE-DRAFT.md`
- `docs/sprints/drafts/SPRINT-006-CODEX-DRAFT.md`
- `docs/sprints/drafts/SPRINT-006-GEMINI-DRAFT.md`
- `docs/sprints/drafts/SPRINT-006-CLAUDE-CRITIQUE.md`
- `docs/sprints/drafts/SPRINT-006-CODEX-CRITIQUE.md`
- `docs/sprints/drafts/SPRINT-006-GEMINI-CRITIQUE.md`
- `docs/architecture/DS4-V100-LAYOUT.md`

## Merge Decision

Use the Codex draft as the main structure and fold in Claude's stricter
fail-closed rails. Gemini's `SHIP / EXTEND / STOP` outcome framing is useful,
but its host-pinned relay fallback, broad `ds4.c` wiring, and full 1328-tensor
skeleton scope are not appropriate for Sprint 006.

## Accepted Changes

- Name the new module `ds4_v100_context`, not `ds4_gpu_context`, so the API is
  clearly V100-appliance specific and does not widen the existing arena API.
- Keep the context as a sidecar to legacy CUDA globals. Do not refactor
  `g_cublas`, `g_cuda_tmp`, or normal decode wiring in this sprint.
- Encode V100 execution policy as data and reports, including a forbidden-claim
  column for BF16, FP8, and FP4.
- State that production dense GEMMs on V100 target FP16 HMMA with FP32
  accumulation; broad FP32 GEMMs are not an acceptable default.
- Treat BF16 as source/probe/explicit-conversion only.
- Treat FP8, MXFP4, and FP4 as packed source/runtime inputs for later
  registered unpack/dequant or low-bit kernels.
- Require device-to-device relay for real multi-GPU validation. Host-pinned
  relay may not make a production topology smoke pass.
- Add topology, descriptor, memory-reserve, relay, and source-layout guard
  STOP conditions.
- Limit descriptor coverage to the contract needed for the no-math skeleton:
  global embedding, representative F32/control tensors, representative FP8 and
  MXFP4 families as descriptors, HC control tensors, and at least one
  representative full layer row set.
- Keep `ds4-server`, MTP, KV population, output-head math, and decode out of
  scope.

## Explicit Answers

- First diagnostic context entry point: keep it tool-private in
  `ds4_v100_context.h`; do not re-export through `ds4.h` unless implementation
  proves a narrow inspect-only reason.
- Layer skeleton coverage: validate all 43 layer ownership records and required
  family classes, but do not require full descriptor binding for every model
  tensor.
- Relay validation: require a full peer matrix and at least one real
  cross-stage device-to-device relay on the V100 pod for `SHIP`.
- `gpu7` reserve: reserve for output head and a small MTP placeholder, but do
  not allocate or execute either path.
- Cluster smoke mode: allow `probe-only` and `use-existing-arenas`; do not make
  full re-upload of all weights a Sprint 006 ship requirement.

## Rejected Items

- Host-pinned or managed-memory relay as a successful production path.
- Any wording that implies native BF16, FP8, or FP4 tensor-core compute on V100.
- Any broad FP32 GEMM fallback for main model math.
- Full source-layout decode, prefill, KV writes, routed MoE execution,
  output-head projection, MTP, server deployment, or throughput benchmarking.
- A full `ds4.c` scheduler integration or a model-wide 1328-tensor execution
  walk.
