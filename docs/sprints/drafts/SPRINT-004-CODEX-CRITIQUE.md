# SPRINT-004 Critique: Claude vs Gemini

This critique is grounded in the Sprint 004 intent and the current repo
shape, not just the prose quality of the drafts.

A few repo-level facts matter up front:

- `ds4.h` already has `inspect_only` in `ds4_engine_options`, and the CLI
  already exposes `--inspect`. A good Sprint 004 plan should extend that
  path instead of inventing a parallel inspection mode.
- `ds4_gpu.h` is still intentionally narrow, while `ds4_cuda.cu` still has
  process-global CUDA/model state (`g_model_host_base`,
  `g_model_device_base`, `g_model_ranges`, `g_cublas`). That makes additive
  residency plumbing much safer than a broad multi-device execution refactor
  in this sprint.
- `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv` already has a concrete
  schema: `source_shape`, `source_offset`, `shard_offset`, `scale_offset`,
  and `checksum` are real columns. Any Sprint 004 parser plan should target
  that exact contract and say how drift or stale artifacts are handled.

## Executive Verdict

The Claude draft is the better base by a clear margin. It matches the sprint
intent, fits the current codebase shape, and is much closer to an
execution-ready plan.

The Gemini draft is directionally right but too thin and too refactor-heavy
for this sprint. It reads more like a planning summary than a sprint spec.

If one draft is chosen as the merge base, it should be the Claude draft, but
it still needs a few corrections and additional edge-case coverage.

## Claude Draft

### Strengths

- It is tightly aligned with the stated sprint boundary: runtime pack loading
  and device residency only, with the source-model decode guard left intact.
- It fits the current repo shape well. A new `ds4_pack` module plus a
  residency-only arena sidecar is a much better match for today's
  single-runtime globals than a full execution refactor.
- The phased plan is credible. The sequence from parser/reconciliation to
  local synthetic smoke to cluster shard emission to full residency smoke is
  the right order.
- It treats `pack-index.tsv` as a structural contract rather than a loose
  hint. That is the correct posture for this sprint.
- Supporting both `gguf` and `shard` providers is a practical choice. It
  preserves fast local iteration while still requiring cluster validation of
  emitted shards.
- Its validation artifacts are materially useful. Reconcile logs, shard size
  facts, shard hashes, per-GPU residency logs, and one cross-provider compare
  are all strong evidence for the next sprint.
- The DoD is substantially better than Gemini's because it demands real
  artifacts, explicit planner comparison, and preservation of the source-model
  generation guard.

### Weaknesses

- It is over-specified in a few places. Exact helper names, exact log names,
  exact report filenames, and even the specific example tensor for the
  cross-provider compare make the plan heavier than it needs to be.
- The draft sometimes says `--inspect-only`, but the current CLI already uses
  `--inspect`. The plan should stay consistent with the repo's existing
  terminology.
- The 64 MiB planner-match tolerance is probably too rigid unless the draft
  explicitly distinguishes logical arena bytes from observed `cudaMemGetInfo`
  deltas. CUDA context overhead and allocator behavior can make a strict
  device-free-memory delta noisy even when the arena logic is correct.
- Some close-out items are lower value than the core residency proof:
  dedicated followup docs and seed docs are useful, but they should not carry
  the same weight as the structural validation outcome.

### Gaps In Risk Analysis

- It does not explicitly call out `CUDA_VISIBLE_DEVICES` remapping or a
  reduced visible-device set as an operational risk. A pack plan with
  `owning_gpu in [0,7]` can still fail if the process sees fewer devices or a
  remapped numbering scheme.
- It does not name stale-artifact drift as a first-order risk. A real failure
  mode is `pack-index.tsv`, source GGUF, and `gpuN.weights` coming from
  different runs or different model files.
- It under-specifies the difference between "arena used bytes match the plan"
  and "device free memory moved by the expected amount." Both are useful, but
  they are not the same measurement.
- It does not explicitly treat interrupted shard emission or a partially
  populated shard directory as a risk worth kill-gating.

### Missing Edge Cases

- Fewer than 8 visible GPUs, or visible GPUs that are renumbered by
  `CUDA_VISIBLE_DEVICES`.
- Missing one shard file, stale shard files from an earlier emission, or a
  partially emitted shard directory after interruption.
- A row where `shard_offset + byte_length` exceeds the shard file size, even
  though the pack-index parsed successfully.
- Duplicate `semantic_tensor_id` rows or duplicate `source_name` rows in the
  index.
- Unexpected but syntactically valid control rows such as `layer_id = -1`
  for embeddings/output tensors.
- GPU 0 having materially different free-memory headroom than the other GPUs
  due to context setup or pod-level overhead.

### Definition Of Done Completeness

- The DoD is strong overall and is close to execution-ready.
- It should add one explicit requirement that the residency smoke fails
  cleanly when the visible device count does not satisfy the pack plan.
- It should add one explicit requirement that the `shard` provider validates
  that the shard files used at smoke time are the same artifacts measured in
  Phase 4, or at least rechecks file size/hash from the same directory.
- It should add one explicit requirement that planner validation reports both
  logical arena bytes and observed device-memory facts, rather than collapsing
  both into a single 64 MiB threshold.
- It should keep terminology aligned with the existing `--inspect` path.

## Gemini Draft

### Strengths

- It identifies the correct top-level goal: consume `pack-index.tsv`, emit
  persistent shards, allocate per-GPU arenas, and prove residency without
  enabling decode.
- It keeps the sprint focused on residency plumbing rather than drifting into
  full numerical correctness work.
- Its shorter format makes the main direction easy to understand quickly.
- It does at least mention bounds-checking and VRAM-budget concerns, which are
  essential to this sprint.

### Weaknesses

- It is too thin to serve as the sprint document. There is no real SHIP /
  EXTEND / STOP flow, no kill-gated phase structure, and not enough execution
  detail to guide the work.
- It asks for a broader multi-device refactor than the sprint needs.
  Introducing `ds4_gpu_context[8]` in Sprint 004 is a much larger change than
  a residency-only sidecar, especially given the current global CUDA state in
  `ds4_cuda.cu`.
- Its runtime-pack story is too shard-centric. By refusing to rely on GGUF
  offsets at all, it gives up the fastest local debug path and makes the
  sprint more cluster-dependent than necessary.
- "Reading bytes directly from shard files into the corresponding device
  memory, bypassing host-side staging where possible" is not a grounded plan
  for this repo. The likely implementation here is still `pread` into host
  buffers plus `cudaMemcpy`.
- It is more invasive than necessary in public structures. Extending
  `struct ds4_model` and broadening `ds4.h`/`ds4_gpu.h` is not obviously
  required for a residency smoke.
- The proposed validation surface is weaker. A smoke test hidden in
  `ds4_cli` or under `tests/` is less clear than a dedicated diagnostic tool.

### Gaps In Risk Analysis

- It misses the biggest repo-specific risk: colliding with today's global
  CUDA/model state while trying to add multi-device residency.
- It does not discuss stale or mismatched artifacts:
  `pack-index.tsv`, source GGUF, and shard files can disagree across runs.
- It does not discuss reduced or remapped visible-device sets.
- It does not discuss the difference between logical planned payload bytes and
  noisy runtime memory-accounting signals from CUDA.
- It does not discuss interrupted shard emission, partial retry states, or how
  a cluster rerun proves that it is using fresh artifacts.
- It does not treat lack of a local synthetic smoke path as a risk, which
  means the cluster could become the first place the orchestration is tested.

### Missing Edge Cases

- Exact parsing of the real `SPRINT-003-PACK-INDEX.tsv` schema, including the
  full header and fields such as `source_shape`, `scale_offset`, and
  `checksum`.
- Duplicate semantic IDs, duplicate source names, malformed numeric fields, or
  comments/trailing newlines in the TSV.
- Control tensors and non-layer tensors that legitimately carry `layer_id=-1`.
- A pack-index that expects 8 GPUs while the runtime sees fewer.
- Missing or partially emitted `gpuN.weights` files.
- CPU/non-CUDA builds that still need to exercise parser and orchestration
  logic locally.
- Runtime behavior when the source guard is preserved and the tool is invoked
  through the existing `--inspect` flow.

### Definition Of Done Completeness

- The DoD is incomplete for a kill-gated sprint.
- It lacks an explicit reconciliation artifact with one row per tensor and
  mechanical mismatch tags.
- It lacks a local synthetic smoke requirement, which is important because the
  cluster should not be the first end-to-end test of the orchestration path.
- It lacks an explicit no-regression gate for the current single-device path
  and for preservation of the source-model generation guard.
- It lacks a precise planner-comparison contract and does not separate shard
  size facts, checksum facts, and runtime VRAM facts into durable artifacts.
- It lacks an explicit failure mode for reduced visible-device sets, stale
  shard directories, or mismatched pack-index/model inputs.

## Recommendation

Use the Claude draft as the base.

Before merging it into the final Sprint 004 plan, tighten it in four ways:

- Add explicit handling for reduced or remapped visible-device sets.
- Add a stronger stale-artifact policy that binds `pack-index.tsv`, source
  GGUF, and emitted shard files to the same run.
- Split logical arena-byte validation from observed CUDA memory-headroom
  validation.
- Add explicit failure handling for interrupted or partial shard emission.

The Gemini draft is still useful as a short summary, but it is not strong
enough to serve as the sprint plan without being expanded toward the shape the
Claude draft already has.
