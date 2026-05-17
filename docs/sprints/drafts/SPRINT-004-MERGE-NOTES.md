# SPRINT-004 Merge Notes

**Date:** 2026-05-17
**Intent:** `docs/sprints/drafts/SPRINT-004-INTENT.md`
**Final plan:** `docs/sprints/SPRINT-004.md`

## Inputs

- `SPRINT-004-CLAUDE-DRAFT.md`
- `SPRINT-004-CODEX-DRAFT.md`
- `SPRINT-004-GEMINI-DRAFT.md`
- `SPRINT-004-CLAUDE-CRITIQUE.md`
- `SPRINT-004-CODEX-CRITIQUE.md`
- `SPRINT-004-GEMINI-CRITIQUE.md`
- `docs/architecture/DS4-V100-LAYOUT.md`
- `/Users/ravi/repos/deepseek/docs/sprints/SPRINT-026-CLUSTER-TESTING.md`

## Draft Comparison

Claude's draft was selected as the merge base. It had the clearest
SHIP/EXTEND/STOP contract, the most useful phase gates, concrete `ds4_pack`
and `ds4_gpu_arena` module boundaries, dual upload providers, and validation
artifacts that can be reused by the first source-format math sprint.

Codex's draft contributed concise use cases, effort weighting, public API
caution, and an explicit warning that host-mapped or managed-memory fallbacks
can create a fake residency win.

Gemini's draft contributed a compact stakeholder summary and the useful
distinction between residency state and execution state. Gemini's critique
also identified two missing checks: upload I/O behavior and peer-to-peer
visibility reporting for the next relay sprint.

## Accepted Critiques

- Add explicit handling for fewer than 8 visible GPUs and
  `CUDA_VISIBLE_DEVICES` remapping.
- Bind `pack-index.tsv`, source GGUF, and `gpuN.weights` files to the same
  run through shard size/hash artifacts and stale-directory checks.
- Separate logical arena bytes from observed CUDA memory deltas. Logical bytes
  must match the pack plan; observed `cudaMemGetInfo` values are recorded and
  used for headroom/reserve decisions, not as a brittle byte-for-byte planner
  match.
- Add partial shard-emission and partial upload failure handling.
- Require header validation for the TSV schema, not blind positional parsing.
- Require `cudaPointerGetAttributes` or equivalent proof that successful
  residency uses device memory, not managed or host-mapped memory.
- Add CPU/stub synthetic smoke so orchestration is tested before the cluster.
- Add a P2P visibility report to the smoke output as signal for the next
  relay sprint, without requiring P2P enablement in Sprint 004.

## Rejected Or Deferred Critiques

- A full `ds4_gpu_context[8]` execution refactor is deferred. Sprint 004 only
  needs upload-only residency arenas.
- A first BF16/F8 source-format math probe is deferred. Sprint 004 stops at
  source-faithful packed bytes resident on device.
- Full per-tensor SHA-256 is deferred. Whole-shard SHA-256 plus per-tensor
  first/last 4 KiB spot checks are sufficient for the structural residency
  contract.
- JSON reports are deferred. TSV/log artifacts are enough for this sprint.
- Optimized parallel shard upload is deferred. Sequential or bounded-chunk
  upload is acceptable if the I/O timing is recorded.

## Interview Defaults Applied

The user had already set the direction in prior discussion: preserve
intelligence, keep source-faithful quantized layouts, avoid default SSD/host
offload, stop on material uncertainty, and defer MTP until feasibility is
understood. The planning workflow therefore proceeded with these defaults
instead of blocking on another interview round:

1. Full real-model shard emission is required for SHIP when persistent scratch
   is available. If scratch is blocked, the sprint may EXTEND after local
   parser and synthetic smoke work land.
2. Runtime supports both source GGUF offsets and emitted `gpuN.weights` shard
   providers. GGUF is the fast iteration provider; shard validates the
   emitted pack.
3. Sprint 004 stops at raw packed device residency. No math path, decode,
   MTP, or speculative decoding.
4. CUDA changes are limited to a residency-only arena sidecar. No broad
   multi-device execution context refactor.
5. Validation artifacts are shard sizes, shard SHA-256, reconcile log,
   residency logs, per-tensor spot checks, and one deterministic provider
   cross-check.

## Final Scope

Sprint 004 will add runtime pack-index loading, source reconciliation,
upload-only per-GPU device arenas, a standalone residency smoke tool, local
synthetic validation, and cluster validation on the 8x V100 node. It will not
enable source-model generation.
