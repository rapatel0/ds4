# Sprint 173 Merge Notes

Date: 2026-05-22

## Inputs

- `docs/sprints/drafts/SPRINT-173-INTENT.md`
- `docs/sprints/drafts/SPRINT-173-CLAUDE-DRAFT.md`
- `docs/sprints/drafts/SPRINT-173-CODEX-DRAFT.md`
- `docs/sprints/drafts/SPRINT-173-GEMINI-DRAFT.md`

The formal cross-critique phase was collapsed into this merge pass. The
direction is already set by the user and the three drafts converged on the same
core plan: build a reusable routed-FFN executor boundary, remove expanded
activation staging first, keep default behavior unchanged, and preserve a
partial-output shape for TP/EP.

## Claude Draft Strengths

- Best grounded implementation split:
  - Slice A: un-expanded activation path using one per-token cast plus indexed A.
  - Slice B: stretch in-kernel F32-to-F16 activation tile load through a new
    TurboMind ABI.
- Correctly identified that current served 6-route decode writes one token's
  hidden vector six times into `a_half`, making it the best first global
  intermediate to remove.
- Included `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6`, launcher allowlist
  updates, and the Sprint 170 failure mode where A/B could not start until the
  allowlist accepted the mode.
- Added a synthetic `FULL` versus `PARTIAL_ACCUMULATE` split smoke, which is the
  right low-cost way to make the TP/EP compatibility claim testable without
  wiring real topology yet.

## Codex Draft Strengths

- Strongest sprint structure and Definition of Done.
- Clear separation between the executor contract, fallback executor, candidate
  executor, liveness accounting, and promotion gate.
- Good risk framing: the sprint fails if it only renames the path without
  eliminating a real global intermediate.
- Useful file summary across `ds4_cuda.cu`, TurboMind ABI files, replay tools,
  launcher, and scheduler/TP smokes.

## Gemini Draft Strengths

- Most direct wording around the precision policy:
  compact FP4/MXFP4 storage should stay compact in memory, while FP16/FP32
  compute happens inside GPU execution near tensor-core work.
- Correctly emphasized that the descriptor is the foundation for future TP/EP
  and that this sprint should not create another one-off sidecar.

## Accepted Critiques

- The first execution slice must be bounded. A fully monolithic gate/up +
  activation + down + reduce kernel is a stretch goal, not the minimum sprint.
- `a_half` removal is necessary but may not be sufficient for a throughput win.
  The sprint therefore separates the architecture gate from the promotion gate.
- Partial-output support must be at least API-visible and smoke-tested, or the
  "helps TP/EP later" claim is too weak.
- Instrumentation must report intermediate liveness explicitly, not just timing,
  because many prior sprints changed wrappers without changing the real memory
  movement.

## Rejected Or Deferred Ideas

- A broad TP/EP scheduler rewrite is deferred. Sprint 164/165 already showed
  that bolting TP onto one layer can regress; Sprint 173 should build the local
  executor boundary that a persistent TP/EP design can reuse.
- Shared FFN and attention fusion are deferred. They can get read-only liveness
  notes, but not implementation work.
- A new public server or CLI surface is deferred. `fused6` remains an internal
  validation mode behind the existing appliance launcher.
- Real MTP integration is out of scope.

## Final Planning Decisions

- Sprint 173 objective: implement the reusable fused routed-FFN executor
  boundary and the first real candidate path for the served 6-route shape.
- Minimum implementation: contract + `fused6` mode + un-expanded `a_half`
  candidate + liveness logs + replay/smoke correctness + served A/B.
- Stretch implementation: TurboMind ABI/kernel that casts F32 activations to
  F16 inside the gate/up A-tile load and removes even the compact per-token
  activation staging.
- Promotion threshold: `>= 10%` continuation/decode tok/s improvement with
  correctness intact.
- Pivot trigger: if no real global expanded intermediate is removed, stop and
  pivot to persistent TP/EP boundary planning.
