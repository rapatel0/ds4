# SPRINT-004 Critique — CODEX and GEMINI Drafts

**Reviewer:** CLAUDE
**Date:** 2026-05-17
**Inputs:** `SPRINT-004-INTENT.md`, `SPRINT-004-CODEX-DRAFT.md`, `SPRINT-004-GEMINI-DRAFT.md`

This critique evaluates both drafts against the Sprint 004 intent and against
the architectural ground rules in `docs/architecture/DS4-V100-LAYOUT.md` and
`AGENT.md`. It is structured per draft for direct comparison.

---

## 1. CODEX Draft

### 1.1 Strengths

- **Strong scoping discipline.** The overview commits up-front to "no decode,
  no math, no broad refactor" and the architecture section explicitly avoids
  refactoring `ds4_cuda.cu`'s globals. This aligns with the intent's caution
  against broad multi-device refactor.
- **Explicit STOP artifact pattern.** Phase 4 names
  `SPRINT-004-STOP.md` as a first-class deliverable if persistent scratch is
  unavailable, and a corresponding risk row covers it. This is the right
  failure mode given the intent's "explicit STOP backed by the concrete storage
  blocker" line.
- **Residency-class reporting is called out by name.** The risk table flags
  "host-mapped or managed fallbacks make the smoke look successful" with a
  mitigation that requires reporting the residency class — a subtle failure
  mode that is easy to miss.
- **Public API growth is treated as a first-class concern.** Open Question 1
  surfaces the trade-off explicitly: ship the bounded entry in `ds4.h` vs.
  accept an internal refactor to avoid public surface growth. This respects the
  `AGENT.md` "keep public APIs narrow" rule.
- **Security section is well-formed.** Path traversal, bounds checks, and
  `O_RDONLY` opens are called out; the source-model decode guard is reaffirmed
  as a security boundary, not just a correctness one.
- **Phase effort percentages.** ~30/20/35/15 gives the reader a credible
  sense of where the cost lives (P3 is the hard one).

### 1.2 Weaknesses

- **No outcome contract.** There is no SHIP/EXTEND/STOP definition tying the
  artifacts together. The DoD is a flat checklist; without an overarching
  verdict structure, an "EXTEND" outcome (P1–P3 local but cluster blocked) has
  no defined shape. Compare to the intent's success criteria #3, which already
  contemplates an explicit STOP.
- **Single upload provider.** Phase 3 says "upload each `gpuN.weights` shard
  into its owning device arena" — i.e., shard-only. There is no fallback or
  cross-check for uploading directly from the source GGUF via `mmap +
  cudaMemcpy`, which is faster to iterate against locally before shards exist
  on persistent scratch. The Open Question 3 about "explicit read buffers vs.
  mmap" stays open, but the plan does not commit a path or a cross-check
  between the two.
- **Reconciliation logic lives in `ds4.c` rather than a new module.** Phase 1
  adds "internal runtime-pack row validation and plan construction" inside
  `ds4.c`. Given that `ds4.c` is already the GGUF loader, source-layout
  binder, source-kernel guard, *and* engine entry point, adding pack
  parsing inside it grows an already-heavy translation unit. A dedicated
  `ds4_pack.h/c` would isolate the parser and let the parser/reconciler be
  unit-tested without dragging in engine-open state.
- **No tolerance spec for VRAM match.** DoD says "without reported overfill of
  any 32 GB V100" but does not give a numeric tolerance against the Sprint 003
  planner. The planner predicts 20.98 GiB on gpu0; if the residency smoke
  reports 21.3 GiB on gpu0, is that pass or fail? The plan needs a measured
  reserve and a delta tolerance.
- **No CPU/stub mode for the smoke tool.** Phase 3's smoke is described only as
  the CUDA cluster workload. The DoD line "builds and can run against
  synthetic or local pack fixtures without decode" implies a non-CUDA mode but
  the implementation does not describe how (CPU stub of `ds4_gpu_arena_*`,
  in-memory upload, etc.). Without a stub the orchestration code is first
  exercised on the cluster, which is the wrong place to discover an off-by-one.
- **No phase 0 hygiene.** Build-hygiene checks (does `make cpu` and the
  existing `make tools/ds4-v100-pack` still build clean on a fresh tree, is
  the cluster pod recipe still good) are folded into Phase 4. A short
  orientation phase keeps the kill-gate clean: if the pod recipe is stale, you
  find out before writing parser code.
- **Validation artifact naming is light.** The Files Summary names
  `SPRINT-004-RESIDENCY-SMOKE.log` and `SPRINT-004-SHARD-FILES.tsv` but the
  reconciliation log, per-shard SHA-256 list, and spot-check results are
  bundled together implicitly. The intent's success criterion #6 asks for
  three distinct facts (shard sizes, checksums, per-GPU VRAM); splitting them
  into separate files makes the artifacts easier to consume in Sprint 005.
- **"Synthetic or local pack fixtures" is undefined.** The DoD requires the
  smoke to run against synthetic fixtures, but no fixture is specified or
  scripted (no `tests/residency_smoke_synthetic.sh` equivalent).

### 1.3 Gaps in Risk Analysis

- **No risk for test-fixture rot.** If `ds4-v100-pack --write-index` later
  adds, drops, or reorders a column (e.g., `kernel_family` moves), a parser
  that uses positional reads will silently misinterpret data. The plan does
  not specify header-by-name validation, so the corresponding risk is missing
  too.
- **No risk for CUDA driver/version drift on the pod.** Sprint 003 ran on a
  specific image; the residency smoke will need a CUDA 12.x driver and
  `sm_70` toolchain. If the cluster image rotates, the build may pass on the
  pod but the runtime may surface a `cudaErrorNoBinaryForGpu` only at first
  `cudaMalloc` per device.
- **No risk for partial-upload device state.** If `cudaMemcpy` fails midway
  through a 20 GiB arena fill, the arena pointer is still valid but contents
  are inconsistent. The smoke must either reset the device on partial failure
  or refuse re-use of an arena that has not completed all expected uploads.
- **No risk for scope creep into source-format math.** Sprint 003's
  follow-ups list FP8/MXFP4 upload-and-dispatch as Critical. The current draft
  could easily drift into "while we're here, let's try one BF16 dispatch" and
  pull in `ds4_gpu_context[gpu]`. A risk row + explicit Open Question
  resolution ("no math this sprint") is needed.
- **No risk that the new arena API becomes the de facto multi-GPU API.** The
  plan correctly keeps the arena API additive, but does not flag the cultural
  risk that future kernel work will reach for the arena handle instead of
  paying for a proper `ds4_gpu_context[gpu]` design.
- **"Storage unavailable" mitigation is good but "host-mapped fallbacks"
  mitigation is weak.** The mitigation says "report residency class
  explicitly" — but the plan does not define what residency class means in
  code (e.g., must the implementation call `cudaPointerGetAttributes` and
  assert `type == cudaMemoryTypeDevice`?). Without that, "report residency
  class" is documentation, not enforcement.

### 1.4 Missing Edge Cases

- **Device enumeration mismatches.** The cluster may surface 7 GPUs (one
  missing) or 9 (MIG slice surprise). The smoke should hard-fail unless
  exactly 8 V100s with ≥32 GiB each are visible, or it should accept a
  command-line override for synthetic/sub-cluster runs.
- **Pack directory pollution.** What if `gpuN.weights` files for N ∈ {0..7}
  are present plus a stale `gpu8.weights` from a prior 9-GPU plan? The
  Security section mentions "reject unexpected shard file names" but the
  parser does not own the directory listing; the plan should specify whether
  the pack directory is walked or only the index is consulted.
- **TSV encoding/whitespace.** UTF-8 BOM, CRLF line endings, embedded tabs
  inside fields, or trailing whitespace from cluster `printf` are likely on
  a pod-built artifact. The plan should commit to LF-only, no BOM, exact
  column count per row.
- **`cudaMemcpy` length limits / alignment.** For tensors near or above 4 GiB
  (gpu0 holds multiple multi-GiB tensors), `cudaMemcpy` itself is fine but
  some intermediate host buffers may not be. Plan does not call out
  alignment or chunking strategy.
- **Prior-test residue in device memory.** If the pod is reused and a prior
  process left memory allocated, `cudaMemGetInfo` will report less free
  than 32 GiB. Plan does not specify `cudaDeviceReset` policy at smoke
  startup.
- **Pack-index entries with no shard.** The `shard_file` column may be empty
  if shards were not emitted (gguf-only provider). The reader must handle
  NULL `shard_file` gracefully; the plan does not call this out.

### 1.5 Definition of Done — Completeness

- DoD line "without reported overfill" is qualitative. Quantify: "every GPU's
  used-arena bytes ≤ 32 GiB − declared reserve, where reserve is documented in
  the report." Add a delta-vs-planner tolerance.
- DoD does not gate on the source-tensor count. The model has ~1328 tensors
  per Sprint 003 plan; an off-by-one in the parser or reconciler can pass
  unit tests but fail on the real model. A DoD line like "reconciliation log
  has N rows where N matches `ds4-v100-pack --write-index` row count" closes
  that loop.
- DoD does not require an end-to-end spot-check pass. The plan mentions
  "byte-spot checks" in Phase 4 Tasks but does not promote them to a DoD
  line item.
- DoD does not name the synthetic test target or specify that it must pass
  before any cluster work. The intent's verification strategy explicitly
  lists "unit or smoke tests for pack-index parsing using a small synthetic
  index" — the DoD should require it.
- DoD does not require `git diff --check`; the intent's verification strategy
  lists it.
- The `make cpu` build gate is missing from the DoD even though it appears in
  verification strategy.

---

## 2. GEMINI Draft

### 2.1 Strengths

- **Compact and readable.** The architecture section partitions cleanly into
  "Pack-Index Contract," "Device Residency Model," and "Multi-Device Refactor"
  with each subsection getting roughly equal weight.
- **Explicit packed-format rule.** "No Dequantization: Weights are loaded in
  their packed source formats (`BF16`, `MXFP4`, `F8_E4M3_B128`) as dictated by
  the manifest." This is the right one-line invariant and is more visible
  than in the Codex draft.
- **Frames residency state vs. execution state distinction.** "The focus is
  on *residency state* (pointers to buffers on specific PIDs/GPUs) rather
  than *execution state* (streams/kernels)" is a clean conceptual frame and a
  useful boundary for the multi-device discussion.
- **Open Question 1 (mixed mode) is a real product question.** Allowing some
  tensors to load from GGUF and others from shards is the natural debug mode
  during bring-up; surfacing it for discussion is a strength.
- **Open Question 3 (one micro-kernel).** Surfacing the BF16 norm/RMS probe
  question is forward-looking; even if the resolution is "no," capturing the
  question makes the next sprint easier to scope.

### 2.2 Weaknesses

- **Multi-device refactor is over-scoped.** Phase 3 says "Refactor
  `ds4_cuda.cu` to support a `ds4_gpu_context[8]` structure." The intent's
  architecture uncertainty section, and both the Codex and Claude drafts,
  explicitly warn against this. `ds4_gpu_context[gpu]` is a multi-sprint
  refactor; lumping it into Sprint 004 sets up scope blow-out, and it is
  not necessary for residency-only upload. A minimal sidecar arena API
  would meet the same need.
- **No outcome contract or kill-gate per phase.** Phase 2 has a single "Kill
  Gate" line ("Stop if persistent storage is unavailable or insufficient
  (requires ~145GB)"), but no equivalent for the parser phase, the CUDA
  phase, or the residency smoke. Without kill-gates, a partial sprint
  delivery has no defined deliverable shape.
- **Validation step in Phase 4 conflates structural and numerical
  correctness.** "Perform spot-checks: copy small ranges back to host and
  compare against source GGUF/manifest" is good. But Phase 4's success line
  reads "checksums match" without saying which checksums (per-shard, per-tensor,
  per-GPU-arena via `cudaMemcpy DtoH` + `sha256sum`). Whole-VRAM hashing
  across 145 GiB is operationally expensive and is not in scope for a smoke;
  the plan needs to be specific about what is being checksummed and when.
- **"Mitigation: fall back to multi-step emission" contradicts the intent.**
  The intent states emission must use persistent scratch, not temporary
  workspaces, and Sprint 003 deferred emission specifically because the pod
  filesystem is disposable. "Fall back to multi-step emission" sounds like
  splitting the 145 GiB write across temporary stages, which violates the
  ground rule. The Codex draft's "treat persistent scratch as a hard
  dependency" mitigation is the correct posture.
- **Single-provider upload path.** Phase 3's upload path is `pread` from
  `gpuN.weights` only. There is no `mmap`-based path from the source GGUF
  for fast iteration before shards exist on persistent scratch. This is
  particularly limiting for local development, because it forces shard
  emission to be a prerequisite for any smoke run.
- **Phase 1 binds shard metadata into `struct ds4_model`.** "Extend `struct
  ds4_model` or equivalent to store shard-resolution metadata" — but
  `struct ds4_model` is not currently part of the source-model decode path,
  and pack-index metadata is not a property of the GGUF model itself. A
  separate `ds4_pack` opaque handle threaded into the engine through a new
  option keeps the layering correct.
- **Public API growth is not bounded.** The plan says `ds4.h` is "Modify:
  Update model structures to hold shard metadata" but does not say which
  fields become public. Without a constraint, this can leak shard internals
  through `ds4.h`.
- **`ds4-v100-pack.c` is listed as Inspect/Use but is not modified or
  hardened.** Phase 4 plans the first emission to persistent scratch — the
  first time it runs at full real-model scale on this cluster — but does not
  budget any time for fixing a packer bug discovered during emission. Sprint
  003's dry-run is not a substitute for full emission.
- **No test infrastructure plan.** The implementation lists "Unit test with
  a synthetic small index and dummy tensor names" as a one-line tail of
  Phase 1, and "Verify that `cudaSetDevice` is correctly managed" in Phase 3
  is not a test, it is a code-review comment. There is no synthetic
  end-to-end smoke for the orchestration code separate from the cluster
  run.
- **No outcome verdict structure.** Unlike "SHIP/EXTEND/STOP" or even
  "complete/blocked," the draft simply lists items; if half the items pass
  and half fail, the operator has no shared language for the result.
- **Files Summary missing.** Several artifacts referenced in implementation
  (checksums, VRAM logs) do not have a row in the Files Summary table. Only
  `SPRINT-004-REPORT.md` and the new test file appear; the per-shard
  checksum file, residency log, reconciliation log, etc. are not listed.

### 2.3 Gaps in Risk Analysis

The risk table has only **4 rows**. Compared to the intent's "Uncertainty
Assessment" (Medium correctness, Medium scope, High architecture), this is
under-specified.

- **No risk row for scope creep.** The plan itself proposes a
  `ds4_gpu_context[8]` refactor (which the intent warns against); a risk row
  capturing "refactor expands past residency-only" would force a mitigation.
- **No risk for spec drift between manifest and source.** Reconciliation can
  find a manifest bug rather than a reader bug; the plan does not commit a
  STOP posture for that case.
- **No risk for test-fixture rot or TSV header drift.** If the column order
  of `pack-index.tsv` changes, a positional parser breaks silently.
- **No risk for partial-upload device state.** Same as Codex critique above.
- **No risk for host-mapped/managed memory accidentally satisfying the
  upload.** The plan does not require the implementation to assert
  `cudaMemoryTypeDevice`, so the smoke could "succeed" with mapped host
  memory.
- **No risk for pod recycling between phases.** Phases 2 and 4 require a pod
  and a cluster; if the pod is GC'd between them, the rebuild adds an
  unbudgeted hour.
- **VRAM-overfill mitigation is vague.** "Use the `ds4-v100-plan` reserve
  (3-4GB) and monitor `cudaMemGetInfo`" — fine, but the plan does not state
  what the residency smoke does when the measured used bytes are within
  reserve of the ceiling (warn? abort? continue?).
- **Persistent-storage mitigation contradicts intent.** Re-stated from
  Weaknesses: "fall back to multi-step emission" is not a real mitigation
  under the intent's ground rules.

### 2.4 Missing Edge Cases

All of the cases from the Codex critique (device enumeration, pack-directory
pollution, TSV encoding, alignment/chunking, prior-test residue, missing
shard_file) also apply here. Additional cases:

- **Mixed-mode partial loads on dev machine.** Open Question 4 ("how to
  handle partial loads for local testing") is acknowledged but not designed
  for. The proposed `ds4_gpu_context[8]` refactor is overkill for a dev-mode
  single-GPU residency check.
- **Multi-step emission failure mode.** If the mitigation in the risk table
  is actually invoked, the resulting shards have no guarantee of consistency
  across steps. There is no plan for verifying cross-step shard integrity.
- **`SHA-256` of 145 GiB at runtime.** Open Question 2 acknowledges this
  cost but the plan still requires "generate SHA-256 checksums for the
  resulting `gpu[0-7].weights` shards" in Phase 2 without a fallback.
- **`cudaMemcpy` failure recovery.** Same as Codex.
- **Tensor count assertion vs. model graph.** No DoD line ties parsed pack
  rows to the source-model tensor count.
- **CUDA error propagation through C calls into a CUDA TU.** `ds4_cuda.cu`
  needs a clear convention for returning `cudaError_t` through a C ABI; this
  isn't called out.

### 2.5 Definition of Done — Completeness

The DoD is the weakest section of the draft.

- "Successfully parses `pack-index.tsv` and binds all model tensors to shard
  locations" — does not require rejection of malformed inputs, duplicate
  rows, or dtype/shape mismatches. The parser could be permissive and still
  satisfy this line.
- "Full 145GB real-model shards emitted and verified on cluster persistent
  scratch" — "verified" is undefined. Verified by `stat`? `sha256sum`?
  Spot-check vs. source GGUF?
- "CUDA backend can allocate and fill 8 separate GPU arenas" — does not say
  the fill must be source-faithful bytes (no dequantization), and does not
  require per-GPU memory reporting.
- "Residency smoke test passes on the 8x V100 node without OOM or data
  corruption" — "data corruption" is unobservable without spot-checks; the
  DoD does not require spot-checks.
- "Per-GPU VRAM usage is recorded and fits within the 32GB-per-GPU budget" —
  no tolerance against the planner; no required artifact path.
- "Generation guard remains active (unless a math harness is explicitly
  added and passes)" — the "unless" clause should be removed for this
  sprint. The intent explicitly defers any decode/math harness, and the
  Open Question 3 suggestion of a micro-kernel should be resolved as **no**
  in the final plan rather than left as an escape hatch in the DoD.
- "Validation artifacts (file sizes, checksums) are recorded in the sprint
  report" — should name files (e.g., `SPRINT-004-SHARD-SHA256.tsv`,
  `SPRINT-004-RESIDENCY.log`) rather than rely on prose in a single report.
- Missing DoD items:
  - No build-hygiene gate (`make cpu`, `make tools/ds4-v100-pack`,
    `git diff --check`).
  - No requirement for synthetic local smoke to pass before cluster run.
  - No reconciliation log artifact requirement.
  - No constraint against persistent F16/F32 dequantized copies.
  - No constraint that the `ds4_server.c` HTTP API is not modified.

---

## 3. Comparative Summary

| Dimension | CODEX | GEMINI |
|---|---|---|
| Scope discipline (no broad CUDA refactor) | Strong | Weak — proposes `ds4_gpu_context[8]` refactor explicitly |
| Outcome contract (SHIP/EXTEND/STOP) | Partial (STOP via artifact only) | Missing |
| Phase kill-gates | Implicit (STOP artifact) | Only in Phase 2 |
| Public API growth bounded | Yes, but punted to Open Question 1 | No — public `ds4.h` model struct grows |
| Single vs. dual upload provider | Single (shard) | Single (shard) |
| Reconciliation logic placement | In `ds4.c` (heavyweight) | In `ds4.c` (heavyweight) |
| Tolerance vs. planner | Qualitative | Qualitative |
| CPU/stub test mode | Implied, not specified | Missing |
| Risk-table completeness | 6 rows, mostly correct | 4 rows, one mitigation contradicts intent |
| Test infrastructure | Synthetic parser tests planned | One-line "unit test" mention only |
| Artifact naming explicit | Partial | Mostly missing |
| Source-faithful packed bytes invariant | In Architecture | In Architecture, more prominent |
| Decode guard preservation | Explicit | Explicit, but DoD leaves an "unless" escape |

### 3.1 Common Gaps Across Both Drafts

Both drafts share several gaps that should be addressed before either is
merged:

1. No quantitative VRAM tolerance against the Sprint 003 planner.
2. No dual upload provider (`gguf` mmap vs. `shard` pread) and no
   cross-provider byte-equality check.
3. No tensor-count gate in DoD (parser row count must match the real model
   tensor count).
4. No verification that uploaded device memory is in fact device-resident
   (e.g., `cudaPointerGetAttributes` check).
5. No CPU/stub end-to-end smoke that exercises the orchestration code
   without CUDA, so the cluster is the first place orchestration bugs
   appear.
6. No explicit policy on `cudaDeviceReset` before/after the smoke to
   avoid prior-process residue.
7. No header-by-name parsing requirement for `pack-index.tsv` to defend
   against future column-order drift in `ds4-v100-pack --write-index`.
8. Neither draft commits a reserve value (3 GiB vs. measured) in the DoD.
9. Neither draft requires `git diff --check` and `make cpu` in the DoD
   even though both appear in the intent's verification strategy.
10. Neither draft addresses partial-upload device state recovery.

### 3.2 Recommendation

- **CODEX draft** is the closer-to-merge artifact. Its scope discipline,
  STOP-artifact pattern, and public-API caution all match the intent.
  Resolving Open Question 1 (where the diagnostic entry lives), adding a
  second upload provider, quantifying the VRAM tolerance, splitting
  reconciliation into a dedicated `ds4_pack` module, and expanding the
  risk table to cover scope creep / fixture rot / partial uploads would
  bring it to merge quality.
- **GEMINI draft** needs structural changes before merge: drop the
  `ds4_gpu_context[8]` refactor in favor of a minimal sidecar arena
  API, replace the "multi-step emission" mitigation with a hard
  persistent-scratch dependency, expand the risk table to at least the
  intent's three uncertainty dimensions, and rewrite the DoD to be
  measurable (artifacts named, tolerances quantified, "unless" clauses
  removed). The architecture framing is sound; the implementation,
  risks, and DoD sections need to catch up.

---

## 4. Items Both Drafts Should Adopt From The Intent

These items in `SPRINT-004-INTENT.md` are partially or fully absent from
both drafts:

- **Success criterion #3.** "Full real-model shard emission is run on
  persistent scratch or an explicit STOP is recorded with the concrete
  storage blocker." Codex captures STOP; Gemini does not.
- **Success criterion #5.** "Source-model generation guard remains in place
  unless a bounded correctness harness is implemented and passes." Both
  drafts preserve the guard, but Gemini's DoD wording leaves an escape.
- **Verification strategy local items.** `make cpu`, `make
  tools/ds4-v100-pack`, `git diff --check`, and synthetic-index unit tests
  are all listed in the intent. Codex covers most; Gemini covers few.
- **Architecture uncertainty: High.** "The current CUDA backend is largely
  global and single-device-oriented." Codex's narrow sidecar respects this;
  Gemini's proposed refactor underestimates it.
- **Open Question 5 resolution.** Both drafts leave it open; the intent
  effectively pre-positions "all of the above" (file sizes + per-shard
  SHA-256 + per-tensor spot checks). The final plan should commit a
  resolution.
