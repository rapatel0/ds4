# Sprint 005 Draft Critique

## Codex Draft

### Strengths

- **Clear architectural separation**: Splits the probe into a generic BF16 matrix-view primitive and a thin `token_embd.weight` resolver. This layering pays forward to Sprint 006+ without over-engineering.
- **Explicit fail-closed semantics**: Phase 1 calls out validation for zero rows, out-of-range indices, invalid arena spans, and undersized output buffers. DoD requires these to pass.
- **Strong scope fencing**: Repeatedly states what the sprint is *not* (no execution-context refactor, no decode enablement, no persistent dequantized buffers) with justification.
- **Comprehensive risk table**: Six risks with likelihood, impact, and concrete mitigations — notably the F16/BF16 confusion and the "probe accidentally reads GGUF bytes" failure mode.
- **Reusable tensor contract use case (UC4)**: Explicitly frames the output as a primitive Sprint 006 inherits, creating accountability for API stability.
- **Security section is specific**: Names bounds validation, read-only invariants, no network exposure, and no arbitrary pointer export.

### Weaknesses

- **Phase structure is front-heavy**: Phase 1 combines API definition, fail-closed validation, CPU reference implementation, *and* dtype-dispatch in `ds4.c`. This is at least two distinct deliverables conflated into one phase.
- **Open Question 4 is a scope leak signal**: Asking whether to fix the existing CPU token-embedding helper "in the same sprint" is dangerous — if answered yes it drags in q2/q4 code path regressions. The draft should take a position.
- **No concrete bit-pattern test vectors**: Phase 1 says "exact synthetic expected values" but doesn't specify even one canonical BF16 bit pattern (e.g., `0x3F80 → 1.0f`). The Gemini draft does better here.
- **HC repetition wrapper is under-specified**: Mentioned in design rules and use cases but never surfaced in a phase task or DoD item. If it ships, it's untested; if it doesn't, UC2's "repeated row selection" is ambiguous.
- **`ds4.c` dtype-dispatch is conditional ("if still needed")**: This hedging leaves it unclear whether the task is in-scope or not — it should be scoped in or explicitly deferred.
- **No performance baseline**: Even for a diagnostic sprint, knowing that a 4096-element BF16 row gather takes <1ms on V100 would catch gross kernel launch overhead bugs. Neither a timing assertion nor a "record latency" task appears.

### Scope Risks

- The `ds4.c` dtype-awareness fix touches shared CPU paths for q2/q4 — regressions here could break existing `make cpu` targets unrelated to BF16.
- The "focused probe-only mode" in Phase 3 (uploading just the needed BF16 span) is a mini-feature that could expand into a partial-upload subsystem.
- Open Question 3 (probing a second BF16 tensor) would double validation surface area if answered yes mid-sprint.

### DoD Completeness

- Covers API existence, arena-only reads, BF16 correctness, model-less tests, CUDA build, cluster probe, no persistent copies, generation guard, and build verification. **Strong.**
- Missing: no explicit DoD item for the "reusable tensor contract" use case — how do we know the descriptor is stable enough for Sprint 006?
- Missing: no DoD item asserting `make test` still passes unchanged (the draft says "scope unchanged unless..." which is ambiguous).

---

## Gemini Draft

### Strengths

- **Concrete API sketch in the doc**: The `ds4_gpu_probe_descriptor` struct and `ds4_gpu_probe_bf16_gather` signature give reviewers something to critique before implementation. Reduces ambiguity.
- **Explicit BF16 conversion algorithm**: Describes the shift/pad bit manipulation and mentions subnormal/NaN handling. This removes guesswork for implementers.
- **Phase 1 is tightly scoped**: Only BF16 logic + unit tests with exact bit patterns. Clean deliverable boundary.
- **Open Question 2 (F32 control tensor)**: Smart isolation idea — probing a known-F32 tensor separates arena access bugs from BF16 conversion bugs. Codex doesn't raise this.
- **Leaner file set**: 8 files vs Codex's 10. Fewer new artifacts reduces merge friction.
- **Backends table is clear**: One-line summary of stub vs CUDA purpose prevents confusion.

### Weaknesses

- **Thinner risk section**: Only 4 bullet-point risks vs Codex's 6-entry table with likelihood/impact. Missing the "probe reads GGUF bytes" and "re-upload cost" risks.
- **No explicit fail-closed semantics in DoD**: Says "range checks prevent out-of-bounds" but doesn't require *specific* failure modes (return code? abort? clamp?). The Codex draft is more prescriptive.
- **Phase 2 conflates bounds checking with stub implementation**: The "handle invalid token_ids (clamping or failing)" choice is deferred to implementation time rather than decided in the plan. This creates spec ambiguity.
- **No security section depth**: Mentions bounds checks and generation guard but doesn't address read-only invariants, no-network-exposure, or diagnostic-only output surface.
- **Missing Use Case for maintainability/diagnostics**: Codex UC5 ("fail-closed diagnostics") and UC4 ("reusable tensor contract") have no equivalent. The Gemini draft focuses on the happy path.
- **Phase 5 has no concrete artifact**: Says "archive the probe output" but doesn't name a log file or format. Codex names `SPRINT-005-RESIDENT-BF16-PROBE.log`.
- **`ds4.c` not touched**: The existing F16-as-BF16 bug in source-layout helpers is not addressed. If Sprint 005 is the BF16 correctness sprint, leaving this known bug unfixed is a missed opportunity (though arguably reduces scope risk).
- **No mention of HC repetition**: The intent doc mentions "repeated HC expansion if included" but the Gemini draft doesn't address it at all.
- **Stream parameter in API but no stream management discussion**: The `ds4_gpu_stream stream` parameter appears in the signature but the doc doesn't discuss who creates/owns it or what happens if `NULL` is passed.

### Scope Risks

- Open Question 1 (F16 output support) could inflate the conversion kernel and test surface significantly.
- Open Question 3 (multi-GPU token splits) is a non-issue per the layout doc, but raising it without resolving it may cause mid-sprint confusion.
- Phase 4's `--compute-probe` flag in the residency smoke tool adds user-facing CLI surface that needs documentation.

### DoD Completeness

- Covers implementation existence, BF16 correctness, row gathering, range checks, local tests, CUDA build, cluster smoke, and vision update. **Adequate but less rigorous than Codex.**
- Missing: no DoD item for "no persistent dequantized copies" (Codex has this).
- Missing: no DoD item for "generation guard remains active" (Codex has this).
- Missing: no DoD item for `git diff --check` or build verification beyond `make cpu`.
- Missing: no assertion that the CUDA path reads from arena memory specifically (not model_map) — Codex DoD item 2 is explicit about this.

---

## Comparative Summary

| Dimension | Codex | Gemini |
|-----------|-------|--------|
| Scope discipline | Stronger — explicit boundary statements | Adequate but less defensive |
| API concreteness | Descriptive only | Includes code sketch |
| Risk analysis | Deeper (6 risks, likelihood/impact) | Lighter (4 bullets) |
| DoD rigor | 9 items, covers negative invariants | 8 items, mostly positive assertions |
| Phase granularity | Front-heavy Phase 1, otherwise good | Cleaner phase boundaries |
| Security | Specific and actionable | Present but thin |
| Implementer guidance | High — design rules section | Higher — bit-level algorithm described |
| Missing from intent | HC repetition under-specified | HC repetition absent |
| Artifact traceability | Named log file | Unspecified archive |

**Recommendation**: Merge the Codex draft's scope fencing, fail-closed DoD items, and risk depth with the Gemini draft's concrete API sketch, clean phase boundaries, and explicit BF16 algorithm description. Resolve the `ds4.c` dtype-dispatch question (defer to follow-up) and drop HC repetition from Sprint 005 scope to keep the sprint bounded.
