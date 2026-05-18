# SPRINT-007 Critique: Claude vs Gemini

This critique is specific to the DS4 V100 source-layout sprint. The key bar for Sprint 007 is not "can we almost decode," but "can we prove exact source-layout semantics and run a guarded one-token oracle path without weakening the existing fail-closed generation guard."

## Executive Verdict

- Claude is the stronger base draft. It treats Sprint 007 as a correctness gate, keeps the V100 numeric policy intact, and is much more explicit about fail-closed behavior.
- Gemini is readable and directionally right, but it is too loose on source-format semantics, guard containment, and no-regression proof for the legacy path.
- Best merge direction: use Claude as the base, then simplify wording where useful without dropping its kill gates or oracle-only boundary.

## Claude Draft

### Strengths

- It is correctly scoped as a source-layout correctness sprint, not a performance or deployment sprint. Prefill, KV growth, multi-slot, server, and production kernels stay out of scope.
- It centers the real DS4 risk: exact F8_E4M3_B128 and MXFP4 semantics must match the upstream dequant source of truth before any oracle decode result is trusted.
- The fail-closed story is strong. The explicit unlock token, `oracle_only` engine mode, unchanged normal guard, and wrapper rejection path are all appropriate for this repo.
- The implementation shape is credible for the codebase: descriptor-keyed dispatch, per-family oracle matvecs, synthetic primitive tests first, then synthetic decode, then cluster `official.vec` comparison.
- The draft has the best execution contract of the two through explicit `SHIP` / `EXTEND` / `STOP` outcomes.

### Weaknesses

- It is somewhat over-specified. The number of phases, files, smoke tests, logs, and follow-up docs adds process weight beyond the narrow proof this sprint needs.
- It still implies a fairly broad sweep through `ds4.c` reference call sites. For a one-token oracle sprint, that is real churn and raises integration risk.
- The draft sometimes assumes that exact primitive dequant equality plus top-K agreement is close to end-to-end correctness. That is useful evidence, but it is not the same as a full local oracle.

### Gaps In Risk Analysis

- It does not foreground the risk that `official.vec` is only a token/top-logprob slice, so mismatches may come from reference incompleteness, tokenizer differences, or serving-side behavior rather than primitive bugs.
- It understates how much hidden behavior may live inside `forward_first_token_cpu` besides matvec dispatch: scratch assumptions, routed-expert plumbing, normalization order, and control-flow details.
- The host-side pack-read proposal is practical, but the draft does not call out the architectural risk that correctness is being proven through a path that bypasses the resident-arena story Sprint 006 just established.

### Missing Edge Cases

- Descriptor exists but its row/scale metadata does not actually match the runtime pack row layout.
- F8/MXFP4 rows with unexpected divisibility, zero columns, or malformed scale spans.
- MXFP4 routed-expert edge cases: zero selected experts, duplicate expert ids, or out-of-range expert ids.
- Partial source-layout classification where one tensor family silently falls back to a legacy path instead of failing closed.

### Definition Of Done Completeness

- This is the stronger DoD. It covers primitives, dispatch, guard regression, cluster artifacts, and explicit non-goals.
- It should be tighter on the verdict bar: the draft argues that exact first-token match should be the real `SHIP` bar, but the DoD still allows top-K membership as a passing outcome.
- It should promote legacy no-regression from a mitigation into an explicit DoD item for the existing `ds4flash.gguf` path.

## Gemini Draft

### Strengths

- It is concise and easy to scan. The high-level goal of a guarded source-layout correctness path is clear quickly.
- It points at the right major work items: source-dtype primitives, BF16/F32 cleanup, guarded decode, and `official.vec` comparison.
- The phase ordering is directionally sensible: primitives first, then decode wiring, then cluster evidence.

### Weaknesses

- It is too loose on containment. A public `ds4_cli.c` flag is the wrong surface for this sprint and is much easier to leak into normal usage than a dedicated oracle-only tool.
- It does not pin the exact source-format contract. Phrases like "likely E8M0 scale interpretation" are too soft for the main correctness blocker in DS4 source layout.
- It does not define descriptor-keyed dispatch or precise supported `(source_dtype, family)` pairings, so the plan can drift into broad `ds4.c` surgery without a hard boundary.
- The validation plan is much thinner than Claude's on unsupported pairings, wrapper rejection, and legacy-path no-regression.

### Gaps In Risk Analysis

- It misses the biggest sprint risk: runtime pack semantics may disagree with `gguf-tools/deepseek4-quantize.c` or with `SPRINT-003-PACK-INDEX.tsv`.
- It does not explicitly treat diagnostic-mode leakage into normal CLI/server/eval paths as a major risk.
- It does not call out the risk of destabilizing the legacy `matvec_f16` / `matvec_q8_0` path while cleaning up source-layout call sites.
- It does not address the possibility that `official.vec` is too weak a reference to prove correctness when logits are not available.

### Missing Edge Cases

- Wrong or missing unlock behavior across `ds4`, bench, eval, and any server/request entry points.
- Unsupported tensor families surfacing only mid-decode after some source-layout work has already run.
- Source-layout tensors that are easy to skip if cleanup focuses only on embedding and output head: HC control, router/control tensors, compressor/indexer projections, and routed-expert-only paths.
- Real-pack shape/stride mismatches that synthetic fixtures would not catch.

### Definition Of Done Completeness

- The DoD is materially weaker for this repo's bar. It lacks explicit `SHIP` / `EXTEND` / `STOP` criteria and does not make fail-closed proof a first-class acceptance item.
- It does not require a durable no-regression check for the legacy model path.
- It should require source-format semantics to be locked against upstream dequant behavior before calling the sprint's decode result "correct."

## Recommendation

- Use Claude as the base draft.
- Borrow only Gemini's brevity, not its public CLI-flag approach or its looser acceptance criteria.
- Before finalizing, tighten Claude in two places: make exact first-token agreement the clear `SHIP` bar, and make legacy-path no-regression an explicit part of the DoD.
