# SPRINT-007 Draft Critique

Reviewing CODEX and GEMINI drafts against `SPRINT-007-INTENT.md`. Both target
the same goal — guarded source-layout single-slot decode correctness — but
diverge in scope, rigor, and how they protect the V100 guard boundary.

---

## CODEX Draft

### Strengths

- **Helper-surface design is right.** Promoting E8M0 / E4M3fn / MXFP4 decode
  into `gguf-tools/quants.[ch]` and having `deepseek4-quantize.c` consume the
  same helpers (Phase 1) directly addresses the intent's "one canonical
  interpretation of source bytes" concern. GEMINI is silent on the quantizer
  re-use.
- **Dispatch breadth is enumerated.** Phase 2 names the actual call sites that
  must change (embedding, output head, HC, router, compressor/indexer, routed
  experts) rather than only listing dtypes. This matches the intent's
  enumeration of legacy F16/Q8_0 leakage points.
- **Bounded-scratch rule is explicit.** Phase 2 and DoD both repeat the
  "no persistent F16/F32 mirrors of large source tensors" invariant —
  directly enforces the layout-doc rule.
- **Guard regression test is a first-class deliverable.** Phase 4 includes a
  test that source-layout open outside oracle mode still fails, plus an
  archived `SPRINT-007-GUARD.log`. That is the kind of evidence the intent
  asks for.
- **Hard-stop gate is named in DoD.** "If FP8/MXFP4 semantics cannot be
  established, exit with STOP" appears in the DoD itself, not just risks.
- **Mode lives in `ds4_engine_options`,** keeping the bypass inside the
  existing engine surface rather than bolting on a parallel entry point.

### Weaknesses

- **Phase weighting omits 0%.** 20+35+15+20+10 = 100, fine, but Phase 3
  (the guarded mode itself) at 15% feels thin given that the intent flags
  scope creep and guard-weakening as the two highest-impact risks. The
  Phase 3 task list (4 bullets) under-specifies how the oracle mode
  interacts with `ds4_v100_context` from Sprint 006 — the intent explicitly
  ties the oracle path to descriptor binding, and CODEX never names the
  context object.
- **Output-head BF16 not called out as a DoD line.** The intent lists "BF16
  output-head support for source layout" as a parking-lot item now relevant.
  CODEX folds it into a Phase 2 bullet but the DoD only says "BF16
  embeddings or output weights are not interpreted as F16," which is
  weaker.
- **No mention of `ds4_cli`.** Whether the oracle is reachable only through
  `ds4_test` or also through CLI is left to an Open Question. That is a
  reasonable deferral, but the DoD should pin which surface exposes the
  mode so the guard test can target it deterministically.

### Risk-analysis gaps

- **No risk for E8M0 scale-byte position or E4M3 variant ambiguity.** The
  intent's open questions explicitly call out "scale as first/last byte,
  E8M0 interpretation, E4M3 variant" — CODEX merges these into a single
  "FP8 semantics differ" row.
- **No risk for fixed-vs-original source GGUF divergence.** GEMINI raises
  this (its Open Question 4) and the intent implies it via the
  `DSv4-Flash-256e-fixed.gguf` artifact name; CODEX is silent.
- **No risk for official.vec being too sparse** to discriminate a passing
  oracle from a coincidentally-matching wrong implementation (top-logprob
  membership can be very forgiving on common tokens).
- **No risk for shared helpers regressing the existing quantizer.** Once
  `deepseek4-quantize.c` is switched to the shared surface, a bug in the
  helper breaks both the oracle and the existing quantizer artifact. No
  mitigation (e.g., golden-bytes round-trip test on a known pack).

### Missing edge cases

- **Mixed-dtype tensors within an MoE layer.** Routed experts (MXFP4) plus
  shared experts or projections at a different dtype in the same layer — the
  reference dispatcher needs to fail closed on unexpected combinations, not
  just on per-tensor type mismatch.
- **Tokenizer / vocab parity.** Even with correct logits, an off-by-one in
  the BPE vocab between the source GGUF and the official vectors will make
  the oracle look broken. Not addressed.
- **RNG/sampling determinism.** Comparison is described as "selected-token
  equality," but selection requires temperature=0 / greedy and a defined
  tie-break. Not specified.
- **Numerical accumulation order.** FP8 blocked matvec accumulation order
  matters for bit-exact comparisons; CODEX promises "exact" semantics
  without saying whether accumulation is FP32 and whether order is
  deterministic.
- **Endianness / alignment** of E8M0 and packed nibbles in
  `quants.c` is not constrained.

### Definition of Done completeness

- Covers: shared helpers, dispatch correction, mode existence, smoke,
  one official-vector case, archived logs, no-persistent-dequant invariant,
  guard still rejects, STOP escape hatch.
- Missing: explicit pass criterion for the model-less smoke (exact bytes
  vs tolerance?), explicit assertion that the quantizer's existing output
  is byte-identical after the helper migration, and an artifact for
  cluster build success on `sm_70`. "Match selected-token behavior plus
  official top-logprob membership" is the weakest line — it should bound
  how many steps and how many prompts.

---

## GEMINI Draft

### Strengths

- **Cleaner narrative framing.** The "correctness gate between skeleton
  and performance" framing in the Overview is sharper than CODEX's longer
  prose and lands the sprint's purpose quickly.
- **Explicit tie-in to Sprint 006 descriptors.** Phase 2 ("Validate that
  embedding/control tensors bind correctly to Sprint 006 descriptors") and
  Phase 3 ("Link the diagnostic path to the Sprint 006 context for tensor
  lookup") name the integration point CODEX glosses over.
- **Names a public C entry point.** `ds4_verify_source_layout_correctness()`
  is a concrete API contract; CODEX leaves the surface shape implicit in
  "an explicit source-oracle flag or mode."
- **Fixed-vs-original GGUF question is raised** (Open Question 4) — a
  real artifact distinction CODEX misses.
- **CLI flag is named** (`--verify-source`), making the test surface
  obvious from day one.

### Weaknesses

- **Helper surface is under-scoped.** Phase 1 mentions FP8 and MXFP4
  primitives but does not require unifying them with
  `deepseek4-quantize.c`. The intent explicitly worries about divergent
  interpretations; GEMINI leaves two copies possible.
- **Dispatch breadth is vague.** Phase 2 says "Update HC/router/output
  control matvecs" without enumerating the compressor/indexer BF16
  family or routed-expert MXFP4 dispatch. The intent calls these out
  by name.
- **"Force-open" framing of the bypass is dangerous.** The Architecture
  diagram literally labels the diagnostic mode as "Force-open ... bypassing
  the `ds4_engine_open` guard." That phrasing risks producing an
  implementation where the guard is sometimes skipped rather than one
  where the guard always runs and an alternate, narrower entry exists.
  CODEX's `ds4_engine_options` flag is structurally safer.
- **No quantizer parity tasks.** If the helper surface is shared, the
  quantizer must keep producing identical bytes; if not shared, the two
  must be diff-tested. Neither appears.
- **Phase weighting absent.** No effort percentages — harder to judge
  whether the sprint is realistic.
- **Smoke test scope is thin.** Phase 1 says "model-less unit tests for
  each primitive" but does not require failure tests for bad block
  shapes or unsupported dtypes, which the intent's success criteria
  treat as part of fail-closed proof.

### Risk-analysis gaps

- **Only 4 risks, vs. CODEX's 5.** Notably absent: scope creep into a
  second runtime (CODEX has this; given the intent's "Architecture
  uncertainty: Medium," it should be present here).
- **No risk for shared-helper regression** of the existing quantizer
  (because GEMINI does not require sharing).
- **No risk for tokenizer / sampling mismatch** producing false
  failures in the oracle.
- **"Diagnostic mode leaks into production" is rated Low.** Given the
  "force-open" framing in Architecture, the likelihood is realistically
  higher than Low; the mitigation ("strict opt-in flags") is generic.

### Missing edge cases

- **F32 control matvec is not in Phase 2 tasks**, only in DoD ("correct
  BF16/F32 types"). The intent calls out F32 control explicitly as a
  parking-lot item now in scope.
- **No mention of bounded scratch / no persistent dequantized weights.**
  This is a Security bullet only, not a DoD line. CODEX makes it a DoD
  invariant. The intent treats it as a hard constraint.
- **No artifact about cluster build / `sm_70` compile.** Phase 5 says
  "Run synthetic primitive tests on the V100 cluster," but the intent's
  verification strategy specifies `CUDA_ARCH=sm_70` build proof.
- **No `Makefile` entry** in Files Summary — adding new test sources
  and a smoke target almost always touches the build.
- **No mention of how many steps / prompts** the official-vector
  comparison runs for. Same gap as CODEX, but compounded by the lack
  of phase weighting.

### Definition of Done completeness

- Covers: primitives unit-tested, BF16/F32 dispatch corrected, guarded
  mode exists, official-vector comparison done, normal guard still
  fail-closed, blocker report if blocked, archived logs.
- Missing: explicit "no persistent dequantized weights" invariant,
  quantizer parity assertion, scope clause that forbids MTP / server /
  multi-slot in this sprint (the intent forbids these — CODEX restates
  the prohibition in its Architecture section, GEMINI does not), and
  a clause that the oracle path stays CPU-only.
- The "blocker report is produced if FP8/MXFP4 semantics cannot be
  established" is good, but unlike CODEX it does not commit to *not*
  weakening the guardrails in the STOP case.

---

## Side-by-side summary

| Dimension | CODEX | GEMINI |
|---|---|---|
| Helper-surface unification w/ quantizer | Yes | No |
| Names Sprint 006 context tie-in | Weak | Explicit |
| Public API surface for oracle | `engine_options` flag (safer) | `ds4_verify_*` entry + CLI (clearer) |
| Dispatch sites enumerated | Comprehensive | Partial |
| No-persistent-dequant in DoD | Yes | No (only in Security) |
| STOP-without-weakening-guard | DoD line | Implicit |
| Phase effort weighting | Yes | No |
| Routed-expert MXFP4 in tasks | Yes | No |
| Quantizer parity protection | Not addressed | Not addressed |
| Tokenizer / sampling determinism | Not addressed | Not addressed |
| Number of risks | 5 | 4 |
| Fixed-vs-original GGUF | Not raised | Raised (OQ) |

## Recommended merge

A stronger Sprint 007 plan would take CODEX's helper unification, dispatch
enumeration, no-persistent-dequant DoD line, and scope-creep risk, combined
with GEMINI's explicit Sprint 006 descriptor tie-in, named C entry point,
CLI flag, and the fixed-vs-original GGUF open question. Both drafts need to
add: quantizer byte-parity protection, sampling/tokenizer determinism, a
concrete bound on prompts × steps for the cluster run, and a `Makefile`
artifact in Files Summary.
