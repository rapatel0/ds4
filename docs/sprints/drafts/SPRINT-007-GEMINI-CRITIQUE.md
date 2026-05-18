# SPRINT-007 GEMINI Critique: Source-Layout Single-Slot Decode Correctness

This critique evaluates `SPRINT-007-CLAUDE-DRAFT.md` (Claude) and `SPRINT-007-CODEX-DRAFT.md` (Codex) against the requirements in `SPRINT-007-INTENT.md`.

## 1. Claude Draft Evaluation

### Strengths
- **Surgical Precision:** Identifies exact line numbers and mislabeled functions in `ds4.c` (e.g., `embed_token_f16`, `output_hc_head_one`) that need correction for source-layout support.
- **Robust Guarding:** Proposes a specific, hardcoded "unlock token" string literal (`DS4_V100_ORACLE_UNLOCK_TOKEN`) for the engine options, which is much safer and more auditable than a simple boolean flag.
- **Detailed Primitives:** Defines clear C function signatures for the row primitives, including error handling and shape validation.
- **Phased Gating:** Every phase has an explicit "Kill gate" (STOP/EXTEND criteria) which is essential given the high correctness risk of FP8/MXFP4 semantics.
- **Architecture Alignment:** Perfectly utilizes the Sprint 006 descriptor policy and classifier to drive the oracle dispatch.

### Weaknesses
- **Module Bloat:** Proposes a new `ds4_v100_oracle.c/h` sidecar. While clean, this may lead to code duplication with existing reference paths in `ds4.c` if not carefully managed.
- **Timing Risk:** The 30-minute-per-token limit on the cluster might be a major blocker for routine validation; it might be better to prioritize a "fast-reference" (device-side or SIMD-optimized) earlier if CPU is that slow.

### Gaps & Edge Cases
- **I32/Index Tensors:** Briefly mentions I32 in Open Question 6 but doesn't explicitly include I32 (used for expert indices) in the primitive set or the P1 tasks.
- **Memory Pressure:** While it avoids persistent dequantized copies, the "row tile in scratch" size for MXFP4 Expert weights (256 experts) could still be non-trivial if many are active.

---

## 2. Codex Draft Evaluation

### Strengths
- **Single Source of Truth:** Correctly identifies that `gguf-tools/quants.[ch]` should own the canonical dequant logic to synchronize the quantizer and the runtime oracle.
- **Conciseness:** Provides a high-level view that is easier to digest for a quick architectural review.
- **Model-less Focus:** Strong emphasis on "Phase 1: Shared Helpers" as the foundation.

### Weaknesses
- **Vague Implementation:** Lacks the specific call-site audits found in the Claude draft. It doesn't name the functions in `ds4.c` that are currently "mislabeled."
- **Weak Guarding:** The "explicit source-oracle flag" is less secure than Claude's unlock token; it's easier to accidentally enable in a production config.
- **Missing Specifics:** Does not define the exact API for the helpers, making it harder to estimate the work for P2 and P4.

### Gaps & Edge Cases
- **Descriptor Integration:** Doesn't explicitly mention how it will consume the Sprint 006 descriptors, which are the source of truth for the V100 layout.
- **Logprob Membership:** Proposes top-logprob membership as a "minimum contract" but doesn't emphasize the importance of exact first-token matching as strongly as Claude.

---

## 3. Comparative Summary

| Dimension | Claude Draft | Codex Draft | Recommendation |
|---|---|---|---|
| **Specificity** | High (exact lines/symbols) | Medium (general areas) | **Claude** for implementation detail. |
| **Security/Guarding** | High (Unlock Token) | Medium (Boolean Flag) | **Claude**'s token is superior. |
| **Code Structure** | Sidecar Module | Integrated Helpers | **Codex**'s shared-helper focus is better for DRY. |
| **Risk Mitigation** | Detailed Kill Gates | General Phase Gates | **Claude** for rigorous risk management. |

### Missing from Both
- **I32/Index correctness:** Neither draft explicitly plans to verify the `ffn_gate_tid2eid` (I32) index tensors which are critical for the routed expert path.
- **CPU Parallelism:** Given the potential slowness of the reference path, neither draft discusses using OpenMP or simple multi-threading for the oracle decode to keep cluster time reasonable.

## 4. Final Verdict for SPRINT-007-REPORT.md

The **Claude Draft** is the superior foundation due to its deep integration with the existing V100 context/descriptor logic and its superior "unlock token" safety mechanism.

**Recommendation for Merge:**
1. Use **Claude's** 6-phase structure and safety guarding.
2. Adopt **Codex's** recommendation to promote the core dequant primitives to `gguf-tools/quants.[ch]` for cross-tool consistency.
3. Add an explicit task to verify **I32 index tensors** in Phase 1 or 2.
4. Add a note to **Phase 5** regarding the use of simple CPU parallelism (e.g., `#pragma omp parallel for`) if the 30-minute-per-token limit is exceeded.
