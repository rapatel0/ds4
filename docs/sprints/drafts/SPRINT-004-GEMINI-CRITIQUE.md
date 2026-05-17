# SPRINT-004 Draft Critique: GEMINI-CRITIQUE

This document evaluates the Sprint 004 drafts produced by Claude and Codex based on the `SPRINT-004-INTENT.md` requirements.

---

## 1. CLAUDE-DRAFT Evaluation

### Strengths
- **Actionable Phasing:** The breakdown into Phases 0-6 with explicit "Kill gates" is superior. Phase 0 (Orientation) is a critical addition for cluster work, ensuring environment readiness before committing to code changes.
- **Concrete API Design:** Proposes specific C signatures (`ds4_pack_open`, `ds4_gpu_arena_upload`) and module boundaries (`ds4_pack.c`, `ds4_gpu.h`). This reduces ambiguity during implementation.
- **Resolution of Open Questions:** Directly addresses all five open questions from the Intent document with clear, justified stances.
- **Validation Rigor:** The list of specific artifacts (SHA256 TSVs, Reconcile logs, Residency logs) and the "one deterministic cross-tensor compare" provide high confidence in the structural correctness goal.
- **Safety First:** Explicitly maintains the generation guard and avoids the "production multi-GPU API" trap by keeping the arena API as a "residency-only" sidecar.

### Weaknesses
- **Complexity Overhead:** Introducing `ds4_pack.c/h` and `ds4_gpu_arena` as separate modules might be slightly high for a single-sprint "smoke" task, although justified by the need for clean boundaries.
- **Dependency on Persistent Scratch:** While it handles the blocker well with kill gates, it doesn't offer a "plan B" if scratch is unavailable other than rolling to Sprint 005.

### Risk Analysis Gaps
- **Inter-GPU Resource Contention:** Does not explicitly address potential side effects of 8 GPUs all pulling ~20GB from the same host scratch drive or GGUF mmap simultaneously (IO saturation).
- **Driver/Toolkit Mismatches:** Phase 0 mentions CUDA 12 but doesn't detail a check for `p2p` support or specific V100 driver features that might affect `cudaSetDevice` behavior in the pod environment.

### Missing Edge Cases
- **Interrupted Uploads:** No mention of how to handle or detect a partial upload if the pod is preempted or the process killed mid-145GB transfer.
- **Multi-Model Collision:** Handling multiple GGUFs or Indices in the same scratch directory (e.g., versioning).

### Definition of Done (DoD) Completeness
- **High.** Covers implementation, validation, artifacts, and documentation. The inclusion of "No persistent F16/F32 dequantized copies" is a critical architectural constraint check.

---

## 2. CODEX-DRAFT Evaluation

### Strengths
- **Focus on Use Cases:** The "Use Cases" section is excellent for aligning stakeholders on *why* this sprint matters and what the developer/operator experience will be.
- **Effort Weighting:** Providing estimated effort percentages (~30%, ~20%, etc.) helps with project management and expectations.
- **Conciseness:** Very readable and gets straight to the "structural proof" goal.
- **Mitigation of "Fake Success":** Identifies the risk of host-mapped fallbacks making the smoke look successful when it isn't.

### Weaknesses
- **Vague Implementation Detail:** Lacks specific C API definitions or file-level change descriptions compared to Claude.
- **Open Questions:** Touches on the intent's open questions but doesn't provide a structured "Resolution" section, making it harder to verify if all concerns were addressed.
- **Kill Gate Omission:** Does not identify early exit points (like Phase 0/1 in Claude) as clearly, which is risky for cluster-dependent work.

### Risk Analysis Gaps
- **Refactor Scope Creep:** Identifies the risk but the mitigation ("Keep Sprint 004 limited...") is somewhat generic without the "sidecar API" boundary proposed by Claude.
- **Security:** Mentions input validation but doesn't specify *how* (e.g., against GGUF metadata).

### Missing Edge Cases
- **GPU Topology Mismatch:** Doesn't explicitly state what happens if the `owning_gpu` in the index exceeds the `cudaGetDeviceCount` on the actual node.
- **Index/GGUF Drift:** While it mentions reconciliation, it doesn't detail the "STOP" condition for manifest errors.

### Definition of Done (DoD) Completeness
- **Medium.** The checklist is functional but lacks the artifact-specific detail (e.g., SHA256, specific log names) found in the Claude draft.

---

## 3. Comparative Summary & Recommendation

| Feature | CLAUDE-DRAFT | CODEX-DRAFT | Winner |
|---|---|---|---|
| **Actionability** | High (Phased/Gates) | Medium (Effort %) | Claude |
| **API Specificity** | High (C Signatures) | Low (Conceptual) | Claude |
| **Risk Mitigation** | Detailed (Kill Gates) | Conceptual | Claude |
| **User Focus** | Medium | High (Use Cases) | Codex |
| **Artifact Definition** | High (TSV/Log names) | Medium | Claude |

**Recommendation:**
The **CLAUDE-DRAFT** should be the primary template for SPRINT-004. It is significantly more "ready-to-code" and provides the necessary safety gates for expensive cluster runs. However, it should be augmented with the **CODEX-DRAFT's** "Use Cases" section and effort estimations to provide better context for the Sprint Report.

**Critical Addition for the Final Plan:**
Both drafts miss a specific check for **Peer-to-Peer (P2P)** capability between the V100s. While not needed for this sprint's residency smoke, verifying P2P visibility *now* (as part of Phase 0 or the Smoke Tool) would provide invaluable signal for the Sprint 005 HC-relay planning.
