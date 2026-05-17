# SPRINT-005 GEMINI Critique: BF16 Resident Compute Probe

## Overview
Both drafts correctly identify the critical transition from "residency" (Sprint 004) to "compute contract" (Sprint 005). They agree on using `token_embd.weight` (BF16) as the first target. Claude's draft is more procedurally detailed and provides specific C code for the BF16 conversion, while Codex's draft focuses on the "View" abstraction and reuse of the probe contract for future layers.

---

## 1. Claude Draft Critique

### Strengths
- **Implementation Detail:** Provides the `bf16_to_f32` implementation and a specific list of test vectors (NaN, Inf, Subnormals), which reduces research time during execution.
- **API Design:** Proposes two distinct layers: a raw `row_gather` and a model-specific `embed_hc`. This captures the DS4 hyper-connection (HC) requirement early.
- **Risk Mitigation:** Explicitly addresses the "BF16 vs FP16" divergence with a mandatory test case, which is a high-likelihood source of silent bugs.
- **Host-Stub Symmetry:** Strong emphasis on proving the contract works in the host-stub before going to CUDA.

### Weaknesses
- **Monolithic Probe:** The probe API takes many parameters (`weight_offset`, `weight_bytes`, `row_stride`, etc.) without a "View" or "Descriptor" struct, which might lead to signature bloat as more dtypes (FP8, MXFP4) are added.
- **Validation Scope:** Does not explicitly mention verifying that the *output* host buffer remains within bounds if the `n_hc` expansion is large.

### Scope Risks
- **Phase 4 Optionality:** Claude marks the cluster test as optional for "SHIP". While pragmatic, shipping a compute contract without seeing it run on a V100 is a risk for the successor sprint (006).

---

## 2. Codex Draft Critique

### Strengths
- **Architectural Abstraction:** Introduces the "BF16 probe descriptor" / "Matrix View" concept. This is superior for long-term maintainability as it decouples the *tensor lookup* (metadata) from the *compute kernel*.
- **Tooling Focus:** Proposes a "probe-only" mode for the residency smoke tool to avoid re-uploading 145GB. This is a significant developer-experience win.
- **Integration Plan:** Explicitly mentions updating `ds4.c` CPU paths to be dtype-aware, preventing the existing codebase from "lying" about BF16 values by treating them as F16.

### Weaknesses
- **Vague on Conversion:** Lacks the explicit bit-shifting logic found in Claude's draft. "Explicit BF16-to-F32 helper" is a bit hand-wavy compared to Claude's `__uint_as_float((uint32_t)x << 16)`.
- **Definition of Done:** Slightly less granular on the exact test cases for the CUDA unit target compared to Claude's list of edge cases.

### Scope Risks
- **Refactoring `ds4.c`:** Modifying existing diagnostic paths in `ds4.c` to be "dtype-aware" is a "High" impact risk for "Low" likelihood of regression, but it could distract from the primary goal of the *resident* probe.

---

## 3. Comparative Summary & Missing Edge Cases

| Dimension | Claude | Codex | Recommendation |
|---|---|---|---|
| **API** | Functional / Flat | Descriptor / View | **Descriptor/View:** Use Codex's abstraction for the probe contract. |
| **Conversion** | Detailed / Explicit | Conceptual | **Claude's Logic:** Use the bit-shift implementation. |
| **Testing** | Edge-case heavy | Tooling/DX heavy | **Combine:** Use Claude's edge cases in Codex's new unit test. |
| **Cluster** | Optional | Required | **Required:** A compute probe isn't "done" until it computes on the target HW. |

### Missing Edge Cases (Both Drafts)
1. **Misaligned Access:** BF16 is 2 bytes. If a `weight_offset` or `row_stride` is odd, CUDA kernels might perform unaligned reads or crash. The API should enforce or validate 2-byte alignment.
2. **HC Expansion Limit:** If `n_hc` is high (e.g., 2048), the output buffer could be several megabytes. The probe should check for `INT_MAX` or memory limits on the output size.
3. **Stream Management:** While both defer "execution context," the CUDA probe should ideally take an optional `cudaStream_t` to avoid blocking the default stream, or at least document that it uses the default stream for now.

---

## 4. Definition of Done Completeness
- **Claude:** Excellent. Includes "git diff --check", bit-exactness, and specific row-gather patterns (row 0, last row).
- **Codex:** Good. Stronger on the "Resident span facts" (logging *where* in the arena the data came from), which is vital for debugging residency.

**Final Verdict:** The best Sprint 005 plan should adopt **Codex's View/Descriptor abstraction** and **tooling efficiency**, but use **Claude's explicit conversion logic and rigorous edge-case test list**. Cluster validation should be mandatory for a "compute" sprint.
