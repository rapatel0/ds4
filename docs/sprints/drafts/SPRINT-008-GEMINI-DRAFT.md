---
sprint: 008
title: Source Oracle Harness And V100 KV Anchors
status: draft
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: SPRINT-008-INTENT.md
---

# SPRINT-008: Source Oracle Harness And V100 KV Anchors

## Overview

Sprint 007 shipped a guarded, CPU-only source-layout oracle that proved exact semantics for BF16, F8_E4M3_B128, and MXFP4 tensors, matching the first token of the official `short_reasoning_plain` fixture. However, the comparison was a manual diagnostic step, and the runtime still lacks the prompt prefill, KV management, and device-side execution anchors needed for production.

Sprint 008 moves from a one-token oracle to a repeatable validation harness and establishes the V100 device-side surfaces for KV cache and source-format tensor access. It introduces prompt prefill logic, defines the KV budget for the layer-owned topology (including SWA-only, Ratio-4, and Ratio-128 classes), and creates the first production-relevant device-side anchors that validate on the V100 cluster.

The V100 precision policy remains:
- F16 KV is the baseline correctness contract.
- BF16 source tensors are decoded/converted for V100 use; no native BF16 Tensor Core GEMMs.
- F8_E4M3_B128 and MXFP4 are packed source inputs; kernels must unpack or dequant into FP16 HMMA tiles.
- Normal source-layout generation remains fail-closed until device-side correctness is proven.

## Use Cases

1. **Automated Vector Validation**: A developer can run a command to verify the current engine's logprobs against all relevant official vectors in `tests/test-vectors/official.vec`.
2. **KV Admission Check**: The runtime can calculate and enforce KV cache budgets for a given context length and layer topology before starting a session.
3. **Prompt Prefill Evidence**: The engine can process a multi-token prompt and establish the hidden-context state for the first generation step.
4. **Device-Side Source Anchor**: A CUDA-resident probe confirms that F8_E4M3_B128 or MXFP4 bytes can be addressed and expanded correctly on V100.
5. **MXFP4 Regresson Guard**: A targeted test ensures that the GGML-compatible low-half/high-half nibble ordering remains stable.

## Architecture

Sprint 008 expands the V100 context and adds an automation layer for the oracle.

```text
Official Vectors (JSONL)
    |
    v
Oracle Harness Tool
    |
    +--> --dump-logprobs
    |
    +--> Comparison Logic (Match Top-1 / Top-K)
    |
    v
V100 Context (Updated)
    |
    +--> KV Budget Surface (F16, SWA, Indexer)
    |
    +--> Prefill Logic & HC State
    |
    +--> Device-Side Anchors (CUDA)
             |
             v
      Source-Format Probes
      - F8_E4M3_B128 Device-Side Address/Expand
      - MXFP4 Device-Side Address/Expand
```

### KV Budget & Topology

The DS4 V100 appliance uses a layer-owned topology. Sprint 008 must codify the KV requirements for:
- **SWA-only layers**: Standard attention with Sliding Window.
- **Ratio-4 (Indexer) layers**: Compressed KV paths.
- **Ratio-128 layers**: High-compression paths.

The context must expose an admission surface that rejects sessions if the 8x V100 VRAM reserve (3 GiB per GPU) would be violated.

### Device-Side Anchors

A "device-side anchor" is a bounded CUDA kernel that proves the runtime can execute a specific source-format operation on the V100. For Sprint 008, this means a resident row-gather/expand probe for F8 or MXFP4, similar to the Sprint 005 BF16 probe, but targeting the more complex blocked/nibble-packed layouts.

## Implementation

### Phase 1: Oracle Automation & MXFP4 Hardening

**Tasks:**
- [ ] Add a small official-vector runner (`tools/ds4-v100-vector-check.py` or C equivalent) around `--dump-logprobs`.
- [ ] Automate the check for `short_reasoning_plain` and any other "cheap" official vectors.
- [ ] Add direct MXFP4 parity hardening tests in `tests/source_dtypes_smoke.c` to lock nibble ordering.
- [ ] Implement targeted source-oracle guard tests covering normal failure and diagnostic-only behavior.

### Phase 2: KV Budget & Admission

**Tasks:**
- [ ] Update `ds4_v100_context` to include KV budget calculations based on `DS4-V100-LAYOUT.md`.
- [ ] Implement the F16 KV admission check for SWA-only, Ratio-4, and Ratio-128 classes.
- [ ] Add a `v100_context_smoke` test case that validates admission for various context lengths (e.g., 4K, 32K, 256K).
- [ ] Expose KV budget diagnostics in the CLI.

### Phase 3: Prompt Prefill & HC State

**Tasks:**
- [ ] Refine `ds4.c` prefill logic to handle multi-token input and hidden-context (HC) state transitions.
- [ ] Ensure prefill path respects the source-layout oracle guard (i.e., remains CPU-only for now or uses guarded anchors).
- [ ] Add a prefill smoke test that records HC relay consistency across tokens.

### Phase 4: Device-Side Anchors (CUDA)

**Tasks:**
- [ ] Implement a V100 CUDA probe for F8_E4M3_B128 row-gather/expand.
- [ ] Implement a V100 CUDA probe for MXFP4 row-gather/expand (GGML layout).
- [ ] Integrate these probes into `tests/cuda_v100_context_smoke.c`.
- [ ] Validate bit-exact host-to-device-to-host expansion on the cluster.

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `tools/ds4-v100-vector-check.py` | Create | Automated harness for official-vector comparison |
| `ds4_v100_context.h` / `.c` | Modify | KV budget surface, admission logic, and prefill state |
| `ds4_cuda.cu` | Modify | First F8/MXFP4 device-side anchor kernels |
| `ds4.c` | Modify | Multi-token prefill logic and HC relay updates |
| `tests/source_dtypes_smoke.c` | Modify | Hardened MXFP4 nibble-ordering tests |
| `tests/v100_context_smoke.c` | Modify | KV admission and budget validation |
| `tests/cuda_v100_context_smoke.c` | Modify | Device-side F8/MXFP4 anchor validation |
| `docs/sprints/SPRINT-008-REPORT.md` | Create | Sprint verdict and evidence (at end of sprint) |

## Definition Of Done

- [ ] Official-vector oracle comparison is automated and passes for `short_reasoning_plain`.
- [ ] MXFP4 nibble ordering is hardened against regression in `source_dtypes_smoke`.
- [ ] V100 context correctly calculates and enforces F16 KV budgets for all layer classes.
- [ ] KV admission surface is validated for 4K and 32K context lengths.
- [ ] Prompt prefill logic handles multi-token inputs and updates HC state.
- [ ] At least one device-side anchor (F8 or MXFP4) is proven on the V100 cluster.
- [ ] Source-layout guard remains fail-closed for normal generation.
- [ ] Validation logs are archived under `docs/sprints/drafts/SPRINT-008-*`.
- [ ] `git diff --check` passes.

## Risks And Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| KV budget for 256K context exceeds VRAM | Medium | High | Re-verify `DS4-V100-LAYOUT.md`; adjust admission limits early. |
| Device-side MXFP4 expansion is slow | Low | Medium | This is a correctness anchor, not a throughput kernel; performance is deferred. |
| Official vectors diverge on long prefill | Medium | High | Use the CPU oracle as the source of truth; document any divergence. |
| Multi-GPU HC relay introduces race conditions | Low | High | Reuse Sprint 006 relay smoke patterns; keep streams explicit. |

## Security Considerations

- Source-layout guard must not be bypassed by the new prefill or KV logic.
- Diagnostic unlock must remain explicit and non-default.
- Ensure CUDA kernel bounds checks for device-side anchors.

## Dependencies

- Sprint 007 source-layout oracle and MXFP4 corrections.
- V100 cluster access for device-side anchors.
- `/models/DSv4-Flash-256e-fixed.gguf` on cluster scratch.

## Open Questions

1. Should the vector check harness be a Python script or a C tool integrated with `ds4_test`?
2. Which specific layers/experts are most critical for the first device-side MXFP4 anchor?
3. Is a 3 GiB reserve per GPU sufficient for the first prefill/KV implementation?
