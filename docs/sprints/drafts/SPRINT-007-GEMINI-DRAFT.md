---
sprint: 007
title: Source-Layout Single-Slot Decode Correctness
status: draft
date: 2026-05-17
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-007-INTENT.md
---

# SPRINT-007: Source-Layout Single-Slot Decode Correctness

## Overview

Sprint 006 established the V100-specific sidecar execution context, descriptor policy, and no-math layer skeleton. Sprint 007 is the correctness gate between that skeleton and future performance/deployment work. The goal is to establish a guarded, one-slot decode path that can correctly process the source-quantized model layout and compare its outputs against trusted reference vectors.

V100 lacks native tensor-core support for BF16, FP8, and MXFP4. This sprint reinforces that policy by implementing source-dtype dequantization and scalar/row primitives as a "correctness oracle" in the CPU reference path. These primitives will enable a diagnostic decode mode that bypasses the normal source-layout guard only for explicit validation tasks.

This sprint does not optimize for throughput, implement prefill, or enable general-purpose generation for the source layout. It focuses strictly on the first-token correctness of the narrow V100 appliance path.

## Use Cases

1. **Source-layout scalar/row diagnostics**: An engineer can use new CPU/reference helpers to dequantize individual scalars or rows of BF16, F8_E4M3_B128, and MXFP4 source tensors for equality checks.
2. **Corrected source embedding**: The runtime correctly handles BF16 source embeddings without mislabeling them as F16 or failing at the descriptor boundary.
3. **Guarded diagnostic decode**: An operator can trigger a single-slot, one-token decode run for a source-layout model using a visible diagnostic flag, while normal generation remains safely guarded.
4. **Official vector comparison**: The diagnostic path can run official API prompts and verify that the generated top-logprobs or tokens match the membership in `tests/test-vectors/official.vec`.
5. **Fail-closed format validation**: The engine explicitly rejects unsupported source-layout variants (e.g., unknown MXFP4 scale semantics) with a precise error rather than drifting or crashing.

## Architecture

Sprint 007 extends the CPU reference path in `ds4.c` and adds source-dtype primitives to `gguf-tools/quants.c`. It utilizes the `ds4_v100_context` descriptors from Sprint 006 to bind source tensors to the oracle path.

```text
GGUF (Source Layout)
    |
    v
ds4_engine_open (Normal Guard: FAIL)
    |
    v
Diagnostic Mode Hook (--verify-source-correctness)
    |
    +--> ds4_v100_context_open (Sprint 006)
    +--> Source Dtype Row Helpers (BF16, FP8, MXFP4)
    +--> CPU Reference Graph (Single-Slot Decode)
    |
    v
Logit/Token Comparison
    |
    +--> tests/test-vectors/official.vec
    |
    v
Correctness Report (PASS/FAIL/BLOCK)
```

### Source Dtype Primitives

The correctness oracle requires reference implementations for:

- **BF16**: Correct conversion to/from FP32/FP16.
- **F8_E4M3_B128**: Dequantization logic matching DeepSeek V4 Flash source semantics (likely E8M0 scale interpretation).
- **MXFP4**: Dequantization logic for the 4-bit block-scaled format, including scale-factor semantics.

### Diagnostic Decode Path

A new guarded entry point `ds4_verify_source_layout_correctness()` will:
1. Force-open a source-layout model bypassing the `ds4_engine_open` guard.
2. Initialize the `ds4_v100_context` to resolve descriptors.
3. Execute a single-token decode using the CPU reference path, injecting source-aware row-gather and matvec helpers.
4. Compare against official vectors.

## Implementation

### Phase 1: Source Dtype Reference Primitives

**Files:**
- `gguf-tools/quants.h`
- `gguf-tools/quants.c`
- `tests/ds4_test.c`

**Tasks:**
- [ ] Implement `dequantize_f8_e4m3_b128` scalar and row helpers.
- [ ] Implement `dequantize_mxfp4` scalar and row helpers.
- [ ] Verify BF16 conversion helpers are robust and used correctly in `ds4.c`.
- [ ] Add model-less unit tests for each primitive using synthetic byte fixtures.
- [ ] Document any discovered scale-factor semantics (e.g., E8M0 for FP8).

### Phase 2: Source-Layout Embedding and Control Cleanup

**Files:**
- `ds4.c`
- `ds4.h`

**Tasks:**
- [ ] Correct `embed_token` to handle BF16 source embeddings properly.
- [ ] Update HC/router/output control matvecs to use BF16 or FP32 where required by the source layout.
- [ ] Ensure `output_logits_one` handles the BF16 output head correctly.
- [ ] Validate that embedding/control tensors bind correctly to Sprint 006 descriptors.

### Phase 3: Guarded Diagnostic Mode

**Files:**
- `ds4.c`
- `ds4_cli.c`
- `ds4_v100_context.h`

**Tasks:**
- [ ] Implement `ds4_verify_source_layout_correctness()` entry point.
- [ ] Add a diagnostic-only CLI flag (e.g., `--verify-source`) that enables this path.
- [ ] Ensure the normal `ds4_engine_open` guard remains active and fail-closed for standard generation.
- [ ] Link the diagnostic path to the Sprint 006 context for tensor lookup.

### Phase 4: Official Vector Comparison

**Files:**
- `ds4.c`
- `tests/test-vectors/`

**Tasks:**
- [ ] Implement a logprob/token comparison helper that reads `tests/test-vectors/official.vec`.
- [ ] Run the diagnostic decode path for official prompts.
- [ ] Compare generated top-k tokens/logprobs against official fixtures.
- [ ] Report precision drift or membership mismatches.

### Phase 5: Cluster Validation and Reporting

**Files:**
- `docs/sprints/drafts/SPRINT-007-CORRECTNESS.log`
- `docs/sprints/drafts/SPRINT-007-GUARD.log`
- `docs/sprints/SPRINT-007-REPORT.md`

**Tasks:**
- [ ] Run synthetic primitive tests on the V100 cluster.
- [ ] Run the diagnostic decode against `/models/DSv4-Flash-256e-fixed.gguf` on the V100 cluster.
- [ ] Archive proof of guard rejection for normal generation.
- [ ] Archive official vector comparison results.
- [ ] Close out with a `SHIP`, `EXTEND`, or `STOP` (blocker report) verdict.

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `gguf-tools/quants.h` / `quants.c` | Modify | Add reference dequant primitives for FP8 and MXFP4 |
| `ds4.c` | Modify | Correct BF16 embedding/control, add diagnostic mode, reference decode |
| `ds4.h` | Modify | Public diagnostic entry point |
| `ds4_cli.c` | Modify | Add `--verify-source` flag |
| `tests/ds4_test.c` | Modify | Unit tests for new source-dtype primitives |
| `docs/sprints/SPRINT-007-REPORT.md` | Create | Final result and validation summary |
| `docs/sprints/drafts/SPRINT-007-*.log` | Create | Diagnostic and guard validation artifacts |

## Definition Of Done

- [ ] Reference dequantization helpers for BF16, F8_E4M3_B128, and MXFP4 exist and are unit-tested.
- [ ] Source-layout embedding, HC control, and output head use correct BF16/F32 types.
- [ ] A guarded diagnostic mode exists and bypasses the source guard only for explicit verification.
- [ ] Official vector comparison (top-k tokens/logprobs) is performed against `official.vec`.
- [ ] Normal source-layout generation remains guarded and fail-closed.
- [ ] A precise blocker report is produced if FP8/MXFP4 semantics cannot be established.
- [ ] Cluster validation logs are archived under `docs/sprints/drafts/`.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| MXFP4/FP8 scale semantics are ambiguous | High | High | Implement exact reference kernels and document semantics early |
| Official vector slices are too sparse for verification | Medium | Medium | Use top-k token membership if logprob drift is too high |
| CPU reference path is too slow for full source run | Medium | Medium | Limit to single-token, small-context diagnostic runs |
| Diagnostic mode leaks into production path | Low | High | Use strict opt-in flags and keep normal guard fail-closed |

## Security

- The diagnostic mode must be opt-in and not exposed to network/server APIs.
- Continue to treat GGUF and pack indices as untrusted inputs.
- Validate all descriptor bindings and arena spans before dequantization.
- Do not introduce persistent dequantized weight buffers.

## Dependencies

- Sprint 006 `ds4_v100_context` and descriptor policy.
- Official API vectors in `tests/test-vectors/official.vec`.
- V100 cluster for full-model source-layout diagnostic runs.

## Open Questions

1. Are MXFP4 scales stored as E8M0 or a different block-wise format in the source GGUF?
2. Is token membership in the official top-k sufficient for a "correctness" pass, or is exact logprob epsilon required?
3. Should the diagnostic path support multi-token generation, or stop at the first token for this sprint?
4. How should the engine handle "fixed" vs "original" source GGUF differences if they impact dequantization semantics?
