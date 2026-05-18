---
sprint: 008
title: Source Oracle Harness And V100 KV Admission Anchors
status: active
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-008-INTENT.md
merge_notes: drafts/SPRINT-008-MERGE-NOTES.md
deferred: SPRINT-008-DEFERRED.md
---

# SPRINT-008: Source Oracle Harness And V100 KV Admission Anchors

## Overview

Sprint 007 proved a narrow but important correctness claim: the native
DeepSeek V4 Flash source layout can match the official
`short_reasoning_plain` first token under a guarded CPU oracle. It also fixed
MXFP4 row semantics and restored F16 KV as the source-layout correctness
baseline.

Sprint 008 turns that one-off proof into repeatable validation and adds the
first production-relevant V100 planning surfaces. The sprint deliberately stops
short of full V100 prefill/decode. It should leave the repo with automated
oracle checks, guard regression coverage, exact F16 KV admission by layer stage,
and one bounded CUDA source-format anchor that future production kernels can
compare against.

Normal source-layout generation remains fail-closed.

## Outcome Contract

- `SHIP`: source-oracle official-vector automation passes the selected token
  for `short_reasoning_plain`; source-layout guard tests pass; V100 F16 KV
  admission/reporting is derived from DS4 layer classes and tested at multiple
  context/slot tiers; MXFP4 parity hardening passes; one bounded CUDA
  source-format anchor passes on `sm_70`; logs and report are archived.
- `EXTEND`: oracle automation, guard tests, and V100 F16 KV admission ship, but
  the CUDA source-format anchor is blocked by cluster access, CUDA build
  issues, or a diagnosed device mismatch. The guard must remain intact.
- `STOP`: implementation requires unlocking normal source-layout serving,
  persistent dequantized large weights, host/SSD/offload as a success path,
  broad production kernels, MTP, multi-slot throughput, tensor parallelism, or
  changing the F16 KV correctness baseline.

## Non-Goals

- No normal source-layout generation unlock.
- No full source-layout V100 prompt prefill execution.
- No production FP8 dense GEMM or MXFP4 routed expert kernel implementation.
- No long-context performance benchmark.
- No multi-slot scheduling, wavefront scheduling, MTP, or speculative decode.
- No server/API deployment.
- No tensor-parallel exceptions.
- No persistent dequantized F16/F32 copies of large FP8/MXFP4 source tensors.
- No F8 KV baseline; F8 KV remains an optimization after F16 KV correctness.

## Planning Inputs

| File | Role |
|---|---|
| `docs/sprints/VISION.md` | North Star and sprint sequence |
| `docs/architecture/DS4-V100-LAYOUT.md` | V100 topology, dtype, and memory contract |
| `docs/sprints/SPRINT-007-REPORT.md` | Source oracle verdict and MXFP4/F16-KV corrections |
| `docs/sprints/SPRINT-007-DEFERRED.md` | Prior deferred prefill/KV/device-anchor work |
| `docs/sprints/SPRINT-007-FOLLOWUPS.md` | Oracle automation and parity hardening follow-ups |
| `ds4.c` | CPU oracle, KV state, compressed attention, session state |
| `ds4_source_formats.[ch]` | Source dtype reference helpers |
| `ds4_v100_context.[ch]` | V100 stage map and memory/reserve context |
| `ds4_cuda.cu`, `ds4_gpu.h` | CUDA arena and source/device probe surfaces |
| `tests/test-vectors/official.vec` | Official selected-token fixtures |

## Use Cases

1. **Automated source oracle**: a developer can run a repo-native command that
   verifies the source oracle against the official short vector without manual
   JSON inspection.
2. **Guard regression detection**: tests prove normal source-layout generation,
   non-CPU source oracle use, MTP sidecars, and sessions without diagnostic
   unlock still fail closed.
3. **V100 KV admission**: the V100 context can report F16 raw, compressed, and
   indexer KV bytes by stage for explicit context/slot settings.
4. **Context-tier planning**: 128K, 256K, 512K, and 1M context tiers have
   deterministic admission tests before runtime allocation work begins.
5. **Device source-format anchor**: a bounded CUDA test validates one packed
   source-format row against `ds4_source_formats` on V100-class hardware.

## Architecture

Sprint 008 adds three adjacent surfaces around the Sprint 007 oracle.

```text
tests/test-vectors/official.vec
    |
    v
source-oracle harness
CPU backend + source-layout oracle + diagnostic session unlock
    |
    +--> selected-token verdict
    +--> guard rejection coverage

DS4 compression schedule + stage map + ctx/slot inputs
    |
    v
V100 F16 KV admission
    |
    +--> raw SWA bytes
    +--> compressed attention KV bytes
    +--> ratio-4 indexer KV bytes
    +--> compressor state bytes
    +--> per-stage reserve verdict

synthetic packed source row
    |
    +--> CPU source helper reference
    +--> CUDA bounded device probe
    |
    v
source-format parity artifact
```

### Oracle Harness

Prefer a `tests/ds4_test.c` mode over a broad CLI feature. The test may require
an explicit source model path via environment variable or command option, so
normal local tests do not require the 145 GiB source model. The implementation
should reuse existing official-vector parsing where practical and should check
selected token equality for `short_reasoning_plain` as the release gate.

### KV Admission

`ds4_v100_context` currently accepts one coarse `planned_kv_bytes_per_gpu`.
Sprint 008 should replace or augment that with DS4-aware F16 KV accounting:

- raw SWA ring: `raw_cap * 512 * sizeof(F16)` per layer and slot;
- ratio-4 attention compressed KV: `(ctx / 4 + overhead) * 512 * sizeof(F16)`;
- ratio-4 indexer KV: `(ctx / 4 + overhead) * 128 * sizeof(F16)`;
- ratio-128 attention compressed KV: `(ctx / 128 + overhead) * 512 * sizeof(F16)`;
- compression-state reserve for attention and indexer state.

The exact helper names are implementation details. The report and tests must
show the layer class and per-stage totals, not only a single opaque KV number.

### Device Anchor

Use one bounded CUDA source-format probe. The preferred first anchor is
`F8_E4M3_B128` row decode or row-dot because dense attention/shared paths depend
on that source layout. MXFP4 receives additional CPU parity hardening in this
sprint, while full MXFP4 routed expert device kernels remain deferred.

The device anchor must be diagnostic only. It should read small synthetic rows
from device memory and compare against `ds4_source_formats`, not wire into
normal source-model generation.

## Implementation

### Phase 1: Oracle Automation And Guards

**Files:**
- `ds4.c`
- `ds4.h`
- `tests/ds4_test.c`
- `tests/test-vectors/README.md`
- `Makefile`

**Tasks:**
- [ ] Add a source-oracle official-vector test mode that opens the source model
      through CPU oracle settings and diagnostic session unlock.
- [ ] Verify selected-token equality for `short_reasoning_plain` without
      manual JSON inspection.
- [ ] Reuse existing official-vector parsing/top-logprob helpers where
      practical.
- [ ] Add or extend guard tests for normal source-layout rejection, non-CPU
      oracle rejection, MTP rejection, and missing diagnostic-session unlock.
- [ ] Document the command and source-model path requirement.

### Phase 2: V100 F16 KV Admission

**Files:**
- `ds4_v100_context.h`
- `ds4_v100_context.c`
- `tests/v100_context_smoke.c`
- `tests/cuda_v100_context_smoke.c`
- `tools/ds4-v100-context-smoke.c`

**Tasks:**
- [x] Add layer-class metadata: SWA-only, ratio-4/indexer, ratio-128.
- [x] Add F16 KV budget structures with raw, compressed attention, indexer, and
      compression-state fields.
- [x] Derive per-stage KV totals from `ctx_size`, `slots`, and layer ownership.
- [x] Add reserve/admission checks that fail closed when the KV plan would
      overfill a V100 stage.
- [x] Print stable context-smoke report fields for context, slots, layer class
      counts, and per-stage KV bytes.

### Phase 3: Source Dtype Parity Hardening

**Files:**
- `ds4_source_formats.[ch]`
- `tests/source_dtypes_smoke.c`

**Tasks:**
- [ ] Add direct MXFP4 regression coverage for GGML `block_mxfp4`
      low-half/high-half nibble ordering.
- [ ] Add malformed span/bounds tests for any source-format helper used by the
      CUDA anchor.
- [ ] Keep helper semantics shared by CPU oracle and diagnostic device anchors.

### Phase 4: CUDA Source-Format Anchor

**Files:**
- `ds4_gpu.h`
- `ds4_cuda.cu`
- `tests/cuda_source_dtypes_smoke.c`
- `Makefile`

**Tasks:**
- [ ] Add a bounded device API for one packed source-format row, preferably
      `F8_E4M3_B128`.
- [ ] Upload synthetic packed rows to a CUDA arena and execute the device probe.
- [ ] Compare device output against `ds4_source_formats` CPU helper results.
- [ ] Fail closed on row bounds, shape/span mismatch, or undersized output.
- [ ] Keep the anchor disconnected from normal decode/prefill paths.

### Phase 5: Validation And Report

**Files:**
- `docs/sprints/drafts/SPRINT-008-ORACLE.log`
- `docs/sprints/drafts/SPRINT-008-GUARD.log`
- `docs/sprints/drafts/SPRINT-008-KV-ADMISSION.log`
- `docs/sprints/drafts/SPRINT-008-CUDA-SOURCE.log`
- `docs/sprints/SPRINT-008-REPORT.md`
- `docs/sprints/SPRINT-008-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

**Tasks:**
- [ ] Run local build and model-less tests.
- [ ] Run the source oracle official-vector automation on the cluster against
      `/models/DSv4-Flash-256e-fixed.gguf`.
- [ ] Run CUDA source-format anchor on `sm_70` if cluster access is available.
- [ ] Archive command logs under `docs/sprints/drafts/`.
- [ ] Write the report with verdict, evidence, deviations, and Sprint 009
      handoff.
- [ ] Update `VISION.md` after the sprint verdict.

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `ds4.c` / `ds4.h` | Modify | Guarded source-oracle test access and shared metadata if needed |
| `tests/ds4_test.c` | Modify | Automated official-vector source oracle and guard checks |
| `ds4_v100_context.[ch]` | Modify | Layer-class and F16 KV admission/reporting surface |
| `tools/ds4-v100-context-smoke.c` | Modify | Print executable KV/admission plan |
| `tests/v100_context_smoke.c` | Modify | Model-less KV admission tests |
| `tests/cuda_v100_context_smoke.c` | Modify | Cluster topology plus KV admission tests |
| `ds4_source_formats.[ch]` | Modify | Source helper coverage needed by anchors |
| `tests/source_dtypes_smoke.c` | Modify | MXFP4 parity and source dtype bounds tests |
| `ds4_gpu.h` / `ds4_cuda.cu` | Modify | Bounded CUDA source-format probe |
| `tests/cuda_source_dtypes_smoke.c` | Create | CUDA source dtype anchor test |
| `Makefile` | Modify | Build new/updated test targets |
| `docs/sprints/SPRINT-008-REPORT.md` | Create | Sprint verdict and evidence |
| `docs/sprints/VISION.md` | Modify | Record outcome and sequence adjustment |

## Definition Of Done

- [ ] Automated source-oracle official-vector validation exists and verifies
      selected-token equality for `short_reasoning_plain`.
- [ ] Source-layout guard tests cover normal rejection, non-CPU oracle
      rejection, MTP rejection, and missing diagnostic-session unlock.
- [ ] V100 context reports F16 raw/compressed/indexer KV bytes by layer class,
      stage, context, and slot count.
- [ ] Model-less tests cover layer class counts, KV byte math, at least one
      admitted context/slot tier, and at least one over-budget rejection.
- [ ] MXFP4 source-format tests lock in the corrected GGML block ordering.
- [ ] One bounded CUDA source-format anchor passes on V100-class `sm_70`, or the
      sprint records `EXTEND` with the exact blocker.
- [ ] Normal source-layout serving remains fail-closed.
- [ ] No persistent dequantized large source tensors are introduced.
- [ ] No host/SSD/offload path is treated as a successful production path.
- [ ] Validation logs are archived under `docs/sprints/drafts/`.
- [ ] `git diff --check` passes.
- [ ] `SPRINT-008-REPORT.md` and `VISION.md` are updated after execution.

## Risks And Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| Scope drifts into full V100 prefill/decode | High | High | Keep prefill execution a non-goal; release-gate on oracle automation, KV admission, and one anchor |
| KV byte math diverges from runtime semantics | Medium | High | Test layer classes and context tiers; keep formulas explicit and tied to DS4 constants |
| Device anchor creates false confidence | Medium | Medium | Report it as source-format parity only, not as production kernel readiness |
| Guard automation accidentally widens source oracle | Medium | High | Add rejection tests and avoid public serving paths |
| Cluster validation blocked | Medium | Medium | Ship local/model-less pieces as `EXTEND` only; archive exact blocker |
| F8 anchor is less useful than MXFP4 for MoE | Medium | Medium | Harden MXFP4 in CPU tests now; defer routed expert device kernels to next production-kernel sprint |

## Security Considerations

- The source oracle remains diagnostic-only and CPU-only.
- Normal server/CLI serving paths must not accept the native source model.
- Device probes must bounds-check arena offsets, row counts, row spans, and
  output sizes before reading.
- Validation logs should store verdicts and small values, not broad raw weight
  dumps.

## Dependencies

- Sprint 007 `SHIP` result and source oracle report.
- Corrected MXFP4 source-format helper semantics.
- Existing official-vector fixtures.
- Existing V100 context and CUDA arena/probe code.
- V100 cluster access for final CUDA/source-model evidence.

## Open Questions

1. Should the source-oracle automation live only in `ds4_test`, or should a thin
   diagnostic tool wrap `--dump-logprobs` for operator use?
2. Should KV byte formulas be factored into a shared internal helper or kept in
   `ds4_v100_context` with tests against DS4 constants?
3. Is `F8_E4M3_B128` the right first CUDA anchor, or should MXFP4 become the
   release-gated anchor if dense FP8 proves too narrow?
4. Which additional official vectors are cheap enough to add after
   `short_reasoning_plain` without turning validation into a long cluster run?
